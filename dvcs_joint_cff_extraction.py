#!/usr/bin/env python3
"""Simultaneous BKM10 extraction of effective ReH and ImH from DVCS surrogates.

This script consumes the common-support surrogate table produced by the
cross-section and beam-helicity-difference workflow.  It performs three linked
operations:

1. Define a strict, auditable common interpolation/reliability domain.
2. Use local simultaneous BKM10 fits to test extractability and initialize the
   global model.
3. Train a smooth DNN CFF surface

       (Q2, xB, t) -> (ReH, ImH)

   through the BKM10 observable layer against both surrogate observables at
   once.

The first production extraction deliberately uses an H-dominance parameterization:
all E, Htilde, and Etilde CFF components are set to zero.  This is not because
those CFFs are physically absent.  It is because the local weighted Jacobian is
well-conditioned for (ReH, ImH) but becomes strongly ill-conditioned when E or
Htilde components are freed with only sigma_UU and Delta-sigma_LU.  The output
therefore represents effective H-dominance CFFs.

The BKM observable is evaluated exactly as a quadratic polynomial in the eight
CFF components.  The polynomial coefficients are generated numerically from the
full BKM10 functions and are verified against direct evaluations.

Typical use
-----------
python dvcs_joint_cff_extraction.py \
    --predictions xsec_diff_surrogate/predictions_with_pulls.csv \
    --bkm-module bkm10_observables_corrected.py \
    --outdir joint_cff_extraction

Outputs are intentionally compact: one self-contained HTML report plus CSV,
JSON, and PyTorch artifacts.  No individual plot files or zip archive are made.
"""
from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import io
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CFF_NAMES = ["ReH", "ReHt", "ReE", "ReEt", "ImH", "ImHt", "ImE", "ImEt"]
H_NAMES = ["ReH", "ImH"]
H_INDICES = np.array([0, 4], dtype=int)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("bkm10_observable_layer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load BKM module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_columns() -> List[str]:
    return [
        "set", "k", "q_squared", "x_b", "t", "phi", "experiment",
        "unp_beam_unp_target_xsec", "unp_beam_unp_target_xsec_err",
        "xsec_diff", "xsec_diff_err",
        "model_xsec", "model_xsec_diff",
    ]


def build_bkm_polynomial(df: pd.DataFrame, bkm) -> Dict[str, np.ndarray]:
    """Build exact constant, linear, and quadratic BKM coefficients.

    For either observable O and CFF vector c,

        O(c) = O0 + L_i c_i + c_i Q_ij c_j.

    BH supplies O0, interference supplies the linear term, and DVCS-squared
    supplies the quadratic term.  Forty-five direct evaluations reconstruct the
    complete eight-CFF polynomial for every row.
    """
    args = bkm.prepare_bkm10_kinematics(
        df["k"].to_numpy(float),
        df["q_squared"].to_numpy(float),
        df["x_b"].to_numpy(float),
        df["t"].to_numpy(float),
        df["phi"].to_numpy(float),
    )

    def evaluate(cff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = [float(cff[i]) for i in range(8)]
        xsec = np.asarray(bkm.bkm10_cross_section(0.0, 0.0, *args, *c), dtype=float)
        plus = np.asarray(bkm.bkm10_cross_section(+1.0, 0.0, *args, *c), dtype=float)
        minus = np.asarray(bkm.bkm10_cross_section(-1.0, 0.0, *args, *c), dtype=float)
        return xsec, 0.5 * (plus - minus)

    n, p = len(df), 8
    c0_x, c0_d = evaluate(np.zeros(p))
    if np.any(~np.isfinite(c0_x)) or np.any(c0_x <= 0.0):
        raise ValueError("Corrected BKM preparation produced nonpositive/nonfinite BH cross sections")

    l_x = np.empty((n, p), dtype=float)
    l_d = np.empty((n, p), dtype=float)
    q_x = np.zeros((n, p, p), dtype=float)
    q_d = np.zeros((n, p, p), dtype=float)
    unit_x: List[np.ndarray] = []
    unit_d: List[np.ndarray] = []

    for i in range(p):
        cp = np.zeros(p); cp[i] = 1.0
        cm = np.zeros(p); cm[i] = -1.0
        fp_x, fp_d = evaluate(cp)
        fm_x, fm_d = evaluate(cm)
        l_x[:, i] = 0.5 * (fp_x - fm_x)
        l_d[:, i] = 0.5 * (fp_d - fm_d)
        q_x[:, i, i] = 0.5 * (fp_x + fm_x) - c0_x
        q_d[:, i, i] = 0.5 * (fp_d + fm_d) - c0_d
        unit_x.append(fp_x)
        unit_d.append(fp_d)

    for i in range(p):
        for j in range(i + 1, p):
            c = np.zeros(p); c[i] = c[j] = 1.0
            f_x, f_d = evaluate(c)
            # In c^T Q c the off-diagonal contribution is 2 Q_ij c_i c_j.
            qij_x = 0.5 * (f_x - unit_x[i] - unit_x[j] + c0_x)
            qij_d = 0.5 * (f_d - unit_d[i] - unit_d[j] + c0_d)
            q_x[:, i, j] = q_x[:, j, i] = qij_x
            q_d[:, i, j] = q_d[:, j, i] = qij_d

    # Exactness check on a few deterministic CFF vectors.
    rng = np.random.default_rng(9157)
    max_err_x = 0.0
    max_err_d = 0.0
    for _ in range(3):
        c = rng.normal(0.0, 3.0, p)
        direct_x, direct_d = evaluate(c)
        poly_x = c0_x + l_x @ c + np.einsum("i,nij,j->n", c, q_x, c)
        poly_d = c0_d + l_d @ c + np.einsum("i,nij,j->n", c, q_d, c)
        max_err_x = max(max_err_x, float(np.max(np.abs(direct_x - poly_x))))
        max_err_d = max(max_err_d, float(np.max(np.abs(direct_d - poly_d))))
    if max_err_x > 1e-8 or max_err_d > 1e-8:
        raise RuntimeError(f"BKM polynomial reconstruction failed: {max_err_x=}, {max_err_d=}")

    return {
        "c0_xsec": c0_x,
        "L_xsec": l_x,
        "Q_xsec": q_x,
        "c0_diff": c0_d,
        "L_diff": l_d,
        "Q_diff": q_d,
        "max_polynomial_error_xsec": np.array(max_err_x),
        "max_polynomial_error_diff": np.array(max_err_d),
    }


def circular_max_gap(phi: np.ndarray) -> float:
    phi = np.sort(np.asarray(phi, dtype=float))
    if len(phi) < 2:
        return 2.0 * np.pi
    return float(np.max(np.diff(np.r_[phi, phi[0] + 2.0 * np.pi])))


def build_interpolation_audit(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for set_id, g in df.groupby("set", sort=True):
        pull_x = (g["model_xsec"] - g["unp_beam_unp_target_xsec"]) / g["unp_beam_unp_target_xsec_err"]
        pull_d = (g["model_xsec_diff"] - g["xsec_diff"]) / g["xsec_diff_err"]
        phi = g["phi"].to_numpy(float)
        xchi = float(np.mean(np.square(pull_x)))
        dchi = float(np.mean(np.square(pull_d)))
        gap_deg = float(np.degrees(circular_max_gap(phi)))
        both_signs = bool(np.any(phi < 0.0) and np.any(phi > 0.0))
        enough_points = len(g) >= args.min_common_points
        good_x = xchi <= args.max_surrogate_chi2
        good_d = dchi <= args.max_surrogate_chi2
        good_gap = gap_deg <= args.max_phi_gap_deg
        selected = enough_points and both_signs and good_gap and good_x and good_d
        reasons = []
        if not enough_points: reasons.append("too_few_common_phi_points")
        if not both_signs: reasons.append("does_not_cover_both_phi_signs")
        if not good_gap: reasons.append("angular_gap_too_large")
        if not good_x: reasons.append("xsec_surrogate_closure")
        if not good_d: reasons.append("diff_surrogate_closure")
        rows.append({
            "set": int(set_id),
            "experiment": str(g["experiment"].iloc[0]),
            "n_common": int(len(g)),
            "k": float(g["k"].mean()),
            "q_squared": float(g["q_squared"].mean()),
            "x_b": float(g["x_b"].mean()),
            "t": float(g["t"].mean()),
            "xsec_surrogate_chi2": xchi,
            "diff_surrogate_chi2": dchi,
            "max_phi_gap_deg": gap_deg,
            "both_phi_signs": both_signs,
            "interpolation_selected": selected,
            "interpolation_exclusion_reason": ";".join(reasons),
        })
    return pd.DataFrame(rows)


def polynomial_observable(c: np.ndarray, c0: np.ndarray, l: np.ndarray, q: np.ndarray) -> np.ndarray:
    return c0 + l @ c + np.einsum("i,nij,j->n", c, q, c)


def local_h_fit(df: pd.DataFrame, poly: Dict[str, np.ndarray], row_indices: np.ndarray) -> Dict[str, float]:
    """Fit one ReH/ImH pair to one fixed-kinematics set."""
    ix = H_INDICES
    g = df.iloc[row_indices]
    c0x = poly["c0_xsec"][row_indices]
    lx = poly["L_xsec"][row_indices][:, ix]
    qx = poly["Q_xsec"][row_indices][:, ix][:, :, ix]
    c0d = poly["c0_diff"][row_indices]
    ld = poly["L_diff"][row_indices][:, ix]
    qd = poly["Q_diff"][row_indices][:, ix][:, :, ix]
    tx = g["model_xsec"].to_numpy(float)
    td = g["model_xsec_diff"].to_numpy(float)
    ex = g["unp_beam_unp_target_xsec_err"].to_numpy(float)
    ed = g["xsec_diff_err"].to_numpy(float)

    def residual(c):
        px = polynomial_observable(c, c0x, lx, qx)
        pdiff = polynomial_observable(c, c0d, ld, qd)
        return np.r_[(px - tx) / ex, (pdiff - td) / ed]

    def jacobian(c):
        jx = (lx + 2.0 * np.einsum("nij,j->ni", qx, c)) / ex[:, None]
        jd = (ld + 2.0 * np.einsum("nij,j->ni", qd, c)) / ed[:, None]
        return np.vstack([jx, jd])

    starts = [np.zeros(2), np.array([-3.0, 3.0]), np.array([-1.0, 5.0]), np.array([-5.0, 5.0])]
    if "ReH" in g and "ImH" in g:
        starts.append(np.clip(np.array([g["ReH"].median(), g["ImH"].median()]), -29.0, 29.0))
    fits = [
        least_squares(
            residual, start, jac=jacobian, bounds=(-30.0, 30.0),
            max_nfev=1200, xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        for start in starts
    ]
    fit = min(fits, key=lambda item: float(np.sum(item.fun * item.fun)))
    px = polynomial_observable(fit.x, c0x, lx, qx)
    pdiff = polynomial_observable(fit.x, c0d, ld, qd)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    condition = float(singular[0] / singular[-1])
    dof = max(1, 2 * len(g) - 2)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * float(np.sum(fit.fun**2) / dof)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    corr = covariance[0, 1] / (errors[0] * errors[1]) if np.all(errors > 0.0) else np.nan
    return {
        "ReH_local": float(fit.x[0]),
        "ImH_local": float(fit.x[1]),
        "ReH_local_curvature_err": float(errors[0]),
        "ImH_local_curvature_err": float(errors[1]),
        "local_corr_ReH_ImH": float(corr),
        "local_bkm_chi2": float(np.mean(fit.fun**2)),
        "local_bkm_chi2_xsec": float(np.mean(np.square((px - tx) / ex))),
        "local_bkm_chi2_diff": float(np.mean(np.square((pdiff - td) / ed))),
        "local_jac_condition": condition,
    }


def identifiability_rows(
    df: pd.DataFrame,
    poly: Dict[str, np.ndarray],
    local: pd.DataFrame,
) -> pd.DataFrame:
    """Condition numbers for increasingly large local CFF parameterizations."""
    models = {
        "H2": ["ReH", "ImH"],
        "H_plus_E4": ["ReH", "ImH", "ReE", "ImE"],
        "H_plus_Htilde4": ["ReH", "ImH", "ReHt", "ImHt"],
        "H_plus_E_plus_Htilde6": ["ReH", "ImH", "ReE", "ImE", "ReHt", "ImHt"],
    }
    name_to_index = {name: i for i, name in enumerate(CFF_NAMES)}
    output = []
    for _, row in local.iterrows():
        set_rows = np.flatnonzero(df["set"].to_numpy() == int(row["set"]))
        g = df.iloc[set_rows]
        c_full = np.zeros(8)
        c_full[0] = row["ReH_local"]
        c_full[4] = row["ImH_local"]
        for model_name, names in models.items():
            indices = np.array([name_to_index[name] for name in names], dtype=int)
            jx = (
                poly["L_xsec"][set_rows][:, indices]
                + 2.0 * np.einsum(
                    "nij,j->ni", poly["Q_xsec"][set_rows][:, indices, :], c_full
                )
            ) / g["unp_beam_unp_target_xsec_err"].to_numpy()[:, None]
            jd = (
                poly["L_diff"][set_rows][:, indices]
                + 2.0 * np.einsum(
                    "nij,j->ni", poly["Q_diff"][set_rows][:, indices, :], c_full
                )
            ) / g["xsec_diff_err"].to_numpy()[:, None]
            singular = np.linalg.svd(np.vstack([jx, jd]), compute_uv=False)
            condition = np.inf if singular[-1] <= 1e-15 else float(singular[0] / singular[-1])
            output.append({
                "set": int(row["set"]),
                "model": model_name,
                "n_parameters": len(names),
                "jac_condition": condition,
                "smallest_singular_value": float(singular[-1]),
                "largest_singular_value": float(singular[0]),
            })
    return pd.DataFrame(output)


def cff_features(q2: np.ndarray, xb: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Smooth, dimensionless CFF-network coordinates."""
    q2 = np.asarray(q2, dtype=float)
    xb = np.asarray(xb, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.any(q2 <= 0.0) or np.any((xb <= 0.0) | (xb >= 1.0)) or np.any(t >= 0.0):
        raise ValueError("CFF features require Q2>0, 0<xB<1, and t<0")
    return np.column_stack([
        np.log(q2),
        np.log(xb / (1.0 - xb)),
        np.log(-t),
    ]).astype(np.float32)


class CFFNet(nn.Module):
    """Small smooth network for the effective H-dominance CFF pair."""
    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        output_mean: np.ndarray,
        output_std: np.ndarray,
        hidden: int,
        depth: int,
        output_limit: float = 6.0,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.output_limit = float(output_limit)
        self.register_buffer("x_mean", torch.tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(feature_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(output_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(output_std, dtype=torch.float32))
        layers: List[nn.Module] = []
        for i in range(depth):
            layers.append(nn.Linear(3 if i == 0 else hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, features_raw: torch.Tensor) -> torch.Tensor:
        z = self.net((features_raw - self.x_mean) / self.x_std)
        if self.output_limit > 0.0:
            z = self.output_limit * torch.tanh(z / self.output_limit)
        return self.y_mean + self.y_std * z


def group_mean(values: torch.Tensor, index: torch.Tensor, n_groups: int) -> torch.Tensor:
    sums = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    sums.index_add_(0, index, values.reshape(-1))
    counts = torch.bincount(index, minlength=n_groups).to(values.dtype)
    return sums / counts.clamp_min(1.0)


def parse_stages(text: str) -> List[Tuple[int, float]]:
    stages = []
    for item in text.split(","):
        epochs, lr = item.split(":", 1)
        stages.append((int(epochs), float(lr)))
    return stages


def train_seed(
    seed: int,
    args,
    train_df: pd.DataFrame,
    local_train: pd.DataFrame,
    row_indices: np.ndarray,
    poly: Dict[str, np.ndarray],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
) -> Tuple[CFFNet, List[Dict[str, float]], Dict[str, float]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CFFNet(
        feature_mean, feature_std, output_mean, output_std,
        hidden=args.hidden, depth=args.depth, output_limit=args.output_limit,
    )

    x_raw = torch.tensor(
        cff_features(train_df["q_squared"], train_df["x_b"], train_df["t"]),
        dtype=torch.float32,
    )
    x_local = torch.tensor(
        cff_features(local_train["q_squared"], local_train["x_b"], local_train["t"]),
        dtype=torch.float32,
    )
    y_local = torch.tensor(local_train[["ReH_local", "ImH_local"]].to_numpy(np.float32))
    y_scale = torch.tensor(output_std, dtype=torch.float32)

    ix = H_INDICES
    tensors = [
        poly["c0_xsec"][row_indices].astype(np.float32),
        poly["L_xsec"][row_indices][:, ix].astype(np.float32),
        poly["Q_xsec"][row_indices][:, ix][:, :, ix].astype(np.float32),
        poly["c0_diff"][row_indices].astype(np.float32),
        poly["L_diff"][row_indices][:, ix].astype(np.float32),
        poly["Q_diff"][row_indices][:, ix][:, :, ix].astype(np.float32),
        train_df["model_xsec"].to_numpy(np.float32),
        train_df["model_xsec_diff"].to_numpy(np.float32),
        train_df["unp_beam_unp_target_xsec_err"].to_numpy(np.float32),
        train_df["xsec_diff_err"].to_numpy(np.float32),
    ]
    c0x, lx, qx, c0d, ld, qd, tx, td, ex, ed = [torch.tensor(item) for item in tensors]

    set_codes, set_labels = pd.factorize(train_df["set"], sort=True)
    experiment_codes, experiment_labels = pd.factorize(train_df["experiment"], sort=True)
    set_to_experiment = np.array([
        experiment_codes[np.flatnonzero(set_codes == i)[0]] for i in range(len(set_labels))
    ], dtype=np.int64)
    set_index = torch.tensor(set_codes, dtype=torch.long)
    set_experiment_index = torch.tensor(set_to_experiment, dtype=torch.long)

    def observable(cff: torch.Tensor, c0: torch.Tensor, linear: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
        return c0 + torch.sum(linear * cff, dim=1) + torch.einsum("ni,nij,nj->n", cff, quad, cff)

    def physics_losses():
        cff = model(x_raw)
        pred_x = observable(cff, c0x, lx, qx)
        pred_d = observable(cff, c0d, ld, qd)
        pull_x = (pred_x - tx) / ex
        pull_d = (pred_d - td) / ed
        set_x = group_mean(pull_x * pull_x, set_index, len(set_labels))
        set_d = group_mean(pull_d * pull_d, set_index, len(set_labels))
        exp_x = group_mean(set_x, set_experiment_index, len(experiment_labels)).mean()
        exp_d = group_mean(set_d, set_experiment_index, len(experiment_labels)).mean()
        return 0.5 * (exp_x + exp_d), exp_x, exp_d, cff, pred_x, pred_d

    history: List[Dict[str, float]] = []
    start_time = time.time()

    # Local fits select the correct quadratic branch and provide a rapid, stable
    # initialization.  They are not used as final CFF data.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.pretrain_epochs + 1):
        optimizer.zero_grad()
        pred_local = model(x_local)
        loss = torch.mean(torch.square((pred_local - y_local) / y_scale))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
    history.append({
        "seed": seed,
        "stage": "local_pretrain",
        "epochs": args.pretrain_epochs,
        "learning_rate": args.pretrain_lr,
        "loss": float(loss.detach()),
        "elapsed_sec": time.time() - start_time,
    })

    best_state = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    for stage_number, (epochs, lr) in enumerate(parse_stages(args.physics_stages), start=1):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss, loss_x, loss_d, _, _, _ = physics_losses()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        with torch.no_grad():
            loss, loss_x, loss_d, _, _, _ = physics_losses()
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "seed": seed,
            "stage": f"physics_{stage_number}",
            "epochs": epochs,
            "learning_rate": lr,
            "balanced_loss": float(loss),
            "balanced_xsec_loss": float(loss_x),
            "balanced_diff_loss": float(loss_d),
            "elapsed_sec": time.time() - start_time,
        })

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        loss, loss_x, loss_d, cff, pred_x, pred_d = physics_losses()
        pull_x = (pred_x - tx) / ex
        pull_d = (pred_d - td) / ed
    metrics = {
        "seed": seed,
        "balanced_loss": float(loss),
        "balanced_xsec_loss": float(loss_x),
        "balanced_diff_loss": float(loss_d),
        "point_mean_chi2": float(torch.mean(torch.cat([pull_x * pull_x, pull_d * pull_d]))),
        "point_xsec_chi2": float(torch.mean(pull_x * pull_x)),
        "point_diff_chi2": float(torch.mean(pull_d * pull_d)),
    }
    return model, history, metrics


def predict_cff(model: CFFNet, q2, xb, t) -> np.ndarray:
    model.eval()
    features = torch.tensor(cff_features(q2, xb, t), dtype=torch.float32)
    with torch.no_grad():
        return model(features).cpu().numpy()


def figure_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_report(
    outdir: Path,
    args,
    audit: pd.DataFrame,
    local: pd.DataFrame,
    ident: pd.DataFrame,
    predictions: pd.DataFrame,
    cff_sets: pd.DataFrame,
    metrics: Dict,
) -> None:
    images: List[Tuple[str, str]] = []

    # Reliability-domain phase-space view.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    excluded = audit[~audit["extraction_selected"]]
    selected = audit[audit["extraction_selected"]]
    axes[0].scatter(excluded["x_b"], excluded["q_squared"], s=14, alpha=0.25, label="excluded")
    sc = axes[0].scatter(selected["x_b"], selected["q_squared"], c=-selected["t"], s=28, label="selected")
    axes[0].set_xlabel(r"$x_B$")
    axes[0].set_ylabel(r"$Q^2\,[\mathrm{GeV}^2]$")
    axes[0].set_title("Strict common interpolation/extraction domain")
    axes[0].legend(fontsize=8)
    fig.colorbar(sc, ax=axes[0], label=r"$-t\,[\mathrm{GeV}^2]$")
    axes[1].scatter(excluded["x_b"], -excluded["t"], s=14, alpha=0.25)
    sc2 = axes[1].scatter(selected["x_b"], -selected["t"], c=selected["q_squared"], s=28)
    axes[1].set_xlabel(r"$x_B$")
    axes[1].set_ylabel(r"$-t\,[\mathrm{GeV}^2]$")
    axes[1].set_title("Selected support in transverse momentum transfer")
    fig.colorbar(sc2, ax=axes[1], label=r"$Q^2\,[\mathrm{GeV}^2]$")
    fig.tight_layout()
    images.append(("Common reliability domain", figure_base64(fig)))

    # Extracted CFF surfaces as irregular point clouds.
    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    p1 = ax1.scatter(cff_sets["x_b"], -cff_sets["t"], cff_sets["ReH"], c=cff_sets["q_squared"], s=25)
    p2 = ax2.scatter(cff_sets["x_b"], -cff_sets["t"], cff_sets["ImH"], c=cff_sets["q_squared"], s=25)
    for ax, zlabel, title in [(ax1, r"$\mathrm{Re}\,\mathcal{H}$", "Effective Re H surface"), (ax2, r"$\mathrm{Im}\,\mathcal{H}$", "Effective Im H surface")]:
        ax.set_xlabel(r"$x_B$")
        ax.set_ylabel(r"$-t\,[\mathrm{GeV}^2]$")
        ax.set_zlabel(zlabel)
        ax.set_title(title)
    fig.colorbar(p2, ax=[ax1, ax2], shrink=0.72, label=r"$Q^2\,[\mathrm{GeV}^2]$")
    images.append(("Extracted H-dominance CFF point cloud", figure_base64(fig)))

    # Global-vs-local comparison, with seed variability.
    merged = cff_sets.merge(local[["set", "ReH_local", "ImH_local"]], on="set", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].errorbar(merged["ReH_local"], merged["ReH"], yerr=merged["ReH_algorithmic_std"], fmt=".", alpha=0.7)
    axes[1].errorbar(merged["ImH_local"], merged["ImH"], yerr=merged["ImH_algorithmic_std"], fmt=".", alpha=0.7)
    for ax, name in zip(axes, ["ReH", "ImH"]):
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xlabel(f"local {name}")
        ax.set_ylabel(f"global DNN {name}")
        ax.grid(alpha=0.25)
    axes[0].set_title("Real CFF: local vs global")
    axes[1].set_title("Imaginary CFF: local vs global")
    fig.tight_layout()
    images.append(("Local/global extraction consistency", figure_base64(fig)))

    # Pulls against the two surrogate means.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(predictions["pull_xsec"], bins=70)
    axes[1].hist(predictions["pull_diff"], bins=70)
    axes[0].set_xlabel("BKM - xsec surrogate [experimental sigma]")
    axes[1].set_xlabel("BKM - difference surrogate [experimental sigma]")
    for ax in axes:
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    images.append(("Observable-level residuals", figure_base64(fig)))

    # Representative simultaneous reconstructions: median and worst set per experiment.
    representative: List[int] = []
    for _, group in cff_sets.groupby("experiment"):
        order = group.sort_values("joint_chi2")
        representative.append(int(order.iloc[len(order) // 2]["set"]))
        representative.append(int(order.iloc[-1]["set"]))
    for set_id in representative:
        g = predictions[predictions["set"] == set_id].sort_values("phi")
        if g.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].errorbar(g["phi"], g["unp_beam_unp_target_xsec"], yerr=g["unp_beam_unp_target_xsec_err"], fmt="o", ms=3, capsize=2, label="data")
        axes[0].plot(g["phi"], g["model_xsec"], linewidth=2, label="xsec surrogate")
        axes[0].plot(g["phi"], g["bkm_xsec"], linestyle="--", linewidth=2, label="BKM from CFF DNN")
        axes[1].errorbar(g["phi"], g["xsec_diff"], yerr=g["xsec_diff_err"], fmt="o", ms=3, capsize=2, label="data")
        axes[1].plot(g["phi"], g["model_xsec_diff"], linewidth=2, label="difference surrogate")
        axes[1].plot(g["phi"], g["bkm_xsec_diff"], linestyle="--", linewidth=2, label="BKM from CFF DNN")
        axes[0].set_ylabel("cross section")
        axes[1].set_ylabel("helicity cross-section difference")
        for ax in axes:
            ax.set_xlabel(r"$\phi$ [rad]")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        center = cff_sets[cff_sets["set"] == set_id].iloc[0]
        fig.suptitle(
            f"set {set_id}, {center['experiment']} | Q2={center['q_squared']:.3g}, "
            f"xB={center['x_b']:.3g}, t={center['t']:.3g} | "
            f"ReH={center['ReH']:.3g}, ImH={center['ImH']:.3g}"
        )
        fig.tight_layout()
        images.append((f"Representative simultaneous fit: set {set_id}", figure_base64(fig)))

    ident_summary = ident.groupby("model")["jac_condition"].agg(["count", "median", "mean", "max"]).reset_index()
    table_ident = ident_summary.to_html(index=False, float_format=lambda x: f"{x:.4g}")
    table_exp = cff_sets.groupby("experiment").agg(
        sets=("set", "count"),
        mean_joint_chi2=("joint_chi2", "mean"),
        median_ReH=("ReH", "median"),
        median_ImH=("ImH", "median"),
    ).reset_index().to_html(index=False, float_format=lambda x: f"{x:.4g}")

    html = [
        "<html><head><meta charset='utf-8'><title>Joint DVCS CFF extraction</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2em auto;line-height:1.45} img{max-width:100%;height:auto} table{border-collapse:collapse} th,td{border:1px solid #bbb;padding:5px 8px} code{background:#eee;padding:2px 4px}</style>",
        "</head><body>",
        "<h1>Simultaneous DVCS cross-section / helicity-difference CFF extraction</h1>",
        f"<p><b>Central parameterization:</b> H dominance, with a DNN surface <code>(Q2,xB,t) → (ReH,ImH)</code>. The BKM layer contains the corrected Ktilde/K kinematics and the full quadratic H contribution.</p>",
        f"<p><b>Strict common domain:</b> {metrics['interpolation_selected_sets']} sets passed both surrogate closure and angular-coverage cuts; {metrics['extraction_selected_sets']} sets ({metrics['extraction_selected_rows']} rows) also passed the local H-dominance BKM closure and identifiability cuts.</p>",
        f"<p><b>Observable reconstruction:</b> pointwise chi2/N = {metrics['point_mean_chi2']:.4f}, xsec = {metrics['point_xsec_chi2']:.4f}, difference = {metrics['point_diff_chi2']:.4f}. These compare the BKM prediction with the central surrogate surfaces using the original experimental point errors.</p>",
        f"<p><b>Algorithmic stability:</b> median seed spread is {metrics['median_ReH_algorithmic_std']:.4g} for ReH and {metrics['median_ImH_algorithmic_std']:.4g} for ImH. This is optimizer/architecture-seed variability only, not the final experimental CFF uncertainty.</p>",
        "<p><b>Interpretation:</b> ReH and ImH below are effective H-dominance CFFs. Freeing E or Htilde produces strongly ill-conditioned local Jacobians, so those components are not separately identified by these two observables without additional priors or observables.</p>",
        "<p><b>Uncertainty status:</b> the next step is to repeat the entire extraction for matched cross-section/difference surrogate replicas. Curvature errors from local fits and multi-seed spread are diagnostics, not a replacement for that replica propagation.</p>",
        "<h2>Results by experiment</h2>", table_exp,
        "<h2>Identifiability diagnostic</h2>", table_ident,
    ]
    for title, image in images:
        html.extend([f"<h2>{title}</h2>", f"<img src='data:image/png;base64,{image}'>"])
    html.extend(["</body></html>"])
    (outdir / "index.html").write_text("\n".join(html), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="xsec_diff_surrogate/predictions_with_pulls.csv")
    parser.add_argument("--bkm-module", default="bkm10_observables_corrected.py")
    parser.add_argument("--outdir", default="joint_cff_extraction")
    parser.add_argument("--min-common-points", type=int, default=12)
    parser.add_argument("--max-surrogate-chi2", type=float, default=2.0)
    parser.add_argument("--max-phi-gap-deg", type=float, default=90.0)
    parser.add_argument("--max-local-bkm-chi2", type=float, default=2.0)
    parser.add_argument("--max-local-jac-condition", type=float, default=10.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--output-limit", type=float, default=6.0)
    parser.add_argument("--seeds", default="11,23,47")
    parser.add_argument("--pretrain-epochs", type=int, default=500)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--physics-stages", default="700:1e-3,500:3e-4,300:1e-4")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    missing = [column for column in required_columns() if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.reset_index(drop=True)
    bkm = load_module(Path(args.bkm_module))
    poly = build_bkm_polynomial(df, bkm)

    audit = build_interpolation_audit(df, args)
    local_rows = []
    for set_id in audit.loc[audit["interpolation_selected"], "set"]:
        row_indices = np.flatnonzero(df["set"].to_numpy() == int(set_id))
        result = local_h_fit(df, poly, row_indices)
        base = audit[audit["set"] == set_id].iloc[0].to_dict()
        local_rows.append({**base, **result})
    local = pd.DataFrame(local_rows)
    local["extraction_selected"] = (
        (local["local_bkm_chi2"] <= args.max_local_bkm_chi2)
        & (local["local_jac_condition"] <= args.max_local_jac_condition)
    )

    audit = audit.merge(
        local[["set", "local_bkm_chi2", "local_bkm_chi2_xsec", "local_bkm_chi2_diff", "local_jac_condition", "extraction_selected"]],
        on="set", how="left",
    )
    audit["extraction_selected"] = audit["extraction_selected"].fillna(False)
    audit["extraction_exclusion_reason"] = ""
    audit.loc[audit["interpolation_selected"] & (audit["local_bkm_chi2"] > args.max_local_bkm_chi2), "extraction_exclusion_reason"] = "H_dominance_BKM_closure"
    audit.loc[audit["interpolation_selected"] & (audit["local_jac_condition"] > args.max_local_jac_condition), "extraction_exclusion_reason"] += ";local_identifiability"

    local_train = local[local["extraction_selected"]].sort_values("set").reset_index(drop=True)
    selected_sets = local_train["set"].astype(int).tolist()
    row_indices = np.flatnonzero(df["set"].isin(selected_sets).to_numpy())
    train_df = df.iloc[row_indices].copy().reset_index(drop=True)

    x_all = cff_features(train_df["q_squared"], train_df["x_b"], train_df["t"])
    feature_mean = x_all.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = x_all.std(axis=0, keepdims=True).astype(np.float32)
    feature_std[feature_std == 0.0] = 1.0
    y_local = local_train[["ReH_local", "ImH_local"]].to_numpy(np.float32)
    output_mean = y_local.mean(axis=0, keepdims=True).astype(np.float32)
    output_std = y_local.std(axis=0, keepdims=True).astype(np.float32)
    output_std[output_std == 0.0] = 1.0

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    models: List[CFFNet] = []
    histories: List[Dict[str, float]] = []
    seed_metrics: List[Dict[str, float]] = []
    for seed in seeds:
        model, history, met = train_seed(
            seed, args, train_df, local_train, row_indices, poly,
            feature_mean, feature_std, output_mean, output_std,
        )
        models.append(model)
        histories.extend(history)
        seed_metrics.append(met)
        print(json.dumps(met), flush=True)

    best_index = int(np.argmin([item["balanced_loss"] for item in seed_metrics]))
    central_model = models[best_index]
    central_seed = seeds[best_index]

    # Central row-level predictions.
    cff_rows = predict_cff(central_model, train_df["q_squared"], train_df["x_b"], train_df["t"])
    ix = H_INDICES
    c0x = poly["c0_xsec"][row_indices]
    lx = poly["L_xsec"][row_indices][:, ix]
    qx = poly["Q_xsec"][row_indices][:, ix][:, :, ix]
    c0d = poly["c0_diff"][row_indices]
    ld = poly["L_diff"][row_indices][:, ix]
    qd = poly["Q_diff"][row_indices][:, ix][:, :, ix]
    pred_x = c0x + np.sum(lx * cff_rows, axis=1) + np.einsum("ni,nij,nj->n", cff_rows, qx, cff_rows)
    pred_d = c0d + np.sum(ld * cff_rows, axis=1) + np.einsum("ni,nij,nj->n", cff_rows, qd, cff_rows)
    predictions = train_df.copy()
    predictions["ReH"] = cff_rows[:, 0]
    predictions["ImH"] = cff_rows[:, 1]
    predictions["bkm_xsec"] = pred_x
    predictions["bkm_xsec_diff"] = pred_d
    predictions["pull_xsec"] = (pred_x - predictions["model_xsec"]) / predictions["unp_beam_unp_target_xsec_err"]
    predictions["pull_diff"] = (pred_d - predictions["model_xsec_diff"]) / predictions["xsec_diff_err"]

    # Evaluate every seed at common set-center kinematics.
    centers = local_train[["set", "experiment", "k", "q_squared", "x_b", "t"]].copy()
    ensemble = np.stack([
        predict_cff(model, centers["q_squared"], centers["x_b"], centers["t"])
        for model in models
    ])
    central_centers = ensemble[best_index]
    centers["ReH"] = central_centers[:, 0]
    centers["ImH"] = central_centers[:, 1]
    centers["ReH_algorithmic_std"] = np.std(ensemble[:, :, 0], axis=0, ddof=1) if len(models) > 1 else 0.0
    centers["ImH_algorithmic_std"] = np.std(ensemble[:, :, 1], axis=0, ddof=1) if len(models) > 1 else 0.0
    set_fit = predictions.groupby("set").apply(
        lambda g: pd.Series({
            "xsec_chi2": float(np.mean(np.square(g["pull_xsec"]))),
            "diff_chi2": float(np.mean(np.square(g["pull_diff"]))),
            "joint_chi2": float(0.5 * (np.mean(np.square(g["pull_xsec"])) + np.mean(np.square(g["pull_diff"])))),
        }), include_groups=False,
    ).reset_index()
    cff_sets = centers.merge(set_fit, on="set", how="left")

    ident = identifiability_rows(df, poly, local_train)
    ident_summary = ident.groupby("model")["jac_condition"].agg(["count", "median", "mean", "max"]).reset_index()

    selected_metrics = seed_metrics[best_index]
    metrics = {
        "model_type": "simultaneous_BKM10_H_dominance_CFF_DNN",
        "central_seed": central_seed,
        "seeds": seeds,
        "input_rows": int(len(df)),
        "input_sets": int(df["set"].nunique()),
        "interpolation_selected_sets": int(audit["interpolation_selected"].sum()),
        "extraction_selected_sets": int(len(selected_sets)),
        "extraction_selected_rows": int(len(predictions)),
        "domain_definition": {
            "min_common_points": args.min_common_points,
            "max_surrogate_chi2_each_observable": args.max_surrogate_chi2,
            "max_circular_phi_gap_deg": args.max_phi_gap_deg,
            "requires_both_phi_signs": True,
            "max_local_H_dominance_BKM_chi2": args.max_local_bkm_chi2,
            "max_local_H2_jacobian_condition": args.max_local_jac_condition,
        },
        "parameterization": "ReH and ImH free; ReE, ImE, ReHt, ImHt, ReEt, ImEt fixed to zero",
        "bkm_kinematics": "Ktilde/K definition from BHDVCS_tf_modified.SetKinematics",
        "point_mean_chi2": float(0.5 * (np.mean(np.square(predictions["pull_xsec"])) + np.mean(np.square(predictions["pull_diff"])))),
        "point_xsec_chi2": float(np.mean(np.square(predictions["pull_xsec"]))),
        "point_diff_chi2": float(np.mean(np.square(predictions["pull_diff"]))),
        "mean_set_joint_chi2": float(cff_sets["joint_chi2"].mean()),
        "max_set_joint_chi2": float(cff_sets["joint_chi2"].max()),
        "median_ReH_algorithmic_std": float(cff_sets["ReH_algorithmic_std"].median()),
        "mean_ReH_algorithmic_std": float(cff_sets["ReH_algorithmic_std"].mean()),
        "max_ReH_algorithmic_std": float(cff_sets["ReH_algorithmic_std"].max()),
        "median_ImH_algorithmic_std": float(cff_sets["ImH_algorithmic_std"].median()),
        "mean_ImH_algorithmic_std": float(cff_sets["ImH_algorithmic_std"].mean()),
        "max_ImH_algorithmic_std": float(cff_sets["ImH_algorithmic_std"].max()),
        "BKM_polynomial_max_abs_error_xsec": float(poly["max_polynomial_error_xsec"]),
        "BKM_polynomial_max_abs_error_diff": float(poly["max_polynomial_error_diff"]),
        "seed_metrics": seed_metrics,
        "identifiability_condition_medians": dict(zip(ident_summary["model"], ident_summary["median"])),
        "uncertainty_status": "algorithmic seed spread only; matched surrogate replicas are still required for experimental CFF errors",
    }

    # Add useful metadata to the selected checkpoint.
    torch.save({
        "model_type": metrics["model_type"],
        "model_state_dict": central_model.state_dict(),
        "feature_definition": ["log(Q2)", "logit(xB)", "log(-t)"],
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "output_mean": output_mean,
        "output_std": output_std,
        "hidden": args.hidden,
        "depth": args.depth,
        "output_limit": args.output_limit,
        "cff_names": H_NAMES,
        "central_seed": central_seed,
        "selected_sets": selected_sets,
        "parameterization": metrics["parameterization"],
        "domain_definition": metrics["domain_definition"],
    }, outdir / "cff_surrogate.pt")

    predictions.to_csv(outdir / "cff_predictions.csv", index=False)
    cff_sets.to_csv(outdir / "cff_sets.csv", index=False)
    audit.to_csv(outdir / "common_domain_audit.csv", index=False)
    local.to_csv(outdir / "local_h2_extraction.csv", index=False)
    ident.to_csv(outdir / "identifiability.csv", index=False)
    ident_summary.to_csv(outdir / "identifiability_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(outdir / "seed_metrics.csv", index=False)
    with open(outdir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    make_report(outdir, args, audit, local_train, ident, predictions, cff_sets, metrics)
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)
    print(f"Wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
