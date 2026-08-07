#!/usr/bin/env python3
"""Optimized hard-odd DNN surrogate for the DVCS beam-helicity cross-section difference.

This script is the companion to ``dvcs_xsec_direct_dnn_optimized.py``.  Run the
cross-section script first on the same common cleaned CSV, then run this script.
The frozen cross-section surrogate supplies the denominator/scale needed to
represent the cross-section difference through a bounded beam-spin asymmetry:

    u(phi) = [1 - cos(phi - phi_center)] / 2

    g       = DNN(k, Q2, xB, t, u)

    A_LU    = tanh[ sin(phi - phi_center) * g ]

    Delta-sigma_LU = sigma_UU_surrogate * A_LU.

Why this form?
--------------
* ``u`` is even in phi and ``sin(phi)`` is odd, so A_LU and Delta-sigma are
  exactly antisymmetric under phi -> -phi.
* tanh keeps the inferred asymmetry in the physical interval (-1,1), while
  still allowing A_LU/sin(phi) to be larger than one when the data require it.
* Multiplication by the already-trained positive cross-section surrogate gives
  a dimensionful cross-section difference and makes the two observable models
  directly compatible for the later BKM/CFF fit.
* The network itself contains no hand-selected Fourier or Bernstein expansion.
  Smoothness is controlled with mild arc-length and curvature penalties on the
  dimensionless A_LU(phi) curve.

Expected directory layout
-------------------------
Place this script beside ``dvcs_xsec_direct_dnn_optimized.py`` and run, e.g.::

    python dvcs_xsec_direct_dnn_optimized.py \
      --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
      --outdir xsec_surrogate

    python dvcs_xsec_diff_direct_dnn_optimized.py \
      --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
      --xsec-checkpoint xsec_surrogate/xsec_surrogate.pt \
      --outdir xsec_diff_surrogate

Outputs are intentionally simple: one checkpoint, one prediction CSV, metrics,
training history, and one self-contained HTML report.  No per-set PNGs or zip
archive are produced.
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_xsec_module(path: Path):
    """Load the companion cross-section script without relying on PYTHONPATH."""
    if not path.exists():
        raise FileNotFoundError(f"Cross-section model script not found: {path}")
    spec = importlib.util.spec_from_file_location("dvcs_xsec_companion", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def u_from_phi(phi: np.ndarray, phi_center: float = 0.0) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float32).reshape(-1, 1)
    return ((1.0 - np.cos(phi - float(phi_center))) / 2.0).astype(np.float32)


def sin_from_phi(phi: np.ndarray, phi_center: float = 0.0) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float32).reshape(-1, 1)
    return np.sin(phi - float(phi_center)).astype(np.float32)


def make_features(base_raw: np.ndarray, phi: np.ndarray, phi_center: float = 0.0) -> np.ndarray:
    base_raw = np.asarray(base_raw, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32).reshape(-1, 1)
    if base_raw.ndim == 1:
        base_raw = np.repeat(base_raw[None, :], len(phi), axis=0)
    if len(base_raw) != len(phi):
        raise ValueError(f"base_raw/phi length mismatch: {len(base_raw)} vs {len(phi)}")
    return np.concatenate([base_raw, u_from_phi(phi, phi_center)], axis=1).astype(np.float32)


class OddXsecDiffDNN(nn.Module):
    """DNN for the even amplitude inside an exactly odd BSA representation."""

    def __init__(
        self,
        base_cols: List[str],
        hidden: int,
        depth: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        output_limit: float = 12.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.base_cols = list(base_cols)
        self.feature_cols = list(base_cols) + ["u"]
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.output_limit = float(output_limit)
        self.activation = str(activation)

        self.register_buffer("x_mean", torch.tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(feature_std, dtype=torch.float32))

        if activation.lower() == "tanh":
            act_factory = nn.Tanh
        elif activation.lower() == "gelu":
            act_factory = nn.GELU
        else:
            act_factory = nn.SiLU

        layers: List[nn.Module] = []
        in_dim = len(self.feature_cols)
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(act_factory())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def amplitude(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.net((features - self.x_mean) / self.x_std)
        if self.output_limit > 0:
            raw = self.output_limit * torch.tanh(raw / self.output_limit)
        return raw

    def bsa(self, features: torch.Tensor, sin_phi: torch.Tensor) -> torch.Tensor:
        # The argument of tanh is odd because amplitude(features) depends on phi
        # only through even u(phi), while sin_phi is odd.
        return torch.tanh(sin_phi * self.amplitude(features))

    def xsec_diff(
        self,
        features: torch.Tensor,
        sin_phi: torch.Tensor,
        xsec: torch.Tensor,
    ) -> torch.Tensor:
        return xsec * self.bsa(features, sin_phi)

    def predict_numpy(
        self,
        xsec_model: nn.Module,
        base_raw: np.ndarray,
        phi: np.ndarray,
        phi_center: float = 0.0,
        batch: int = 16384,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return model_xsec, model_bsa, and model_xsec_diff."""
        self.eval()
        xsec_model.eval()
        features = make_features(base_raw, phi, phi_center)
        sin_phi = sin_from_phi(phi, phi_center)
        xs_out: List[np.ndarray] = []
        bsa_out: List[np.ndarray] = []
        diff_out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), batch):
                ft = torch.tensor(features[start:start + batch], dtype=torch.float32)
                st = torch.tensor(sin_phi[start:start + batch], dtype=torch.float32)
                xs = xsec_model.sigma(ft)
                bsa = self.bsa(ft, st)
                diff = xs * bsa
                xs_out.append(xs.cpu().numpy())
                bsa_out.append(bsa.cpu().numpy())
                diff_out.append(diff.cpu().numpy())
        return (
            np.concatenate(xs_out).ravel(),
            np.concatenate(bsa_out).ravel(),
            np.concatenate(diff_out).ravel(),
        )


def group_mean(values: torch.Tensor, index: torch.Tensor, n_groups: int) -> torch.Tensor:
    sums = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    sums.index_add_(0, index, values.reshape(-1))
    counts = torch.bincount(index, minlength=n_groups).to(values.dtype)
    return sums / counts.clamp_min(1.0)


def balanced_loss(
    point_losses: torch.Tensor,
    set_index: torch.Tensor,
    n_sets: int,
    set_experiment_index: torch.Tensor,
    n_experiments: int,
    balance: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    point_mean = point_losses.mean()
    set_losses = group_mean(point_losses, set_index, n_sets)
    set_mean = set_losses.mean()
    experiment_mean = group_mean(set_losses, set_experiment_index, n_experiments).mean()
    if balance == "point":
        chosen = point_mean
    elif balance == "set":
        chosen = set_mean
    else:
        chosen = experiment_mean
    return chosen, set_losses, point_mean


def metrics_np(y: np.ndarray, err: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    pull = (pred.ravel() - y.ravel()) / err.ravel()
    abs_pull = np.abs(pull)
    return {
        "n_points": int(len(pull)),
        "chi2_per_point": float(np.mean(pull**2)),
        "pull_rms": float(np.sqrt(np.mean(pull**2))),
        "pull_mean": float(np.mean(pull)),
        "pull_std": float(np.std(pull)),
        "median_abs_pull": float(np.median(abs_pull)),
        "frac_abs_pull_lt_1": float(np.mean(abs_pull < 1.0)),
        "frac_abs_pull_lt_2": float(np.mean(abs_pull < 2.0)),
        "frac_abs_pull_lt_3": float(np.mean(abs_pull < 3.0)),
        "max_abs_pull": float(np.max(abs_pull)),
    }


def instantiate_xsec_model(module, checkpoint: Dict) -> nn.Module:
    required = [
        "base_cols", "hidden", "depth", "feature_mean", "feature_std",
        "log_y_mean", "log_y_std", "model_state_dict",
    ]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Cross-section checkpoint is missing: {missing}")
    model = module.EvenDirectDNN(
        base_cols=list(checkpoint["base_cols"]),
        hidden=int(checkpoint["hidden"]),
        depth=int(checkpoint["depth"]),
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        log_y_mean=float(checkpoint["log_y_mean"]),
        log_y_std=float(checkpoint["log_y_std"]),
        output_limit=float(checkpoint.get("output_limit", 10.0)),
        activation=str(checkpoint.get("activation", "silu")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def instantiate_diff_model(checkpoint: Dict) -> OddXsecDiffDNN:
    """Reconstruct a trained xsec-difference model from its checkpoint."""
    required = [
        "base_cols", "hidden", "depth", "feature_mean", "feature_std",
        "model_state_dict",
    ]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Xsec-difference checkpoint is missing: {missing}")
    model = OddXsecDiffDNN(
        base_cols=list(checkpoint["base_cols"]),
        hidden=int(checkpoint["hidden"]),
        depth=int(checkpoint["depth"]),
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        output_limit=float(checkpoint.get("output_limit", 12.0)),
        activation=str(checkpoint.get("activation", "silu")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_data(args: argparse.Namespace, xsec_checkpoint: Dict) -> Tuple:
    frame = pd.read_csv(args.csv)
    base_cols = list(xsec_checkpoint["base_cols"])
    required = base_cols + [
        "phi", "set", args.target_col, args.error_col,
        args.xsec_target_col, args.xsec_error_col,
    ]
    if args.balance == "experiment":
        required.append(args.experiment_col)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    use = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    use = use[(use[args.error_col] > 0) & (use[args.xsec_target_col] > 0)].copy()

    duplicate_subset = base_cols + ["phi", args.target_col, args.error_col, "set"]
    duplicate_count = int(len(use) - len(use.drop_duplicates(subset=duplicate_subset)))
    if not args.keep_exact_duplicates:
        use = use.drop_duplicates(subset=duplicate_subset).copy()

    base_all = use[base_cols].to_numpy(np.float32)
    phi_all = use[["phi"]].to_numpy(np.float32)
    features_all = make_features(base_all, phi_all, args.phi_center)
    sin_all = sin_from_phi(phi_all, args.phi_center)
    y_all = use[[args.target_col]].to_numpy(np.float32)
    err_all = use[[args.error_col]].to_numpy(np.float32)
    base_unique = np.unique(base_all, axis=0).astype(np.float32)

    # Reuse the cross-section feature scaling.  This makes the two companion
    # networks use exactly the same numerical coordinate system.
    feature_mean = np.asarray(xsec_checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(xsec_checkpoint["feature_std"], dtype=np.float32)

    set_codes, set_values = pd.factorize(use["set"], sort=True)
    if args.balance == "experiment":
        set_experiments = (
            use[["set", args.experiment_col]]
            .drop_duplicates("set")
            .set_index("set")
            .loc[set_values, args.experiment_col]
        )
        experiment_codes, experiment_values = pd.factorize(set_experiments, sort=True)
    else:
        experiment_codes = np.zeros(len(set_values), dtype=np.int64)
        experiment_values = np.asarray(["all"])

    return (
        use, base_cols, base_all, phi_all, features_all, sin_all, y_all, err_all,
        base_unique, feature_mean, feature_std, duplicate_count,
        set_codes.astype(np.int64), set_values,
        experiment_codes.astype(np.int64), experiment_values,
    )


def make_regularization_grid(
    base_unique: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, float]:
    phi_grid = np.linspace(
        args.reg_phi_min, args.reg_phi_max, args.reg_grid_points, dtype=np.float32
    ).reshape(-1, 1)
    reg_base = np.repeat(base_unique, args.reg_grid_points, axis=0)
    reg_phi = np.tile(phi_grid, (len(base_unique), 1))
    reg_features = torch.tensor(
        make_features(reg_base, reg_phi, args.phi_center), dtype=torch.float32
    )
    reg_sin = torch.tensor(sin_from_phi(reg_phi, args.phi_center), dtype=torch.float32)
    dphi = float((args.reg_phi_max - args.reg_phi_min) / (args.reg_grid_points - 1))
    return reg_features, reg_sin, len(base_unique), args.reg_grid_points, dphi


def smoothness_losses(
    model: OddXsecDiffDNN,
    reg_features: torch.Tensor,
    reg_sin: torch.Tensor,
    n_base: int,
    n_grid: int,
    dphi: float,
    worst_fraction: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return average and worst-subset BSA smoothness diagnostics.

    An average penalty alone can hide a small number of very wiggly phase-space
    curves among more than one thousand well-behaved curves.  The optional
    worst-subset terms are the mean arc-length/curvature of the worst fraction
    of base kinematic points.  They let the optimizer suppress local failures
    without replacing the DNN by a hand-chosen harmonic basis.
    """
    amplitude = model.amplitude(reg_features).reshape(n_base, n_grid)
    bsa = torch.tanh(reg_sin.reshape(n_base, n_grid) * amplitude)
    d1 = (bsa[:, 1:] - bsa[:, :-1]) / dphi
    length_per_base = torch.mean(torch.sqrt(1.0 + d1**2) - 1.0, dim=1)
    d2 = (bsa[:, 2:] - 2.0 * bsa[:, 1:-1] + bsa[:, :-2]) / (dphi**2)
    curvature_per_base = torch.mean(d2**2, dim=1)
    length_loss = length_per_base.mean()
    curvature_loss = curvature_per_base.mean()
    amplitude_l2 = torch.mean(amplitude**2)

    if worst_fraction > 0.0:
        n_top = max(1, int(math.ceil(float(worst_fraction) * n_base)))
        worst_length = torch.topk(length_per_base, n_top).values.mean()
        worst_curvature = torch.topk(curvature_per_base, n_top).values.mean()
    else:
        worst_length = torch.zeros((), dtype=length_loss.dtype, device=length_loss.device)
        worst_curvature = torch.zeros((), dtype=curvature_loss.dtype, device=curvature_loss.device)
    return length_loss, curvature_loss, amplitude_l2, worst_length, worst_curvature


def evaluate_model(
    model: OddXsecDiffDNN,
    xsec_model: nn.Module,
    features_t: torch.Tensor,
    sin_t: torch.Tensor,
    y: np.ndarray,
    err: np.ndarray,
    use: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], pd.Series]:
    with torch.no_grad():
        xsec = xsec_model.sigma(features_t)
        bsa = model.bsa(features_t, sin_t)
        pred = xsec * bsa
    xsec_np = xsec.cpu().numpy().ravel()
    bsa_np = bsa.cpu().numpy().ravel()
    pred_np = pred.cpu().numpy().ravel()
    metrics = metrics_np(y, err, pred_np)
    pull = (pred_np - y.ravel()) / err.ravel()
    temp = use[["set"]].copy()
    temp["pull2"] = pull**2
    per_set = temp.groupby("set")["pull2"].mean()
    metrics["mean_set_chi2"] = float(per_set.mean())
    metrics["max_set_chi2"] = float(per_set.max())
    return xsec_np, bsa_np, pred_np, metrics, per_set


def train(args: argparse.Namespace):
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_path = Path(args.xsec_model_script)
    xsec_module = load_xsec_module(script_path)
    checkpoint = torch.load(args.xsec_checkpoint, map_location="cpu", weights_only=False)
    xsec_model = instantiate_xsec_model(xsec_module, checkpoint)

    (
        use, base_cols, base_all, phi_all, features_all, sin_all, y_all, err_all,
        base_unique, feature_mean, feature_std, duplicate_count,
        set_codes, set_values, experiment_codes, experiment_values,
    ) = load_data(args, checkpoint)

    if base_cols != list(checkpoint["base_cols"]):
        raise ValueError("Clean data base columns do not match the xsec checkpoint")
    checkpoint_phi_center = float(checkpoint.get("phi_center", 0.0))
    if abs(checkpoint_phi_center - args.phi_center) > 1e-12:
        raise ValueError(
            f"phi_center mismatch: xsec checkpoint={checkpoint_phi_center}, diff={args.phi_center}"
        )

    model = OddXsecDiffDNN(
        base_cols=base_cols,
        hidden=args.hidden,
        depth=args.depth,
        feature_mean=feature_mean,
        feature_std=feature_std,
        output_limit=args.output_limit,
        activation=args.activation,
    )
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        init_model = instantiate_diff_model(init_checkpoint)
        if list(init_model.base_cols) != base_cols:
            raise ValueError("Initial xsec-difference checkpoint has incompatible base columns")
        if init_model.hidden != args.hidden or init_model.depth != args.depth:
            raise ValueError(
                "Initial xsec-difference checkpoint architecture does not match "
                "--hidden/--depth"
            )
        model.load_state_dict(init_model.state_dict())

    features_t = torch.tensor(features_all, dtype=torch.float32)
    sin_t = torch.tensor(sin_all, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.float32)
    err_t = torch.tensor(err_all, dtype=torch.float32)
    with torch.no_grad():
        xsec_t = xsec_model.sigma(features_t)

    set_index_t = torch.tensor(set_codes, dtype=torch.long)
    set_experiment_t = torch.tensor(experiment_codes, dtype=torch.long)
    n_sets = len(set_values)
    n_experiments = len(experiment_values)

    reg_features_t, reg_sin_t, n_base, n_grid, dphi = make_regularization_grid(
        base_unique, args
    )

    history: List[Dict[str, float]] = []
    start_time = time.time()
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")

    # Stage 1: obtain a complete uncertainty-weighted fit before smoothing.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    for epoch in range(1, args.epochs + 1):
        frac = epoch / max(1, args.epochs)
        if frac > args.lr_drop2_frac:
            lr = args.lr * args.lr_final_factor
        elif frac > args.lr_drop1_frac:
            lr = args.lr * args.lr_mid_factor
        else:
            lr = args.lr
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        bsa = model.bsa(features_t, sin_t)
        pred = xsec_t * bsa
        point_losses = ((pred - y_t) / err_t) ** 2
        data_loss, set_losses, point_mean = balanced_loss(
            point_losses, set_index_t, n_sets, set_experiment_t,
            n_experiments, args.balance,
        )
        if args.worst_set_weight > 0:
            n_top = max(1, int(math.ceil(args.worst_set_fraction * n_sets)))
            worst_loss = torch.topk(set_losses, n_top).values.mean()
        else:
            worst_loss = torch.tensor(0.0)

        # With the default zero stage-one smoothness weights, do not evaluate
        # the dense regularization grid.  Keep only a tiny amplitude L2 term at
        # the measured points; the expensive BSA length/curvature pass is used
        # during the low-learning-rate finishing stage.
        if args.stage1_length_lambda != 0.0 or args.stage1_curvature_lambda != 0.0:
            length_loss, curvature_loss, amplitude_l2, worst_length_loss, worst_curvature_loss = smoothness_losses(
                model, reg_features_t, reg_sin_t, n_base, n_grid, dphi,
                args.worst_curve_fraction,
            )
            reg_scale = max(0.0, min(1.0, (frac - args.reg_start_frac) / max(1e-12, args.reg_full_frac - args.reg_start_frac)))
        else:
            length_loss = torch.tensor(0.0)
            curvature_loss = torch.tensor(0.0)
            worst_length_loss = torch.tensor(0.0)
            worst_curvature_loss = torch.tensor(0.0)
            amplitude_l2 = torch.mean(model.amplitude(features_t) ** 2)
            reg_scale = 1.0
        loss = (
            data_loss
            + args.worst_set_weight * worst_loss
            + reg_scale * (
                args.stage1_length_lambda * (length_loss + args.worst_curve_weight * worst_length_loss)
                + args.stage1_curvature_lambda * (curvature_loss + args.worst_curve_weight * worst_curvature_loss)
                + args.amplitude_l2 * amplitude_l2
            )
        )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            _, _, _, eval_metrics, _ = evaluate_model(
                model, xsec_model, features_t, sin_t, y_all, err_all, use
            )
            score = eval_metrics["chi2_per_point"] + 0.20 * eval_metrics["mean_set_chi2"]
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            record = {
                "stage": 1,
                "epoch": int(epoch),
                "chi2_per_point": eval_metrics["chi2_per_point"],
                "mean_set_chi2": eval_metrics["mean_set_chi2"],
                "max_set_chi2": eval_metrics["max_set_chi2"],
                "data_loss": float(data_loss.detach()),
                "point_mean_loss": float(point_mean.detach()),
                "length_loss": float(length_loss.detach()),
                "curvature_loss": float(curvature_loss.detach()),
                "worst_length_loss": float(worst_length_loss.detach()),
                "worst_curvature_loss": float(worst_curvature_loss.detach()),
                "amplitude_l2": float(amplitude_l2.detach()),
                "lr": float(lr),
                "elapsed_sec": float(time.time() - start_time),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    model.load_state_dict(best_state)

    # Stage 2: low-learning-rate smooth finish.  A pointwise component preserves
    # conventional global chi2 while experiment balancing protects sparse data.
    if args.finetune_epochs > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=0.0)
        for epoch in range(1, args.finetune_epochs + 1):
            frac = epoch / max(1, args.finetune_epochs)
            if frac < 0.60:
                lr = args.finetune_lr
            elif frac < 0.85:
                lr = args.finetune_lr * 0.25
            else:
                lr = args.finetune_lr * 0.08
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            bsa = model.bsa(features_t, sin_t)
            pred = xsec_t * bsa
            point_losses = ((pred - y_t) / err_t) ** 2
            balanced, _, point_mean = balanced_loss(
                point_losses, set_index_t, n_sets, set_experiment_t,
                n_experiments, args.balance,
            )
            data_loss = (
                args.finetune_point_fraction * point_mean
                + (1.0 - args.finetune_point_fraction) * balanced
            )
            length_loss, curvature_loss, amplitude_l2, worst_length_loss, worst_curvature_loss = smoothness_losses(
                model, reg_features_t, reg_sin_t, n_base, n_grid, dphi,
                args.worst_curve_fraction,
            )
            loss = (
                data_loss
                + args.length_lambda * (length_loss + args.worst_curve_weight * worst_length_loss)
                + args.curvature_lambda * (curvature_loss + args.worst_curve_weight * worst_curvature_loss)
                + args.amplitude_l2 * amplitude_l2
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if epoch == 1 or epoch % args.finetune_print_every == 0 or epoch == args.finetune_epochs:
                _, _, _, eval_metrics, _ = evaluate_model(
                    model, xsec_model, features_t, sin_t, y_all, err_all, use
                )
                score = eval_metrics["chi2_per_point"] + 0.20 * eval_metrics["mean_set_chi2"]
                if score < best_score:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                record = {
                    "stage": 2,
                    "epoch": int(epoch),
                    "chi2_per_point": eval_metrics["chi2_per_point"],
                    "mean_set_chi2": eval_metrics["mean_set_chi2"],
                    "max_set_chi2": eval_metrics["max_set_chi2"],
                    "data_loss": float(data_loss.detach()),
                    "point_mean_loss": float(point_mean.detach()),
                    "length_loss": float(length_loss.detach()),
                    "curvature_loss": float(curvature_loss.detach()),
                    "worst_length_loss": float(worst_length_loss.detach()),
                    "worst_curvature_loss": float(worst_curvature_loss.detach()),
                    "amplitude_l2": float(amplitude_l2.detach()),
                    "lr": float(lr),
                    "elapsed_sec": float(time.time() - start_time),
                }
                history.append(record)
                print(json.dumps(record), flush=True)

    if not args.keep_final_state:
        model.load_state_dict(best_state)
    model_xsec, model_bsa, model_diff, metrics, per_set = evaluate_model(
        model, xsec_model, features_t, sin_t, y_all, err_all, use
    )
    pull = (model_diff - y_all.ravel()) / err_all.ravel()

    # Numerical symmetry test on the actual base points at a generic angle.
    test_phi = np.full(len(base_unique), 0.73, dtype=np.float32)
    _, bsa_plus, diff_plus = model.predict_numpy(xsec_model, base_unique, test_phi, args.phi_center)
    _, bsa_minus, diff_minus = model.predict_numpy(xsec_model, base_unique, -test_phi, args.phi_center)
    metrics.update(
        {
            "model_type": "direct_odd_xsec_diff_dnn_optimized",
            "n_sets": int(use["set"].nunique()),
            "n_unique_kinematics": int(len(base_unique)),
            "base_cols": base_cols,
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "activation": args.activation,
            "balance": args.balance,
            "length_lambda": float(args.length_lambda),
            "curvature_lambda": float(args.curvature_lambda),
            "amplitude_l2": float(args.amplitude_l2),
            "worst_curve_fraction": float(args.worst_curve_fraction),
            "worst_curve_weight": float(args.worst_curve_weight),
            "phi_center": float(args.phi_center),
            "output_limit": float(args.output_limit),
            "max_bsa_antisymmetry_abs_sum": float(np.max(np.abs(bsa_plus + bsa_minus))),
            "max_xsec_diff_antisymmetry_abs_sum": float(np.max(np.abs(diff_plus + diff_minus))),
            "max_abs_model_bsa_at_data": float(np.max(np.abs(model_bsa))),
            "exact_duplicate_rows_removed": int(0 if args.keep_exact_duplicates else duplicate_count),
            "xsec_checkpoint": str(Path(args.xsec_checkpoint).resolve()),
            "feature_definition": "base kinematics plus u=(1-cos(phi-phi_center))/2; no Fourier/Bernstein basis",
            "observable_definition": "A_LU=tanh(sin(phi-phi_center)*DNN); xsec_diff=xsec_surrogate*A_LU",
            "training_definition": "uncertainty-weighted balanced chi2 followed by low-LR BSA length/curvature finish",
        }
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = use.copy()
    output["model_xsec"] = model_xsec
    output["model_bsa"] = model_bsa
    output["model_xsec_diff"] = model_diff
    output["pull"] = pull
    output.to_csv(outdir / "predictions_with_pulls.csv", index=False)
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    torch.save(
        {
            "model_type": "direct_odd_xsec_diff_dnn_optimized",
            "model_state_dict": model.state_dict(),
            "base_cols": base_cols,
            "target_col": args.target_col,
            "error_col": args.error_col,
            "xsec_target_col": args.xsec_target_col,
            "xsec_error_col": args.xsec_error_col,
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": args.activation,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "phi_center": args.phi_center,
            "output_limit": args.output_limit,
            "length_lambda": args.length_lambda,
            "curvature_lambda": args.curvature_lambda,
            "amplitude_l2": args.amplitude_l2,
            "worst_curve_fraction": args.worst_curve_fraction,
            "worst_curve_weight": args.worst_curve_weight,
            "balance": args.balance,
            "xsec_checkpoint": str(Path(args.xsec_checkpoint).resolve()),
        },
        outdir / "xsec_diff_surrogate.pt",
    )
    return model, xsec_model, output, metrics, per_set


def fig_to_html_img(fig: plt.Figure, dpi: int) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return '<img src="data:image/png;base64,' + encoded + '" style="max-width:950px;width:100%;">'


def make_html_report(
    model: OddXsecDiffDNN,
    xsec_model: nn.Module,
    out: pd.DataFrame,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    outdir = Path(args.outdir)
    style = """
    body{font-family:Arial,sans-serif;margin:24px;max-width:1050px}
    table{border-collapse:collapse;margin:12px 0}
    td,th{border:1px solid #bbb;padding:4px 8px;text-align:right}
    th:first-child,td:first-child{text-align:left}
    .plot{border-top:1px solid #bbb;margin-top:26px;padding-top:18px}
    code{background:#f2f2f2;padding:2px 4px}
    """
    parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>DVCS xsec-difference surrogate</title>",
        f"<style>{style}</style></head><body>",
        "<h1>DVCS beam-helicity cross-section-difference surrogate</h1>",
        "<p>The frozen cross-section surrogate supplies <code>model_xsec</code>. "
        "The companion DNN predicts an even amplitude and constructs "
        "<code>model_bsa=tanh(sin(phi)*amplitude)</code>, followed by "
        "<code>model_xsec_diff=model_xsec*model_bsa</code>.  This enforces exact "
        "phi antisymmetry without a Fourier or Bernstein expansion.</p>",
    ]
    keys = [
        "chi2_per_point", "mean_set_chi2", "max_set_chi2", "pull_rms",
        "median_abs_pull", "frac_abs_pull_lt_1", "frac_abs_pull_lt_2",
        "frac_abs_pull_lt_3", "max_abs_pull", "n_points", "n_sets",
        "max_bsa_antisymmetry_abs_sum", "max_xsec_diff_antisymmetry_abs_sum",
    ]
    parts.append("<h2>Global metrics</h2><table>")
    for key in keys:
        value = metrics.get(key, "")
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        parts.append(f"<tr><th>{key}</th><td>{text}</td></tr>")
    parts.append("</table>")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(out["pull"].to_numpy(), bins=80)
    ax.set_xlabel("pull = (model_xsec_diff - data) / total error")
    ax.set_ylabel("count")
    ax.set_title("Global xsec-difference pull distribution")
    fig.tight_layout()
    parts.append("<h2>Global diagnostics</h2>" + fig_to_html_img(fig, args.html_dpi))

    y = out[args.target_col].to_numpy()
    pred = out["model_xsec_diff"].to_numpy()
    err = out[args.error_col].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.errorbar(y, pred, xerr=err, fmt=".", markersize=2, alpha=0.35)
    lo = min(float(y.min()), float(pred.min()))
    hi = max(float(y.max()), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("data xsec_diff")
    ax.set_ylabel("model_xsec_diff")
    ax.set_title(f"Predicted vs data, chi2/N={metrics['chi2_per_point']:.3f}")
    fig.tight_layout()
    parts.append(fig_to_html_img(fig, args.html_dpi))

    set_rows = []
    for set_id, group in out.groupby("set", sort=True):
        pulls = group["pull"].to_numpy()
        set_rows.append(
            {
                "set": int(set_id), "n": int(len(group)),
                "chi2_per_point": float(np.mean(pulls**2)),
                # Some published sets use bin-averaged kinematics that vary
                # slightly with phi.  Report the median and draw the dense line
                # at those representative kinematics; model markers at the
                # measured bins still use each row's actual kinematics.
                "k": float(group["k"].median()),
                "q_squared": float(group["q_squared"].median()),
                "x_b": float(group["x_b"].median()),
                "t": float(group["t"].median()),
            }
        )
    set_table = pd.DataFrame(set_rows).sort_values("chi2_per_point", ascending=False)
    set_table.to_csv(outdir / "set_metrics.csv", index=False)

    parts.append("<h2>Set-level summary (worst first)</h2><table>")
    parts.append("<tr><th>set</th><th>N</th><th>chi2/N</th><th>k</th><th>Q2</th><th>xB</th><th>t</th></tr>")
    for _, row in set_table.iterrows():
        parts.append(
            f"<tr><td><a href='#set-{int(row['set'])}'>{int(row['set'])}</a></td>"
            f"<td>{int(row['n'])}</td><td>{row['chi2_per_point']:.3f}</td>"
            f"<td>{row['k']:.5g}</td><td>{row['q_squared']:.5g}</td>"
            f"<td>{row['x_b']:.5g}</td><td>{row['t']:.5g}</td></tr>"
        )
    parts.append("</table><h2>Individual fixed-kinematics sets</h2>")

    plot_rows = set_table if args.max_html_sets <= 0 else set_table.head(args.max_html_sets)
    for index, row in enumerate(plot_rows.itertuples(index=False), start=1):
        group = out[out["set"] == row.set].sort_values("phi")
        phi = group["phi"].to_numpy()
        y = group[args.target_col].to_numpy()
        err = group[args.error_col].to_numpy()
        pred = group["model_xsec_diff"].to_numpy()
        pulls = group["pull"].to_numpy()
        base = group[model.base_cols].median().to_numpy(np.float32)

        if args.plot_full_phi:
            lo, hi = -math.pi, math.pi
        else:
            span = float(phi.max() - phi.min()) if phi.max() > phi.min() else 1.0
            pad = 0.02 * span
            lo, hi = float(phi.min() - pad), float(phi.max() + pad)
        grid = np.linspace(lo, hi, args.phi_grid_points, dtype=np.float32)
        line_xsec, line_bsa, line_diff = model.predict_numpy(
            xsec_model, base, grid, args.phi_center
        )

        fig, axes = plt.subplots(2, 1, figsize=(8, 8.0), sharex=True)
        axes[0].plot(grid, line_diff, linewidth=2, label="hard-odd xsec-difference DNN")
        axes[0].errorbar(phi, y, yerr=err, fmt="o", markersize=4, capsize=2, label="data +/- total error")
        axes[0].plot(phi, pred, "x", markersize=4, label="model_xsec_diff at data bins")
        axes[0].axhline(0.0, linewidth=0.8)
        axes[0].set_ylabel("beam-helicity xsec difference")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8, loc="best")

        axes[1].plot(grid, line_bsa, linewidth=2, label="model_bsa")
        axes[1].scatter(phi, group[args.target_col] / group[args.xsec_target_col], s=13, label="central-value ratio data")
        axes[1].axhline(0.0, linewidth=0.8)
        axes[1].set_xlabel("phi [rad]")
        axes[1].set_ylabel("A_LU = xsec_diff / xsec")
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8, loc="best")

        fig.suptitle(
            f"set {int(row.set)} | median k={row.k:.4g}, Q2={row.q_squared:.4g}, "
            f"xB={row.x_b:.4g}, t={row.t:.4g}\n"
            f"N={len(group)}, chi2/N={np.mean(pulls**2):.3f}, "
            f"pull RMS={np.sqrt(np.mean(pulls**2)):.3f}, "
            f"median |pull|={np.median(np.abs(pulls)):.3f}, "
            f"max |pull|={np.max(np.abs(pulls)):.3g}",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        parts.append(
            f"<div class='plot' id='set-{int(row.set)}'><h3>Set {int(row.set)}</h3>"
            + fig_to_html_img(fig, args.html_dpi)
            + "</div>"
        )
        if index % args.plot_print_every == 0:
            print(f"embedded {index} set plots", flush=True)

    if args.max_html_sets > 0 and len(set_table) > args.max_html_sets:
        parts.append(
            f"<p>Only the worst {args.max_html_sets} sets were embedded because "
            "--max-html-sets was nonzero.  The CSV contains all sets.</p>"
        )
    parts.append("</body></html>")
    with open(outdir / "index.html", "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--csv", required=True)
    parser.add_argument("--xsec-checkpoint", default="xsec_surrogate/xsec_surrogate.pt")
    parser.add_argument(
        "--xsec-model-script",
        default=str(here / "dvcs_xsec_direct_dnn_optimized.py"),
        help="Companion script defining EvenDirectDNN.",
    )
    parser.add_argument("--outdir", default="xsec_diff_surrogate")
    parser.add_argument("--target-col", default="xsec_diff")
    parser.add_argument("--error-col", default="xsec_diff_err")
    parser.add_argument("--xsec-target-col", default="unp_beam_unp_target_xsec")
    parser.add_argument("--xsec-error-col", default="unp_beam_unp_target_xsec_err")
    parser.add_argument("--experiment-col", default="experiment_year")
    parser.add_argument("--keep-exact-duplicates", action="store_true")

    parser.add_argument("--phi-center", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--activation", choices=["tanh", "silu", "gelu"], default="silu")
    parser.add_argument("--output-limit", type=float, default=12.0)
    parser.add_argument(
        "--init-checkpoint", default=None,
        help="Optional xsec-difference checkpoint used to warm-start training.",
    )
    parser.add_argument(
        "--keep-final-state", action="store_true",
        help=(
            "Keep the final fine-tuned state instead of restoring the checkpoint "
            "with the best unregularized chi2 score. Useful for an explicit "
            "strong-smoothing polish pass."
        ),
    )

    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-drop1-frac", type=float, default=0.55)
    parser.add_argument("--lr-mid-factor", type=float, default=0.30)
    parser.add_argument("--lr-drop2-frac", type=float, default=0.85)
    parser.add_argument("--lr-final-factor", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--balance", choices=["point", "set", "experiment"], default="experiment")
    parser.add_argument("--worst-set-weight", type=float, default=0.0)
    parser.add_argument("--worst-set-fraction", type=float, default=0.10)

    parser.add_argument("--stage1-length-lambda", type=float, default=0.0)
    parser.add_argument("--stage1-curvature-lambda", type=float, default=0.0)
    parser.add_argument("--reg-start-frac", type=float, default=0.35)
    parser.add_argument("--reg-full-frac", type=float, default=0.75)

    parser.add_argument("--finetune-epochs", type=int, default=600)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-point-fraction", type=float, default=0.20)
    parser.add_argument("--length-lambda", type=float, default=0.02)
    parser.add_argument("--curvature-lambda", type=float, default=0.0005)
    parser.add_argument("--amplitude-l2", type=float, default=1e-6)
    parser.add_argument(
        "--worst-curve-fraction", type=float, default=0.0,
        help="Fraction of base-kinematic curves included in the worst-curve smoothness term.",
    )
    parser.add_argument(
        "--worst-curve-weight", type=float, default=0.0,
        help="Multiplier on worst-subset length and curvature in addition to their global means.",
    )

    parser.add_argument("--reg-grid-points", type=int, default=81)
    parser.add_argument("--reg-phi-min", type=float, default=-math.pi)
    parser.add_argument("--reg-phi-max", type=float, default=math.pi)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=300)
    parser.add_argument("--finetune-print-every", type=int, default=200)

    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--plot-full-phi", action="store_true", default=True)
    parser.add_argument("--phi-grid-points", type=int, default=400)
    parser.add_argument("--html-dpi", type=int, default=90)
    parser.add_argument("--max-html-sets", type=int, default=0, help="0 embeds all sets; positive embeds only the worst N.")
    parser.add_argument("--plot-print-every", type=int, default=50)
    parser.add_argument(
        "--report-only", action="store_true",
        help=(
            "Do not train. Rebuild index.html from the checkpoint, metrics, "
            "and predictions already present in --outdir."
        ),
    )
    parser.add_argument(
        "--diff-checkpoint", default=None,
        help="Checkpoint for --report-only; defaults to OUTDIR/xsec_diff_surrogate.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report_only:
        outdir = Path(args.outdir)
        diff_path = Path(args.diff_checkpoint) if args.diff_checkpoint else outdir / "xsec_diff_surrogate.pt"
        prediction_path = outdir / "predictions_with_pulls.csv"
        metrics_path = outdir / "metrics.json"
        for path in [diff_path, prediction_path, metrics_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required report input not found: {path}")

        xsec_module = load_xsec_module(Path(args.xsec_model_script))
        xsec_checkpoint = torch.load(args.xsec_checkpoint, map_location="cpu", weights_only=False)
        xsec_model = instantiate_xsec_model(xsec_module, xsec_checkpoint)
        diff_checkpoint = torch.load(diff_path, map_location="cpu", weights_only=False)
        model = instantiate_diff_model(diff_checkpoint)
        out = pd.read_csv(prediction_path)
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        make_html_report(model, xsec_model, out, metrics, args)
        print(f"Wrote {outdir / 'index.html'}", flush=True)
        return

    model, xsec_model, out, metrics, _ = train(args)
    if not args.no_html:
        make_html_report(model, xsec_model, out, metrics, args)
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)
    print(f"Wrote {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
