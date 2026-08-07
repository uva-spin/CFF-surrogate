#!/usr/bin/env python3
"""Propagate experimental DVCS errors into smooth ReH and ImH CFF surfaces.

This program is the replica stage that follows the central simultaneous CFF fit.
It keeps the strict common interpolation/extraction domain fixed and propagates
experimental cross-section and helicity-difference uncertainties through the
entire inference chain:

    measured (sigma_UU, Delta sigma_LU)
        -> matched observable surrogate replicas
        -> one simultaneous BKM CFF DNN per replica
        -> ReH/ImH percentile surfaces.

The two observable replicas share the same replica index and kinematic rows.  A
row-wise xsec/difference correlation coefficient may be supplied if an
experimental covariance estimate becomes available.  The default is zero
because the provided table contains no cross-observable covariance matrix.

For computational efficiency each observable replica is warm-started from the
validated central surrogate and gently fine-tuned only inside the fixed common
interpolation domain.  Each CFF replica is warm-started from the central CFF
checkpoint, preserving the selected quadratic BKM branch and isolating the
experimental component of the uncertainty.

Outputs are compact: one self-contained HTML report, CSV/JSON tables, and one
PyTorch ensemble checkpoint.  No per-plot PNG files or zip archive are written.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gc
import importlib.util
import io
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


# -----------------------------------------------------------------------------
# Dynamic imports keep this script usable beside the existing analysis scripts.
# -----------------------------------------------------------------------------

def load_python_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_stages(text: str) -> List[Tuple[int, float]]:
    stages: List[Tuple[int, float]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        epochs, lr = item.split(":", 1)
        stages.append((int(epochs), float(lr)))
    if not stages:
        raise ValueError("At least one EPOCHS:LR training stage is required")
    return stages


# -----------------------------------------------------------------------------
# Differentiable group-balanced losses.
# -----------------------------------------------------------------------------

def group_mean(values: torch.Tensor, index: torch.Tensor, n_groups: int) -> torch.Tensor:
    sums = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    sums.index_add_(0, index, values.reshape(-1))
    counts = torch.bincount(index, minlength=n_groups).to(values.dtype)
    return sums / counts.clamp_min(1.0)


def make_group_indices(frame: pd.DataFrame) -> Dict[str, object]:
    set_codes, set_labels = pd.factorize(frame["set"], sort=True)
    experiment_codes, experiment_labels = pd.factorize(frame["experiment"], sort=True)
    set_to_experiment = np.array(
        [experiment_codes[np.flatnonzero(set_codes == i)[0]] for i in range(len(set_labels))],
        dtype=np.int64,
    )
    return {
        "set_index": torch.tensor(set_codes, dtype=torch.long),
        "set_labels": np.asarray(set_labels),
        "set_experiment_index": torch.tensor(set_to_experiment, dtype=torch.long),
        "experiment_labels": np.asarray(experiment_labels),
    }


def experiment_balanced_loss(point_losses: torch.Tensor, groups: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    set_losses = group_mean(
        point_losses.reshape(-1),
        groups["set_index"],
        len(groups["set_labels"]),
    )
    experiment_losses = group_mean(
        set_losses,
        groups["set_experiment_index"],
        len(groups["experiment_labels"]),
    )
    return experiment_losses.mean(), set_losses.mean(), point_losses.mean()


# -----------------------------------------------------------------------------
# Experimental replicas.
# -----------------------------------------------------------------------------

def correlated_normals(rng: np.random.Generator, n: int, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    z1 = rng.standard_normal(n)
    z2_independent = rng.standard_normal(n)
    z2 = rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z2_independent
    return z1, z2


def sample_matched_observables(
    rng: np.random.Generator,
    xsec: np.ndarray,
    xsec_err: np.ndarray,
    diff: np.ndarray,
    diff_err: np.ndarray,
    rho: float,
    max_resample_rounds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Draw one matched replica.

    The signed difference is Gaussian.  The positive cross section uses a
    Gaussian with rejection of nonpositive draws.  Rejection is preferable to
    hard clipping because clipping would create an artificial pileup at zero.
    """
    zx, zd = correlated_normals(rng, len(xsec), rho)
    sampled_x = xsec + xsec_err * zx
    sampled_d = diff + diff_err * zd

    bad = sampled_x <= 0.0
    total_redraws = int(np.sum(bad))
    rounds = 0
    while np.any(bad) and rounds < max_resample_rounds:
        sampled_x[bad] = xsec[bad] + xsec_err[bad] * rng.standard_normal(np.sum(bad))
        bad = sampled_x <= 0.0
        total_redraws += int(np.sum(bad))
        rounds += 1
    if np.any(bad):
        # This should be extremely rare.  Use a tiny positive floor only after
        # repeated rejection has failed, and record the event in the summary.
        sampled_x[bad] = np.maximum(xsec[bad] * 1.0e-6, np.finfo(np.float32).tiny)
    return sampled_x.astype(np.float32), sampled_d.astype(np.float32), total_redraws


# -----------------------------------------------------------------------------
# Model reconstruction helpers.
# -----------------------------------------------------------------------------

def instantiate_xsec_model(xsec_module, checkpoint: Dict) -> nn.Module:
    model = xsec_module.EvenDirectDNN(
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
    return model


def instantiate_diff_model(diff_module, checkpoint: Dict) -> nn.Module:
    model = diff_module.OddXsecDiffDNN(
        base_cols=list(checkpoint["base_cols"]),
        hidden=int(checkpoint["hidden"]),
        depth=int(checkpoint["depth"]),
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        output_limit=float(checkpoint.get("output_limit", 12.0)),
        activation=str(checkpoint.get("activation", "silu")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def instantiate_cff_model(joint_module, checkpoint: Dict) -> nn.Module:
    model = joint_module.CFFNet(
        np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(checkpoint["feature_std"], dtype=np.float32),
        np.asarray(checkpoint["output_mean"], dtype=np.float32),
        np.asarray(checkpoint["output_std"], dtype=np.float32),
        hidden=int(checkpoint["hidden"]),
        depth=int(checkpoint["depth"]),
        output_limit=float(checkpoint.get("output_limit", 6.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# -----------------------------------------------------------------------------
# Observable-surrogate replica fine tuning.
# -----------------------------------------------------------------------------

def train_xsec_replica(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    error: torch.Tensor,
    groups: Dict[str, object],
    stages: Sequence[Tuple[int, float]],
    weight_decay: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    for epochs, lr in stages:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model.sigma(features)
            point_losses = torch.square((pred - target) / error)
            balanced, _, point_mean = experiment_balanced_loss(point_losses, groups)
            # A small pointwise term keeps the conventional global chi-square
            # represented while experiment balancing protects small data sets.
            loss = 0.85 * balanced + 0.15 * point_mean
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model.sigma(features)
        pull2 = torch.square((pred - target) / error)
        balanced, set_mean, point_mean = experiment_balanced_loss(pull2, groups)
    return {
        "xsec_replica_balanced_chi2": float(balanced),
        "xsec_replica_set_chi2": float(set_mean),
        "xsec_replica_point_chi2": float(point_mean),
    }


def train_diff_replica(
    model: nn.Module,
    features: torch.Tensor,
    sin_phi: torch.Tensor,
    paired_xsec_prediction: torch.Tensor,
    target: torch.Tensor,
    error: torch.Tensor,
    groups: Dict[str, object],
    stages: Sequence[Tuple[int, float]],
    weight_decay: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    paired_xsec_prediction = paired_xsec_prediction.detach()
    for epochs, lr in stages:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model.xsec_diff(features, sin_phi, paired_xsec_prediction)
            point_losses = torch.square((pred - target) / error)
            balanced, _, point_mean = experiment_balanced_loss(point_losses, groups)
            loss = 0.85 * balanced + 0.15 * point_mean
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model.xsec_diff(features, sin_phi, paired_xsec_prediction)
        pull2 = torch.square((pred - target) / error)
        balanced, set_mean, point_mean = experiment_balanced_loss(pull2, groups)
    return {
        "diff_replica_balanced_chi2": float(balanced),
        "diff_replica_set_chi2": float(set_mean),
        "diff_replica_point_chi2": float(point_mean),
    }


# -----------------------------------------------------------------------------
# Fast simultaneous CFF-replica fitting.
# -----------------------------------------------------------------------------

def make_cff_training_cache(frame: pd.DataFrame, poly: Dict[str, np.ndarray], joint_module) -> Dict[str, object]:
    # The CFF network is independent of phi and k.  Many rows have identical
    # (Q2,xB,t), so evaluate the DNN only once per unique hadronic point and map
    # the result back to all angular rows.  This makes replica fitting fast.
    hadronic = frame[["q_squared", "x_b", "t"]].to_numpy(np.float32)
    unique_hadronic, inverse = np.unique(hadronic, axis=0, return_inverse=True)
    features_unique = joint_module.cff_features(
        unique_hadronic[:, 0], unique_hadronic[:, 1], unique_hadronic[:, 2]
    )
    ix = joint_module.H_INDICES
    groups = make_group_indices(frame)
    return {
        "features_unique": torch.tensor(features_unique, dtype=torch.float32),
        "inverse": torch.tensor(inverse, dtype=torch.long),
        "c0_xsec": torch.tensor(poly["c0_xsec"].astype(np.float32)),
        "L_xsec": torch.tensor(poly["L_xsec"][:, ix].astype(np.float32)),
        "Q_xsec": torch.tensor(poly["Q_xsec"][:, ix][:, :, ix].astype(np.float32)),
        "c0_diff": torch.tensor(poly["c0_diff"].astype(np.float32)),
        "L_diff": torch.tensor(poly["L_diff"][:, ix].astype(np.float32)),
        "Q_diff": torch.tensor(poly["Q_diff"][:, ix][:, :, ix].astype(np.float32)),
        "xsec_error": torch.tensor(frame["unp_beam_unp_target_xsec_err"].to_numpy(np.float32)),
        "diff_error": torch.tensor(frame["xsec_diff_err"].to_numpy(np.float32)),
        "groups": groups,
        "unique_hadronic": unique_hadronic,
    }


def quadratic_observable(cff: torch.Tensor, c0: torch.Tensor, linear: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
    return c0 + torch.sum(linear * cff, dim=1) + torch.einsum("ni,nij,nj->n", cff, quad, cff)


def train_cff_replica(
    model: nn.Module,
    cache: Dict[str, object],
    target_xsec: torch.Tensor,
    target_diff: torch.Tensor,
    stages: Sequence[Tuple[int, float]],
    weight_decay: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    for epochs, lr in stages:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            optimizer.zero_grad()
            cff_unique = model(cache["features_unique"])
            cff = cff_unique[cache["inverse"]]
            pred_x = quadratic_observable(cff, cache["c0_xsec"], cache["L_xsec"], cache["Q_xsec"])
            pred_d = quadratic_observable(cff, cache["c0_diff"], cache["L_diff"], cache["Q_diff"])
            x_loss = torch.square((pred_x - target_xsec) / cache["xsec_error"])
            d_loss = torch.square((pred_d - target_diff) / cache["diff_error"])
            bx, _, px = experiment_balanced_loss(x_loss, cache["groups"])
            bd, _, pd = experiment_balanced_loss(d_loss, cache["groups"])
            balanced = 0.5 * (bx + bd)
            point_mean = 0.5 * (px + pd)
            loss = 0.90 * balanced + 0.10 * point_mean
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        cff_unique = model(cache["features_unique"])
        cff = cff_unique[cache["inverse"]]
        pred_x = quadratic_observable(cff, cache["c0_xsec"], cache["L_xsec"], cache["Q_xsec"])
        pred_d = quadratic_observable(cff, cache["c0_diff"], cache["L_diff"], cache["Q_diff"])
        x_loss = torch.square((pred_x - target_xsec) / cache["xsec_error"])
        d_loss = torch.square((pred_d - target_diff) / cache["diff_error"])
        bx, sx, px = experiment_balanced_loss(x_loss, cache["groups"])
        bd, sd, pd = experiment_balanced_loss(d_loss, cache["groups"])
    return {
        "cff_balanced_joint_chi2": float(0.5 * (bx + bd)),
        "cff_balanced_xsec_chi2": float(bx),
        "cff_balanced_diff_chi2": float(bd),
        "cff_point_joint_chi2": float(0.5 * (px + pd)),
        "cff_point_xsec_chi2": float(px),
        "cff_point_diff_chi2": float(pd),
        "cff_set_joint_chi2": float(0.5 * (sx + sd)),
    }


# -----------------------------------------------------------------------------
# CFF surface grids and support mask.
# -----------------------------------------------------------------------------

def parse_values(text: str, data: pd.Series, quantiles=(0.2, 0.5, 0.8)) -> List[float]:
    if str(text).strip():
        return [float(item.strip()) for item in str(text).split(",") if item.strip()]
    return [float(data.quantile(q)) for q in quantiles]


def support_setup(cff_sets: pd.DataFrame, checkpoint: Dict, neighbors: int, quantile: float, scale: float):
    # Import-independent version of the feature transformation used by CFFNet.
    q2 = cff_sets["q_squared"].to_numpy(float)
    xb = cff_sets["x_b"].to_numpy(float)
    t = cff_sets["t"].to_numpy(float)
    features = np.column_stack([np.log(q2), np.log(xb / (1.0 - xb)), np.log(-t)]).astype(np.float32)
    standardized = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    tree = cKDTree(standardized)
    k_ref = min(neighbors + 1, len(standardized))
    reference_distances = tree.query(standardized, k=k_ref)[0]
    reference_radius = reference_distances if reference_distances.ndim == 1 else reference_distances[:, -1]
    threshold = float(np.quantile(reference_radius, quantile) * scale)
    return tree, threshold


def support_mask(q2, xb, t, checkpoint: Dict, tree: cKDTree, threshold: float, neighbors: int):
    features = np.column_stack([
        np.log(np.asarray(q2, float)),
        np.log(np.asarray(xb, float) / (1.0 - np.asarray(xb, float))),
        np.log(-np.asarray(t, float)),
    ]).astype(np.float32)
    standardized = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    k = min(neighbors, tree.n)
    distances = tree.query(standardized, k=k)[0]
    kth = distances if np.ndim(distances) == 1 else distances[:, -1]
    return kth <= threshold, kth


def make_slice_grid(kind: str, value: float, cff_sets: pd.DataFrame, n: int):
    q2_min, q2_max = cff_sets["q_squared"].min(), cff_sets["q_squared"].max()
    xb_min, xb_max = cff_sets["x_b"].min(), cff_sets["x_b"].max()
    mt_min, mt_max = (-cff_sets["t"]).min(), (-cff_sets["t"]).max()

    if kind == "q2":
        a = np.linspace(xb_min, xb_max, n)
        b = np.linspace(mt_min, mt_max, n)
        A, B = np.meshgrid(a, b)
        q2 = np.full(A.size, value)
        xb = A.ravel()
        t = -B.ravel()
        xlabel, ylabel = r"$x_B$", r"$-t\,[\mathrm{GeV}^2]$"
        title = rf"$Q^2={value:.3g}\,\mathrm{{GeV}}^2$"
    elif kind == "xb":
        a = np.linspace(q2_min, q2_max, n)
        b = np.linspace(mt_min, mt_max, n)
        A, B = np.meshgrid(a, b)
        q2 = A.ravel()
        xb = np.full(A.size, value)
        t = -B.ravel()
        xlabel, ylabel = r"$Q^2\,[\mathrm{GeV}^2]$", r"$-t\,[\mathrm{GeV}^2]$"
        title = rf"$x_B={value:.3g}$"
    elif kind == "minus_t":
        a = np.linspace(q2_min, q2_max, n)
        b = np.linspace(xb_min, xb_max, n)
        A, B = np.meshgrid(a, b)
        q2 = A.ravel()
        xb = B.ravel()
        t = np.full(A.size, -value)
        xlabel, ylabel = r"$Q^2\,[\mathrm{GeV}^2]$", r"$x_B$"
        title = rf"$-t={value:.3g}\,\mathrm{{GeV}}^2$"
    else:
        raise ValueError(kind)
    return A, B, q2, xb, t, xlabel, ylabel, title


def cff_features_numpy(q2, xb, t) -> np.ndarray:
    return np.column_stack([
        np.log(np.asarray(q2, float)),
        np.log(np.asarray(xb, float) / (1.0 - np.asarray(xb, float))),
        np.log(-np.asarray(t, float)),
    ]).astype(np.float32)


def predict_cff_model(model: nn.Module, q2, xb, t, batch: int = 65536) -> np.ndarray:
    features = cff_features_numpy(q2, xb, t)
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch):
            outputs.append(model(torch.tensor(features[start:start + batch], dtype=torch.float32)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def ensemble_summary(values: np.ndarray) -> Dict[str, np.ndarray]:
    # values shape: [replica, point, CFF]
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0, ddof=1),
        "q16": np.quantile(values, 0.16, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q84": np.quantile(values, 0.84, axis=0),
    }


def figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=165, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def nearby_points(cff_sets: pd.DataFrame, kind: str, value: float, checkpoint: Dict, band: float) -> pd.DataFrame:
    features = cff_features_numpy(cff_sets["q_squared"], cff_sets["x_b"], cff_sets["t"])
    z = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    if kind == "q2":
        index = 0; raw = math.log(value)
    elif kind == "xb":
        index = 1; raw = math.log(value / (1.0 - value))
    else:
        index = 2; raw = math.log(value)
    z0 = (raw - float(checkpoint["feature_mean"][0, index])) / float(checkpoint["feature_std"][0, index])
    return cff_sets[np.abs(z[:, index] - z0) <= band].copy()


def local_xy(kind: str, points: pd.DataFrame):
    if kind == "q2":
        return points["x_b"].to_numpy(), (-points["t"]).to_numpy()
    if kind == "xb":
        return points["q_squared"].to_numpy(), (-points["t"]).to_numpy()
    return points["q_squared"].to_numpy(), points["x_b"].to_numpy()


def plot_band_slice(
    kind: str,
    value: float,
    A: np.ndarray,
    B: np.ndarray,
    mask: np.ndarray,
    central: np.ndarray,
    summary: Dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    points: pd.DataFrame,
    elev: float,
    azim: float,
):
    shape = A.shape
    mask2 = mask.reshape(shape)
    fields = {}
    for key in ["mean", "std", "q16", "q50", "q84"]:
        fields[key] = np.where(mask2[..., None], summary[key].reshape(shape + (2,)), np.nan)
    central_grid = np.where(mask2[..., None], central.reshape(shape + (2,)), np.nan)

    px, py = local_xy(kind, points) if len(points) else (np.array([]), np.array([]))
    fig = plt.figure(figsize=(14.2, 10.4))
    ax_re = fig.add_subplot(221, projection="3d")
    ax_im = fig.add_subplot(222, projection="3d")
    ax_re_unc = fig.add_subplot(223)
    ax_im_unc = fig.add_subplot(224)

    for component, ax, cmap, zlabel in [
        (0, ax_re, "coolwarm", r"$\mathrm{Re}\,\mathcal{H}_{\mathrm{eff}}$"),
        (1, ax_im, "viridis", r"$\mathrm{Im}\,\mathcal{H}_{\mathrm{eff}}$"),
    ]:
        mean_z = fields["mean"][..., component]
        low_z = fields["q16"][..., component]
        high_z = fields["q84"][..., component]
        ax.plot_surface(A, B, mean_z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.92)
        ax.plot_surface(A, B, low_z, color="tab:red", linewidth=0, alpha=0.18)
        ax.plot_surface(A, B, high_z, color="tab:red", linewidth=0, alpha=0.18)
        if len(points):
            col = "ReH" if component == 0 else "ImH"
            ax.scatter(px, py, points[col], s=17, c="black", depthshade=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(rf"Replica mean and 68% surfaces | {title}")

    for component, ax, cmap, label in [
        (0, ax_re_unc, "magma", r"$(q_{84}-q_{16})/2$ for $\mathrm{Re}\,\mathcal{H}$"),
        (1, ax_im_unc, "magma", r"$(q_{84}-q_{16})/2$ for $\mathrm{Im}\,\mathcal{H}$"),
    ]:
        half_width = 0.5 * (fields["q84"][..., component] - fields["q16"][..., component])
        contour = ax.contourf(A, B, half_width, levels=30, cmap=cmap)
        if len(points):
            ax.scatter(px, py, s=13, facecolors="none", edgecolors="white", linewidths=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label + " | " + title)
        fig.colorbar(contour, ax=ax, pad=0.02)
        ax.grid(alpha=0.12)

    fig.suptitle(
        "Experimental CFF replica surface: central curve is the ensemble mean; "
        "translucent surfaces are q16 and q84",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# -----------------------------------------------------------------------------
# Main analysis.
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="xsec_diff_surrogate/predictions_with_pulls.csv")
    p.add_argument("--domain-audit", default="joint_cff_extraction/common_domain_audit.csv")
    p.add_argument("--cff-sets", default="joint_cff_extraction/cff_sets.csv")
    p.add_argument("--xsec-checkpoint", default="xsec_surrogate/xsec_surrogate.pt")
    p.add_argument("--diff-checkpoint", default="xsec_diff_surrogate/xsec_diff_surrogate.pt")
    p.add_argument("--cff-checkpoint", default="joint_cff_extraction/cff_surrogate.pt")
    p.add_argument("--xsec-script", default="dvcs_xsec_direct_dnn_optimized.py")
    p.add_argument("--diff-script", default="dvcs_xsec_diff_direct_dnn_optimized.py")
    p.add_argument("--joint-script", default="dvcs_joint_cff_extraction.py")
    p.add_argument("--bkm-module", default="bkm10_observables_corrected.py")
    p.add_argument("--outdir", default="cff_experimental_replica_surfaces")
    p.add_argument("--n-replicas", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--xsec-diff-correlation", type=float, default=0.0)
    p.add_argument("--observable-stages", default="20:2e-4,20:5e-5")
    p.add_argument("--cff-stages", default="250:3e-4,150:1e-4,100:3e-5")
    p.add_argument("--weight-decay", type=float, default=1e-7)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--grid-size", type=int, default=58)
    p.add_argument("--q2-slices", default="")
    p.add_argument("--xb-slices", default="")
    p.add_argument("--minus-t-slices", default="")
    p.add_argument("--support-neighbors", type=int, default=5)
    p.add_argument("--support-quantile", type=float, default=0.80)
    p.add_argument("--support-scale", type=float, default=1.0)
    p.add_argument("--slice-band", type=float, default=0.25)
    p.add_argument("--elev", type=float, default=28.0)
    p.add_argument("--azim", type=float, default=-58.0)
    p.add_argument("--save-observable-states", action="store_true", help="Also retain all large xsec/difference replica states")
    p.add_argument("--print-every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (-0.999 < args.xsec_diff_correlation < 0.999):
        raise ValueError("--xsec-diff-correlation must be between -0.999 and 0.999")
    torch.set_num_threads(max(1, args.threads))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load code and checkpoints.
    xsec_module = load_python_module(Path(args.xsec_script), "xsec_replica_module")
    diff_module = load_python_module(Path(args.diff_script), "diff_replica_module")
    joint_module = load_python_module(Path(args.joint_script), "joint_cff_replica_module")
    bkm_module = load_python_module(Path(args.bkm_module), "bkm_replica_module")
    xsec_checkpoint = torch.load(args.xsec_checkpoint, map_location="cpu", weights_only=False)
    diff_checkpoint = torch.load(args.diff_checkpoint, map_location="cpu", weights_only=False)
    cff_checkpoint = torch.load(args.cff_checkpoint, map_location="cpu", weights_only=False)

    frame = pd.read_csv(args.predictions)
    audit = pd.read_csv(args.domain_audit)
    cff_sets = pd.read_csv(args.cff_sets)

    interpolation_sets = set(audit.loc[audit["interpolation_selected"].astype(bool), "set"].astype(int))
    extraction_sets = set(int(item) for item in cff_checkpoint["selected_sets"])
    interpolation_frame = frame[frame["set"].isin(interpolation_sets)].copy().reset_index(drop=True)
    extraction_frame = frame[frame["set"].isin(extraction_sets)].copy().reset_index(drop=True)
    cff_sets = cff_sets[cff_sets["set"].isin(extraction_sets)].copy().sort_values("set").reset_index(drop=True)

    # Observable feature tensors for the fixed common interpolation domain.
    base_cols = list(xsec_checkpoint["base_cols"])
    interp_base = interpolation_frame[base_cols].to_numpy(np.float32)
    interp_phi = interpolation_frame[["phi"]].to_numpy(np.float32)
    interp_features = torch.tensor(
        xsec_module.make_features(interp_base, interp_phi, float(xsec_checkpoint.get("phi_center", 0.0))),
        dtype=torch.float32,
    )
    interp_sin = torch.tensor(
        diff_module.sin_from_phi(interp_phi, float(diff_checkpoint.get("phi_center", 0.0))),
        dtype=torch.float32,
    )
    interp_x = interpolation_frame["unp_beam_unp_target_xsec"].to_numpy(np.float64)
    interp_ex = interpolation_frame["unp_beam_unp_target_xsec_err"].to_numpy(np.float64)
    interp_d = interpolation_frame["xsec_diff"].to_numpy(np.float64)
    interp_ed = interpolation_frame["xsec_diff_err"].to_numpy(np.float64)
    interp_groups = make_group_indices(interpolation_frame)

    # Map extraction rows into the interpolation frame so paired replica model
    # predictions are taken at exactly the central extraction coordinates.
    key_cols = ["source_row"] if "source_row" in frame.columns else ["set", "bin"]
    lookup = pd.Series(np.arange(len(interpolation_frame)), index=pd.MultiIndex.from_frame(interpolation_frame[key_cols]))
    extract_index = lookup.loc[pd.MultiIndex.from_frame(extraction_frame[key_cols])].to_numpy(int)

    # Build BKM polynomial once on the fixed central extraction domain.
    poly = joint_module.build_bkm_polynomial(extraction_frame, bkm_module)
    cff_cache = make_cff_training_cache(extraction_frame, poly, joint_module)

    observable_stages = parse_stages(args.observable_stages)
    cff_stages = parse_stages(args.cff_stages)
    rng = np.random.default_rng(args.seed)
    training_rows: List[Dict[str, float]] = []
    cff_state_dicts: List[Dict[str, torch.Tensor]] = []
    xsec_state_dicts: List[Dict[str, torch.Tensor]] = []
    diff_state_dicts: List[Dict[str, torch.Tensor]] = []

    # Store CFF predictions at the selected set centers for diagnostic bands.
    center_q2 = cff_sets["q_squared"].to_numpy(float)
    center_xb = cff_sets["x_b"].to_numpy(float)
    center_t = cff_sets["t"].to_numpy(float)
    center_replicas = np.empty((args.n_replicas, len(cff_sets), 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Null-replica calibration
    # ------------------------------------------------------------------
    # Warm-start/finite-epoch replica fitting is intentionally much cheaper
    # than repeating each central training from random initialization.  Even on
    # the unsmeared data, however, that short fine tuning can move the models
    # slightly.  The nonlinear BKM inverse can amplify a very small observable
    # shift into a visible CFF shift.  We therefore run the exact same short
    # pipeline once on the unsmeared data and subtract this deterministic
    # optimizer drift from every replica.  Only the response to the sampled
    # experimental fluctuation is retained.
    interp_x_t = torch.tensor(interp_x.reshape(-1, 1).astype(np.float32))
    interp_ex_t = torch.tensor(interp_ex.reshape(-1, 1).astype(np.float32))
    interp_d_t = torch.tensor(interp_d.reshape(-1, 1).astype(np.float32))
    interp_ed_t = torch.tensor(interp_ed.reshape(-1, 1).astype(np.float32))
    central_x_all = torch.tensor(interpolation_frame["model_xsec"].to_numpy(np.float32))
    central_d_all = torch.tensor(interpolation_frame["model_xsec_diff"].to_numpy(np.float32))

    null_xsec_model = instantiate_xsec_model(xsec_module, xsec_checkpoint)
    null_xsec_metrics = train_xsec_replica(
        null_xsec_model, interp_features, interp_x_t, interp_ex_t, interp_groups,
        observable_stages, args.weight_decay, args.grad_clip,
    )
    with torch.no_grad():
        null_xsec_all = null_xsec_model.sigma(interp_features).reshape(-1)

    null_diff_model = instantiate_diff_model(diff_module, diff_checkpoint)
    null_diff_metrics = train_diff_replica(
        null_diff_model, interp_features, interp_sin, null_xsec_all.reshape(-1, 1),
        interp_d_t, interp_ed_t, interp_groups,
        observable_stages, args.weight_decay, args.grad_clip,
    )
    with torch.no_grad():
        null_diff_all = null_diff_model.xsec_diff(
            interp_features, interp_sin, null_xsec_all.reshape(-1, 1)
        ).reshape(-1)

    # The CFF optimizer has its own much smaller deterministic finishing drift.
    # Fit the central observable surfaces with the replica schedule once; future
    # raw replica CFF predictions are calibrated by central + raw - null.
    central_target_x = central_x_all[extract_index].detach()
    central_target_d = central_d_all[extract_index].detach()
    null_cff_model = instantiate_cff_model(joint_module, cff_checkpoint)
    null_cff_metrics = train_cff_replica(
        null_cff_model, cff_cache, central_target_x, central_target_d,
        cff_stages, args.weight_decay, args.grad_clip,
    )
    central_cff_model = instantiate_cff_model(joint_module, cff_checkpoint)
    central_centers = predict_cff_model(central_cff_model, center_q2, center_xb, center_t)
    null_cff_centers = predict_cff_model(null_cff_model, center_q2, center_xb, center_t)

    calibration_summary = {
        "null_xsec_median_shift_in_error_units": float(np.median(
            np.abs(null_xsec_all.numpy() - central_x_all.numpy()) / interp_ex
        )),
        "null_diff_median_shift_in_error_units": float(np.median(
            np.abs(null_diff_all.numpy() - central_d_all.numpy()) / interp_ed
        )),
        "null_cff_center_ReH_rms_shift": float(np.sqrt(np.mean(
            np.square(null_cff_centers[:, 0] - central_centers[:, 0])
        ))),
        "null_cff_center_ImH_rms_shift": float(np.sqrt(np.mean(
            np.square(null_cff_centers[:, 1] - central_centers[:, 1])
        ))),
    }

    start_time = time.time()
    for replica in range(args.n_replicas):
        rep_seed = int(args.seed + 1009 * replica)
        np.random.seed(rep_seed)
        torch.manual_seed(rep_seed)
        sampled_x, sampled_d, redraws = sample_matched_observables(
            rng, interp_x, interp_ex, interp_d, interp_ed, args.xsec_diff_correlation
        )
        sampled_x_t = torch.tensor(sampled_x.reshape(-1, 1), dtype=torch.float32)
        sampled_d_t = torch.tensor(sampled_d.reshape(-1, 1), dtype=torch.float32)
        xsec_model = instantiate_xsec_model(xsec_module, xsec_checkpoint)
        x_metrics = train_xsec_replica(
            xsec_model, interp_features, sampled_x_t, interp_ex_t, interp_groups,
            observable_stages, args.weight_decay, args.grad_clip,
        )
        with torch.no_grad():
            paired_xsec_all = xsec_model.sigma(interp_features)

        diff_model = instantiate_diff_model(diff_module, diff_checkpoint)
        d_metrics = train_diff_replica(
            diff_model, interp_features, interp_sin, paired_xsec_all,
            sampled_d_t, interp_ed_t, interp_groups,
            observable_stages, args.weight_decay, args.grad_clip,
        )
        with torch.no_grad():
            paired_diff_all = diff_model.xsec_diff(interp_features, interp_sin, paired_xsec_all)

        # Remove deterministic short-training drift at the observable level.
        calibrated_x_all = central_x_all + (paired_xsec_all.reshape(-1) - null_xsec_all)
        calibrated_d_all = central_d_all + (paired_diff_all.reshape(-1) - null_diff_all)
        target_x = calibrated_x_all[extract_index].detach()
        target_d = calibrated_d_all[extract_index].detach()

        cff_model = instantiate_cff_model(joint_module, cff_checkpoint)
        c_metrics = train_cff_replica(
            cff_model, cff_cache, target_x, target_d,
            cff_stages, args.weight_decay, args.grad_clip,
        )
        raw_centers = predict_cff_model(cff_model, center_q2, center_xb, center_t)
        # Final null calibration isolates the experimental response of the CFF
        # estimator while retaining the validated central CFF surface.
        center_replicas[replica] = central_centers + (raw_centers - null_cff_centers)
        cff_state_dicts.append({key: value.detach().cpu().clone() for key, value in cff_model.state_dict().items()})
        if args.save_observable_states:
            xsec_state_dicts.append({key: value.detach().cpu().clone() for key, value in xsec_model.state_dict().items()})
            diff_state_dicts.append({key: value.detach().cpu().clone() for key, value in diff_model.state_dict().items()})

        training_rows.append({
            "replica": replica,
            "seed": rep_seed,
            "xsec_nonpositive_redraw_count": redraws,
            **x_metrics,
            **d_metrics,
            **c_metrics,
            "elapsed_sec": time.time() - start_time,
        })
        if (replica + 1) % max(1, args.print_every) == 0 or replica == 0 or replica + 1 == args.n_replicas:
            print(json.dumps(training_rows[-1]), flush=True)

        del xsec_model, diff_model, cff_model, sampled_x_t, sampled_d_t, paired_xsec_all, paired_diff_all
        gc.collect()

    training = pd.DataFrame(training_rows)
    training.to_csv(outdir / "replica_training_summary.csv", index=False)

    # Save the complete CFF ensemble in one file.  This permits arbitrary future
    # CFF-surface evaluation without retraining the replicas.
    ensemble_checkpoint = {
        "model_type": "experimental_replica_CFF_surface_ensemble",
        "n_replicas": args.n_replicas,
        "base_cff_checkpoint": cff_checkpoint,
        "raw_replica_state_dicts": cff_state_dicts,
        "null_cff_state_dict": {key: value.detach().cpu().clone() for key, value in null_cff_model.state_dict().items()},
        "null_xsec_state_dict": {key: value.detach().cpu().clone() for key, value in null_xsec_model.state_dict().items()},
        "null_diff_state_dict": {key: value.detach().cpu().clone() for key, value in null_diff_model.state_dict().items()},
        "calibration_formula": "F_replica_calibrated(x) = F_central(x) + F_replica_raw(x) - F_null_raw(x)",
        "calibration_summary": calibration_summary,
        "null_xsec_metrics": null_xsec_metrics,
        "null_diff_metrics": null_diff_metrics,
        "null_cff_metrics": null_cff_metrics,
        "selected_sets": sorted(extraction_sets),
        "interpolation_sets": sorted(interpolation_sets),
        "sampling": {
            "xsec": "Gaussian with rejection of nonpositive values",
            "xsec_diff": "signed Gaussian",
            "xsec_diff_row_correlation": args.xsec_diff_correlation,
            "covariance_note": "No experimental cross-observable or bin-to-bin covariance matrix was provided; default replicas are row-wise independent.",
        },
        "observable_stages": observable_stages,
        "cff_stages": cff_stages,
    }
    if args.save_observable_states:
        ensemble_checkpoint["xsec_replica_state_dicts"] = xsec_state_dicts
        ensemble_checkpoint["diff_replica_state_dicts"] = diff_state_dicts
    torch.save(ensemble_checkpoint, outdir / "cff_replica_ensemble.pt")

    # Set-center experimental bands and CFF correlation.
    center_summary = ensemble_summary(center_replicas)
    center_output = cff_sets.copy()
    for component, name in enumerate(["ReH", "ImH"]):
        center_output[f"{name}_central"] = central_centers[:, component]
        for stat in ["mean", "std", "q16", "q50", "q84"]:
            center_output[f"{name}_{stat}"] = center_summary[stat][:, component]
    center_corr = []
    for i in range(len(center_output)):
        center_corr.append(float(np.corrcoef(center_replicas[:, i, 0], center_replicas[:, i, 1])[0, 1]))
    center_output["corr_ReH_ImH"] = center_corr
    center_output.to_csv(outdir / "cff_set_experimental_bands.csv", index=False)

    # Prepare support mask and slices.
    tree, threshold = support_setup(
        cff_sets, cff_checkpoint, args.support_neighbors,
        args.support_quantile, args.support_scale,
    )
    q2_slices = parse_values(args.q2_slices, cff_sets["q_squared"])
    xb_slices = parse_values(args.xb_slices, cff_sets["x_b"])
    mt_slices = parse_values(args.minus_t_slices, -cff_sets["t"])
    slice_specs = [("q2", v) for v in q2_slices] + [("xb", v) for v in xb_slices] + [("minus_t", v) for v in mt_slices]

    html_images: List[Tuple[str, str]] = []
    surface_rows: List[pd.DataFrame] = []
    first_half_metrics: List[float] = []
    second_half_metrics: List[float] = []

    # Reconstruct models once for surface evaluation.
    replica_models: List[nn.Module] = []
    for state in cff_state_dicts:
        model = instantiate_cff_model(joint_module, cff_checkpoint)
        model.load_state_dict(state)
        model.eval()
        replica_models.append(model)

    for kind, value in slice_specs:
        A, B, q2, xb, t, xlabel, ylabel, title = make_slice_grid(kind, value, cff_sets, args.grid_size)
        mask, distance = support_mask(
            q2, xb, t, cff_checkpoint, tree, threshold, args.support_neighbors
        )
        central = predict_cff_model(central_cff_model, q2, xb, t)
        null_surface = predict_cff_model(null_cff_model, q2, xb, t)
        values = np.empty((args.n_replicas, len(q2), 2), dtype=np.float32)
        for r, model in enumerate(replica_models):
            raw = predict_cff_model(model, q2, xb, t)
            values[r] = central + (raw - null_surface)
        summary = ensemble_summary(values)

        # Split-half convergence diagnostic for the quoted 68% half-width.
        split = args.n_replicas // 2
        if split >= 5 and args.n_replicas - split >= 5:
            first = ensemble_summary(values[:split])
            second = ensemble_summary(values[split:])
            for comp in range(2):
                hw1 = 0.5 * (first["q84"][:, comp] - first["q16"][:, comp])
                hw2 = 0.5 * (second["q84"][:, comp] - second["q16"][:, comp])
                valid = mask & np.isfinite(hw1) & np.isfinite(hw2)
                denom = np.maximum(0.5 * (np.abs(hw1[valid]) + np.abs(hw2[valid])), 1e-8)
                rel = np.abs(hw1[valid] - hw2[valid]) / denom
                (first_half_metrics if comp == 0 else second_half_metrics).append(float(np.median(rel)))

        points = nearby_points(cff_sets, kind, value, cff_checkpoint, args.slice_band)
        fig = plot_band_slice(
            kind, value, A, B, mask, central, summary,
            xlabel, ylabel, title, points, args.elev, args.azim,
        )
        html_images.append((f"Experimental 68% CFF surfaces: {title}", figure_to_base64(fig)))

        row = pd.DataFrame({
            "slice_kind": kind,
            "slice_value": value,
            "q_squared": q2,
            "x_b": xb,
            "t": t,
            "inside_support": mask,
            "support_distance": distance,
            "ReH_central": central[:, 0],
            "ReH_mean": summary["mean"][:, 0],
            "ReH_std": summary["std"][:, 0],
            "ReH_q16": summary["q16"][:, 0],
            "ReH_q50": summary["q50"][:, 0],
            "ReH_q84": summary["q84"][:, 0],
            "ImH_central": central[:, 1],
            "ImH_mean": summary["mean"][:, 1],
            "ImH_std": summary["std"][:, 1],
            "ImH_q16": summary["q16"][:, 1],
            "ImH_q50": summary["q50"][:, 1],
            "ImH_q84": summary["q84"][:, 1],
        })
        # Pointwise ReH-ImH experimental correlation across replicas.
        correlations = np.empty(len(row), dtype=float)
        for i in range(len(row)):
            correlations[i] = np.corrcoef(values[:, i, 0], values[:, i, 1])[0, 1]
        row["corr_ReH_ImH"] = correlations
        surface_rows.append(row)

    surface_table = pd.concat(surface_rows, ignore_index=True)
    surface_table.to_csv(outdir / "cff_surface_experimental_bands.csv", index=False)

    metrics = {
        "model_type": "matched_observable_replica_to_CFF_surface_ensemble",
        "n_replicas": args.n_replicas,
        "interpolation_sets": len(interpolation_sets),
        "interpolation_rows": len(interpolation_frame),
        "extraction_sets": len(extraction_sets),
        "extraction_rows": len(extraction_frame),
        "sampling": ensemble_checkpoint["sampling"],
        "median_replica_xsec_point_chi2": float(training["xsec_replica_point_chi2"].median()),
        "median_replica_diff_point_chi2": float(training["diff_replica_point_chi2"].median()),
        "median_replica_cff_point_joint_chi2": float(training["cff_point_joint_chi2"].median()),
        "median_center_ReH_std": float(center_output["ReH_std"].median()),
        "median_center_ImH_std": float(center_output["ImH_std"].median()),
        "median_center_ReH_68_half_width": float((0.5 * (center_output["ReH_q84"] - center_output["ReH_q16"])).median()),
        "median_center_ImH_68_half_width": float((0.5 * (center_output["ImH_q84"] - center_output["ImH_q16"])).median()),
        "median_center_ReH_ImH_correlation": float(center_output["corr_ReH_ImH"].median()),
        "split_half_median_relative_68_width_difference_ReH": float(np.median(first_half_metrics)) if first_half_metrics else None,
        "split_half_median_relative_68_width_difference_ImH": float(np.median(second_half_metrics)) if second_half_metrics else None,
        "support_definition": {
            "neighbors": args.support_neighbors,
            "reference_quantile": args.support_quantile,
            "scale": args.support_scale,
            "threshold_standardized_feature_distance": threshold,
        },
        "null_replica_calibration": calibration_summary,
        "uncertainty_scope": "experimental replica component only; deterministic warm-start drift is null-calibrated; algorithmic and methodological components remain separate",
        "elapsed_sec": time.time() - start_time,
    }
    with open(outdir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    # Self-contained HTML report.
    experiment_summary = (
        center_output.groupby("experiment")
        .agg(
            n_sets=("set", "nunique"),
            ReH_median=("ReH_mean", "median"),
            ReH_median_1sigma=("ReH_std", "median"),
            ImH_median=("ImH_mean", "median"),
            ImH_median_1sigma=("ImH_std", "median"),
            median_corr=("corr_ReH_ImH", "median"),
        )
        .reset_index()
    )
    html = [
        "<html><head><meta charset='utf-8'><title>Experimental CFF replica surfaces</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:auto;padding:24px;line-height:1.45}img{max-width:100%;height:auto}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:5px 8px}code{background:#eee;padding:2px 4px}</style></head><body>",
        "<h1>Experimental 1σ surfaces for effective ReH and ImH</h1>",
        f"<p><b>Replica ensemble:</b> {args.n_replicas} matched cross-section/difference replicas. The strict central domain is frozen at {len(interpolation_sets)} interpolation sets and {len(extraction_sets)} CFF-extraction sets.</p>",
        "<p><b>Definition:</b> each translucent lower/upper surface is the pointwise 16th/84th percentile of independently refitted CFF DNNs. These are experimental-replica intervals, not local Hessian errors and not optimizer-seed spread.</p>",
        "<p><b>Null-replica calibration:</b> the same finite warm-start schedule is first run on the unsmeared central data. Its deterministic observable/CFF drift is subtracted replica by replica, so the quoted spread reflects sampled experimental fluctuations rather than a change of optimizer objective.</p>",
        "<p><b>Covariance limitation:</b> no cross-observable or bin-to-bin covariance matrix was supplied. The baseline replicas therefore use row-wise total-error Gaussian sampling with zero xsec/difference correlation. The same replica index and common kinematic rows are retained through both observable surrogates and the BKM CFF fit.</p>",
        f"<p><b>Typical center uncertainty:</b> median 68% half-width = {metrics['median_center_ReH_68_half_width']:.4g} for ReH and {metrics['median_center_ImH_68_half_width']:.4g} for ImH. Median ReH–ImH replica correlation = {metrics['median_center_ReH_ImH_correlation']:.3f}.</p>",
        f"<p><b>Split-half stability:</b> median relative difference of 68% widths is {metrics['split_half_median_relative_68_width_difference_ReH']:.3f} for ReH and {metrics['split_half_median_relative_68_width_difference_ImH']:.3f} for ImH.</p>",
        "<h2>Summary by experiment</h2>",
        experiment_summary.to_html(index=False, float_format=lambda x: f"{x:.4g}"),
    ]
    for title, image in html_images:
        html.extend([f"<h2>{title}</h2>", f"<img src='data:image/png;base64,{image}'>"])
    html.extend(["</body></html>"])
    (outdir / "index.html").write_text("\n".join(html), encoding="utf-8")

    print("FINAL_METRICS " + json.dumps(metrics), flush=True)
    print(f"Wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
