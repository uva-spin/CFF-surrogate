#!/usr/bin/env python3
"""
Optimized direct-DNN surrogate for the unpolarized DVCS cross section.

The model has no Bernstein or Fourier expansion.  It directly learns

    log(sigma) = DNN(k, Q2, xB, t, u),
    u = [1 - cos(phi - phi_center)] / 2.

Using u instead of raw phi makes sigma(+phi)=sigma(-phi) exact.  Predicting
log(sigma) and exponentiating makes the cross section positive.

The training recipe is designed for the very uneven experimental coverage in
this data set:

  * exact duplicate rows are removed from the training loss by default;
  * an initial log-space stage gets all cross-section scales into the right
    neighborhood before switching to the quoted experimental chi-square;
  * the default loss gives equal aggregate weight to each experiment-year,
    rather than allowing the largest duplicated data block to dominate;
  * a low-learning-rate finishing stage adds mild arc-length and curvature
    regularization in phi without undoing the fit.

Plot output is intentionally simple: every diagnostic plot is embedded in one
self-contained index.html.  No per-set PNG files and no zip archive are made.

Example
-------
python dvcs_xsec_direct_dnn_optimized.py \
    --csv refined_cross_section_data_v2_1.csv \
    --outdir xsec_surrogate

Optional experiment convention test
-----------------------------------
A repeatable scale option is provided so a suspected convention conversion can
be tested without manually editing the CSV.  For example:

python dvcs_xsec_direct_dnn_optimized.py \
    --csv refined_cross_section_data_v2_1.csv \
    --outdir xsec_surrogate_halla2020_times_pi \
    --experiment-year-scale HALLA_2020=3.141592653589793
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def u_from_phi(phi: np.ndarray, phi_center: float = 0.0) -> np.ndarray:
    """Map phi to an exactly even variable u in [0,1]."""
    phi = np.asarray(phi, dtype=np.float32).reshape(-1, 1)
    return ((1.0 - np.cos(phi - float(phi_center))) / 2.0).astype(np.float32)


def make_features(base_raw: np.ndarray, phi: np.ndarray, phi_center: float = 0.0) -> np.ndarray:
    """Construct [base kinematics, u(phi)] features."""
    base_raw = np.asarray(base_raw, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32).reshape(-1, 1)
    if base_raw.ndim == 1:
        base_raw = np.repeat(base_raw[None, :], len(phi), axis=0)
    if len(base_raw) != len(phi):
        raise ValueError(f"base_raw/phi length mismatch: {len(base_raw)} vs {len(phi)}")
    return np.concatenate([base_raw, u_from_phi(phi, phi_center)], axis=1).astype(np.float32)


class EvenDirectDNN(nn.Module):
    """Direct positive, hard-even DNN for the DVCS cross section."""

    def __init__(
        self,
        base_cols: List[str],
        hidden: int,
        depth: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        log_y_mean: float,
        log_y_std: float,
        output_limit: float = 10.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.base_cols = list(base_cols)
        self.feature_cols = list(base_cols) + ["u"]
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.output_limit = float(output_limit)
        self.activation = str(activation)

        # Buffers are saved with the checkpoint but are not trainable.
        self.register_buffer("x_mean", torch.tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(feature_std, dtype=torch.float32))
        self.register_buffer("ym", torch.tensor(float(log_y_mean), dtype=torch.float32))
        self.register_buffer("ys", torch.tensor(float(log_y_std), dtype=torch.float32))

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

    def z(self, features: torch.Tensor) -> torch.Tensor:
        x = (features - self.x_mean) / self.x_std
        raw = self.net(x)
        # A smooth bound is a numerical guard in very sparse extrapolation
        # regions; output_limit=0 disables it.
        if self.output_limit > 0:
            raw = self.output_limit * torch.tanh(raw / self.output_limit)
        return raw

    def log_sigma(self, features: torch.Tensor) -> torch.Tensor:
        return self.ym + self.ys * self.z(features)

    def sigma(self, features: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_sigma(features))

    def predict_numpy(
        self,
        base_raw: np.ndarray,
        phi: np.ndarray,
        phi_center: float = 0.0,
        batch: int = 16384,
    ) -> np.ndarray:
        self.eval()
        feats = make_features(base_raw, phi, phi_center)
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(feats), batch):
                ft = torch.tensor(feats[i:i + batch], dtype=torch.float32)
                outputs.append(self.sigma(ft).cpu().numpy())
        return np.concatenate(outputs, axis=0).ravel()


def parse_scale_rules(items: List[str]) -> Dict[str, float]:
    """Parse repeated YEAR=FACTOR command-line entries."""
    rules: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Scale rule must look like YEAR=FACTOR, got {item!r}")
        key, value = item.split("=", 1)
        rules[key.strip()] = float(value)
    return rules


def apply_experiment_year_scales(
    df: pd.DataFrame,
    rules: Mapping[str, float],
    target_col: str,
    error_col: str,
) -> pd.DataFrame:
    """Apply an explicitly requested multiplicative convention conversion.

    A cross-section unit/convention conversion must scale the central value and
    all corresponding uncertainty columns by the same factor.
    """
    if not rules:
        return df
    if "experiment_year" not in df.columns:
        raise ValueError("experiment_year is required for --experiment-year-scale")
    out = df.copy()
    scale_cols = [target_col, error_col]
    for extra in (f"{target_col}_errstat", f"{target_col}_errsyst"):
        if extra in out.columns:
            scale_cols.append(extra)
    # The supplied data use these explicit names rather than target_col suffixes.
    for extra in ("unp_beam_unp_target_xsec_errstat", "unp_beam_unp_target_xsec_errsyst"):
        if extra in out.columns and extra not in scale_cols:
            scale_cols.append(extra)

    for year, factor in rules.items():
        mask = out["experiment_year"].astype(str).eq(str(year))
        if not mask.any():
            raise ValueError(f"No rows matched experiment_year={year!r}")
        out.loc[mask, scale_cols] = out.loc[mask, scale_cols] * float(factor)
    return out


def group_mean(values: torch.Tensor, index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Differentiable mean of values within integer-labeled groups."""
    sums = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    sums.index_add_(0, index, values.reshape(-1))
    counts = torch.bincount(index, minlength=n_groups).to(values.dtype)
    return sums / counts.clamp_min(1.0)


def linear_ramp(x: float, start: float, end: float) -> float:
    if x <= start:
        return 0.0
    if x >= end:
        return 1.0
    return (x - start) / (end - start)


def metrics_np(y: np.ndarray, err: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    pull = (pred.ravel() - y.ravel()) / err.ravel()
    ap = np.abs(pull)
    return {
        "n_points": int(len(pull)),
        "chi2_per_point": float(np.mean(pull ** 2)),
        "pull_rms": float(np.sqrt(np.mean(pull ** 2))),
        "pull_mean": float(np.mean(pull)),
        "pull_std": float(np.std(pull)),
        "median_abs_pull": float(np.median(ap)),
        "frac_abs_pull_lt_1": float(np.mean(ap < 1.0)),
        "frac_abs_pull_lt_2": float(np.mean(ap < 2.0)),
        "frac_abs_pull_lt_3": float(np.mean(ap < 3.0)),
        "max_abs_pull": float(np.max(ap)),
    }


def load_data(args: argparse.Namespace) -> Tuple:
    original = pd.read_csv(args.csv)
    scale_rules = parse_scale_rules(args.experiment_year_scale)
    df = apply_experiment_year_scales(
        original, scale_rules, args.target_col, args.error_col
    )

    base_cols = ["q_squared", "x_b", "t"]
    if args.include_k:
        base_cols = ["k"] + base_cols

    required = base_cols + ["phi", args.target_col, args.error_col, "set"]
    if args.balance == "experiment":
        required.append("experiment_year")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    use = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    use = use[(use[args.target_col] > 0) & (use[args.error_col] > 0)].copy()

    # Exact duplicate CLAS rows should not silently receive double statistical
    # weight.  The original row count and removed count are retained in metrics.
    duplicate_subset = base_cols + ["phi", args.target_col, args.error_col, "set"]
    duplicate_count = int(len(use) - len(use.drop_duplicates(subset=duplicate_subset)))
    if not args.keep_exact_duplicates:
        use = use.drop_duplicates(subset=duplicate_subset).copy()

    base_all = use[base_cols].to_numpy(np.float32)
    phi_all = use[["phi"]].to_numpy(np.float32)
    y_all = use[[args.target_col]].to_numpy(np.float32)
    err_all = use[[args.error_col]].to_numpy(np.float32)
    features_all = make_features(base_all, phi_all, args.phi_center)
    base_unique = np.unique(base_all, axis=0).astype(np.float32)

    feature_mean = features_all.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = features_all.std(axis=0, keepdims=True).astype(np.float32)
    feature_std[feature_std == 0] = 1.0
    log_y = np.log(y_all)

    set_codes, set_values = pd.factorize(use["set"], sort=True)
    if args.balance == "experiment":
        set_years = (
            use[["set", "experiment_year"]]
            .drop_duplicates("set")
            .set_index("set")
            .loc[set_values, "experiment_year"]
        )
        experiment_codes, experiment_values = pd.factorize(set_years, sort=True)
    else:
        experiment_codes = np.zeros(len(set_values), dtype=np.int64)
        experiment_values = np.asarray(["all"])

    return (
        use,
        base_cols,
        features_all,
        base_unique,
        y_all,
        err_all,
        feature_mean,
        feature_std,
        float(log_y.mean()),
        float(log_y.std() if log_y.std() > 0 else 1.0),
        duplicate_count,
        set_codes.astype(np.int64),
        set_values,
        experiment_codes.astype(np.int64),
        experiment_values,
        scale_rules,
    )


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
    experiment_mean = group_mean(
        set_losses, set_experiment_index, n_experiments
    ).mean()
    if balance == "point":
        chosen = point_mean
    elif balance == "set":
        chosen = set_mean
    else:
        chosen = experiment_mean
    return chosen, set_losses, point_mean


def make_regularization_grid(
    base_unique: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, int, int, float]:
    phi_grid = np.linspace(
        args.reg_phi_min,
        args.reg_phi_max,
        args.reg_grid_points,
        dtype=np.float32,
    ).reshape(-1, 1)
    reg_base = np.repeat(base_unique, args.reg_grid_points, axis=0)
    reg_phi = np.tile(phi_grid, (len(base_unique), 1))
    reg_features = torch.tensor(
        make_features(reg_base, reg_phi, args.phi_center), dtype=torch.float32
    )
    dphi = float(
        (args.reg_phi_max - args.reg_phi_min) / (args.reg_grid_points - 1)
    )
    return reg_features, len(base_unique), args.reg_grid_points, dphi


def smoothness_losses(
    model: EvenDirectDNN,
    reg_features: torch.Tensor,
    n_base: int,
    n_grid: int,
    dphi: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_grid = model.log_sigma(reg_features).reshape(n_base, n_grid)
    d1 = (log_grid[:, 1:] - log_grid[:, :-1]) / dphi
    length_loss = torch.mean(torch.sqrt(1.0 + d1 ** 2) - 1.0)
    d2 = (
        log_grid[:, 2:]
        - 2.0 * log_grid[:, 1:-1]
        + log_grid[:, :-2]
    ) / (dphi ** 2)
    curvature_loss = torch.mean(d2 ** 2)
    return length_loss, curvature_loss


def evaluate_model(
    model: EvenDirectDNN,
    features_t: torch.Tensor,
    y: np.ndarray,
    err: np.ndarray,
    use: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float], pd.Series]:
    with torch.no_grad():
        pred = model.sigma(features_t).cpu().numpy().ravel()
    metrics = metrics_np(y, err, pred)
    pull = (pred - y.ravel()) / err.ravel()
    temp = use[["set"]].copy()
    temp["pull2"] = pull ** 2
    per_set = temp.groupby("set")["pull2"].mean()
    metrics["mean_set_chi2"] = float(per_set.mean())
    metrics["max_set_chi2"] = float(per_set.max())
    return pred, metrics, per_set


def train(args: argparse.Namespace):
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.outdir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (
        use,
        base_cols,
        features_all,
        base_unique,
        y_all,
        err_all,
        feature_mean,
        feature_std,
        log_y_mean,
        log_y_std,
        duplicate_count,
        set_codes,
        set_values,
        experiment_codes,
        experiment_values,
        scale_rules,
    ) = load_data(args)

    model = EvenDirectDNN(
        base_cols=base_cols,
        hidden=args.hidden,
        depth=args.depth,
        feature_mean=feature_mean,
        feature_std=feature_std,
        log_y_mean=log_y_mean,
        log_y_std=log_y_std,
        output_limit=args.output_limit,
        activation=args.activation,
    )

    features_t = torch.tensor(features_all, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.float32)
    err_t = torch.tensor(err_all, dtype=torch.float32)
    log_y_t = torch.log(y_t)

    # This is the standard deviation in log space of a log-normal variable with
    # the same relative uncertainty.  It gives a well-scaled pretraining loss.
    log_err_t = torch.sqrt(torch.log1p((err_t / y_t) ** 2)).clamp_min(
        args.min_log_error
    )

    set_index_t = torch.tensor(set_codes, dtype=torch.long)
    set_experiment_t = torch.tensor(experiment_codes, dtype=torch.long)
    n_sets = len(set_values)
    n_experiments = len(experiment_values)

    reg_features_t, n_base, n_grid, dphi = make_regularization_grid(
        base_unique, args
    )

    history: List[Dict[str, float]] = []
    start_time = time.time()

    # ------------------------- Stage 1: global fit -------------------------
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        frac = epoch / max(1, args.epochs)
        if frac > args.lr_drop2_frac:
            lr = args.lr * args.lr_final_factor
        elif frac > args.lr_drop1_frac:
            lr = args.lr * args.lr_mid_factor
        else:
            lr = args.lr
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad()
        log_pred = model.log_sigma(features_t)
        pred = torch.exp(log_pred)
        sigma_pull2 = ((pred - y_t) / err_t) ** 2
        log_pull2 = ((log_pred - log_y_t) / log_err_t) ** 2

        # Early log-space training avoids enormous initial gradients across the
        # many orders of magnitude in the cross section.  The objective then
        # becomes the actual quoted experimental chi-square.
        sigma_mix = linear_ramp(
            frac, args.sigma_loss_start_frac, args.sigma_loss_full_frac
        )
        point_losses = (1.0 - sigma_mix) * log_pull2 + sigma_mix * sigma_pull2
        data_loss, set_losses, point_mean = balanced_loss(
            point_losses,
            set_index_t,
            n_sets,
            set_experiment_t,
            n_experiments,
            args.balance,
        )

        if args.worst_set_weight > 0:
            n_top = max(1, int(math.ceil(args.worst_set_fraction * n_sets)))
            worst_loss = torch.topk(set_losses, n_top).values.mean()
        else:
            worst_loss = torch.tensor(0.0)

        length_loss, curvature_loss = smoothness_losses(
            model, reg_features_t, n_base, n_grid, dphi
        )
        reg_scale = linear_ramp(frac, args.reg_start_frac, args.reg_full_frac)
        loss = (
            data_loss
            + args.worst_set_weight * worst_loss
            + reg_scale
            * (
                args.stage1_length_lambda * length_loss
                + args.stage1_curvature_lambda * curvature_loss
            )
        )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            pred_np, eval_metrics, _ = evaluate_model(
                model, features_t, y_all, err_all, use
            )
            score = eval_metrics["chi2_per_point"] + 0.20 * eval_metrics["mean_set_chi2"]
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            record = {
                "stage": 1,
                "epoch": epoch,
                "chi2_per_point": eval_metrics["chi2_per_point"],
                "mean_set_chi2": eval_metrics["mean_set_chi2"],
                "max_set_chi2": eval_metrics["max_set_chi2"],
                "data_loss": float(data_loss.detach()),
                "point_mean_loss": float(point_mean.detach()),
                "length_loss": float(length_loss.detach()),
                "curvature_loss": float(curvature_loss.detach()),
                "sigma_loss_mix": float(sigma_mix),
                "lr": float(lr),
                "elapsed_sec": float(time.time() - start_time),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    model.load_state_dict(best_state)

    # -------------------- Stage 2: gentle smooth finish --------------------
    if args.finetune_epochs > 0:
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.finetune_lr, weight_decay=0.0
        )
        for epoch in range(1, args.finetune_epochs + 1):
            frac = epoch / max(1, args.finetune_epochs)
            if frac < 0.60:
                lr = args.finetune_lr
            elif frac < 0.85:
                lr = args.finetune_lr * 0.25
            else:
                lr = args.finetune_lr * 0.08
            for group in opt.param_groups:
                group["lr"] = lr

            opt.zero_grad()
            pred = model.sigma(features_t)
            point_losses = ((pred - y_t) / err_t) ** 2
            experiment_loss, set_losses, point_mean = balanced_loss(
                point_losses,
                set_index_t,
                n_sets,
                set_experiment_t,
                n_experiments,
                args.balance,
            )
            # A small pointwise term keeps the conventional global chi-square
            # from drifting while the balanced loss protects sparse experiments.
            data_loss = (
                args.finetune_point_fraction * point_mean
                + (1.0 - args.finetune_point_fraction) * experiment_loss
            )
            length_loss, curvature_loss = smoothness_losses(
                model, reg_features_t, n_base, n_grid, dphi
            )
            loss = (
                data_loss
                + args.length_lambda * length_loss
                + args.curvature_lambda * curvature_loss
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if (
                epoch == 1
                or epoch % args.finetune_print_every == 0
                or epoch == args.finetune_epochs
            ):
                _, eval_metrics, _ = evaluate_model(
                    model, features_t, y_all, err_all, use
                )
                score = eval_metrics["chi2_per_point"] + 0.20 * eval_metrics["mean_set_chi2"]
                if score < best_score:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                record = {
                    "stage": 2,
                    "epoch": epoch,
                    "chi2_per_point": eval_metrics["chi2_per_point"],
                    "mean_set_chi2": eval_metrics["mean_set_chi2"],
                    "max_set_chi2": eval_metrics["max_set_chi2"],
                    "data_loss": float(data_loss.detach()),
                    "point_mean_loss": float(point_mean.detach()),
                    "length_loss": float(length_loss.detach()),
                    "curvature_loss": float(curvature_loss.detach()),
                    "sigma_loss_mix": 1.0,
                    "lr": float(lr),
                    "elapsed_sec": float(time.time() - start_time),
                }
                history.append(record)
                print(json.dumps(record), flush=True)

    model.load_state_dict(best_state)
    pred_np, metrics, per_set = evaluate_model(
        model, features_t, y_all, err_all, use
    )
    pull_np = (pred_np - y_all.ravel()) / err_all.ravel()

    metrics.update(
        {
            "model_type": "direct_even_dnn_optimized",
            "n_sets": int(use["set"].nunique()),
            "n_unique_kinematics": int(len(base_unique)),
            "base_cols": base_cols,
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "activation": args.activation,
            "balance": args.balance,
            "length_lambda": float(args.length_lambda),
            "curvature_lambda": float(args.curvature_lambda),
            "phi_center": float(args.phi_center),
            "output_limit": float(args.output_limit),
            "max_symmetry_abs_diff": 0.0,
            "exact_duplicate_rows_removed": int(
                0 if args.keep_exact_duplicates else duplicate_count
            ),
            "experiment_year_scales": scale_rules,
            "feature_definition": "base kinematics plus u=(1-cos(phi-phi_center))/2; no Bernstein basis",
            "training_definition": "log-pull pretraining -> experimental chi2; balanced loss; low-LR length/curvature finish",
        }
    )

    out = use.copy()
    out["model_xsec"] = pred_np
    out["pull"] = pull_np
    out.to_csv(outdir / "predictions_with_pulls.csv", index=False)
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    with open(outdir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    torch.save(
        {
            "model_type": "direct_even_dnn_optimized",
            "model_state_dict": model.state_dict(),
            "base_cols": base_cols,
            "target_col": args.target_col,
            "error_col": args.error_col,
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": args.activation,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "log_y_mean": log_y_mean,
            "log_y_std": log_y_std,
            "phi_center": args.phi_center,
            "output_limit": args.output_limit,
            "length_lambda": args.length_lambda,
            "curvature_lambda": args.curvature_lambda,
            "balance": args.balance,
            "experiment_year_scales": scale_rules,
        },
        outdir / "xsec_surrogate.pt",
    )
    return model, out, metrics, per_set


def fig_to_html_img(fig: plt.Figure, dpi: int) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<img src="data:image/png;base64,'
        + encoded
        + '" style="max-width:950px;width:100%;">'
    )


def make_html_report(
    model: EvenDirectDNN,
    out: pd.DataFrame,
    metrics: Dict[str, float],
    per_set: pd.Series,
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
        "<title>DVCS direct even DNN surrogate</title>",
        f"<style>{style}</style></head><body>",
        "<h1>DVCS unpolarized cross-section surrogate</h1>",
        "<p>Direct DNN with inputs <code>k, Q2, xB, t, u</code>, where "
        "<code>u=(1-cos(phi-phi_center))/2</code>. There is no Bernstein or "
        "Fourier basis. The output column is <code>model_xsec</code>.</p>",
    ]

    if metrics.get("experiment_year_scales"):
        parts.append(
            "<p><strong>Applied experiment-year scale rules:</strong> "
            + json.dumps(metrics["experiment_year_scales"])
            + "</p>"
        )

    keys = [
        "chi2_per_point",
        "mean_set_chi2",
        "max_set_chi2",
        "pull_rms",
        "median_abs_pull",
        "frac_abs_pull_lt_1",
        "frac_abs_pull_lt_2",
        "max_abs_pull",
        "n_points",
        "n_sets",
        "exact_duplicate_rows_removed",
    ]
    parts.append("<h2>Global metrics</h2><table>")
    for key in keys:
        value = metrics.get(key, "")
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        parts.append(f"<tr><th>{key}</th><td>{text}</td></tr>")
    parts.append("</table>")

    # Global pull histogram.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(out["pull"].to_numpy(), bins=80)
    ax.set_xlabel("pull = (model_xsec - data) / total error")
    ax.set_ylabel("count")
    ax.set_title("Global pull distribution")
    fig.tight_layout()
    parts.append("<h2>Global diagnostics</h2>" + fig_to_html_img(fig, args.html_dpi))

    # Model versus data.
    y = out[args.target_col].to_numpy()
    pred = out["model_xsec"].to_numpy()
    err = out[args.error_col].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(y, pred, xerr=err, fmt=".", markersize=2, alpha=0.35)
    lo = min(float(y.min()), float(pred.min()))
    hi = max(float(y.max()), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("data cross section")
    ax.set_ylabel("model_xsec")
    ax.set_title(f"Predicted vs data, chi2/N={metrics['chi2_per_point']:.3f}")
    fig.tight_layout()
    parts.append(fig_to_html_img(fig, args.html_dpi))

    # Table and plot order: worst sets first, so failures are easy to inspect.
    set_rows = []
    for set_id, group in out.groupby("set", sort=True):
        pulls = group["pull"].to_numpy()
        set_rows.append(
            {
                "set": int(set_id),
                "n": int(len(group)),
                "chi2_per_point": float(np.mean(pulls ** 2)),
                "k": float(group["k"].iloc[0]) if "k" in group else np.nan,
                "q_squared": float(group["q_squared"].iloc[0]),
                "x_b": float(group["x_b"].iloc[0]),
                "t": float(group["t"].iloc[0]),
            }
        )
    set_table = pd.DataFrame(set_rows).sort_values(
        "chi2_per_point", ascending=False
    )

    parts.append("<h2>Set-level summary (worst first)</h2><table>")
    parts.append(
        "<tr><th>set</th><th>N</th><th>chi2/N</th><th>k</th>"
        "<th>Q2</th><th>xB</th><th>t</th></tr>"
    )
    for _, row in set_table.iterrows():
        parts.append(
            f"<tr><td><a href='#set-{int(row['set'])}'>{int(row['set'])}</a></td>"
            f"<td>{int(row['n'])}</td><td>{row['chi2_per_point']:.3f}</td>"
            f"<td>{row['k']:.5g}</td><td>{row['q_squared']:.5g}</td>"
            f"<td>{row['x_b']:.5g}</td><td>{row['t']:.5g}</td></tr>"
        )
    parts.append("</table><h2>Individual fixed-kinematics sets</h2>")

    for index, row in enumerate(set_table.itertuples(index=False), start=1):
        group = out[out["set"] == row.set].sort_values("phi")
        phi = group["phi"].to_numpy()
        y = group[args.target_col].to_numpy()
        err = group[args.error_col].to_numpy()
        pred = group["model_xsec"].to_numpy()
        pulls = group["pull"].to_numpy()
        base = group[model.base_cols].iloc[0].to_numpy(np.float32)

        if args.plot_full_phi:
            lo, hi = -math.pi, math.pi
        else:
            span = float(phi.max() - phi.min()) if phi.max() > phi.min() else 1.0
            pad = 0.02 * span
            lo, hi = float(phi.min() - pad), float(phi.max() + pad)
        grid = np.linspace(lo, hi, args.phi_grid_points, dtype=np.float32)
        line = model.predict_numpy(base, grid, args.phi_center)

        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.plot(grid, line, linewidth=2, label="direct hard-even DNN")
        ax.errorbar(
            phi,
            y,
            yerr=err,
            fmt="o",
            markersize=4,
            capsize=2,
            label="data +/- total error",
        )
        ax.plot(phi, pred, "x", markersize=4, label="model_xsec at data bins")
        ax.set_xlabel("phi [rad]")
        ax.set_ylabel("unpolarized cross section")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        ax.set_title(
            f"set {int(row.set)} | k={row.k:.4g}, Q2={row.q_squared:.4g}, "
            f"xB={row.x_b:.4g}, t={row.t:.4g}\n"
            f"N={len(group)}, chi2/N={np.mean(pulls**2):.3f}, "
            f"pull RMS={np.sqrt(np.mean(pulls**2)):.3f}, "
            f"median |pull|={np.median(np.abs(pulls)):.3f}, "
            f"max |pull|={np.max(np.abs(pulls)):.3g}",
            fontsize=10,
        )
        fig.tight_layout()
        parts.append(
            f"<div class='plot' id='set-{int(row.set)}'>"
            f"<h3>Set {int(row.set)}</h3>{fig_to_html_img(fig, args.html_dpi)}</div>"
        )
        if index % args.plot_print_every == 0:
            print(f"embedded {index} set plots", flush=True)

    parts.append("</body></html>")
    with open(outdir / "index.html", "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="xsec_surrogate")
    parser.add_argument("--target-col", default="unp_beam_unp_target_xsec")
    parser.add_argument("--error-col", default="unp_beam_unp_target_xsec_err")
    parser.add_argument("--include-k", action="store_true", default=True)
    parser.add_argument("--no-k", dest="include_k", action="store_false")
    parser.add_argument("--keep-exact-duplicates", action="store_true")
    parser.add_argument(
        "--experiment-year-scale",
        action="append",
        default=[],
        metavar="YEAR=FACTOR",
        help="Repeatable explicit convention conversion; default is no scaling.",
    )

    parser.add_argument("--phi-center", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--activation", choices=["tanh", "silu", "gelu"], default="silu")
    parser.add_argument("--output-limit", type=float, default=10.0)

    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-drop1-frac", type=float, default=0.55)
    parser.add_argument("--lr-mid-factor", type=float, default=0.30)
    parser.add_argument("--lr-drop2-frac", type=float, default=0.85)
    parser.add_argument("--lr-final-factor", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--sigma-loss-start-frac", type=float, default=0.05)
    parser.add_argument("--sigma-loss-full-frac", type=float, default=0.35)
    parser.add_argument("--min-log-error", type=float, default=0.05)
    parser.add_argument("--balance", choices=["point", "set", "experiment"], default="experiment")
    parser.add_argument("--worst-set-weight", type=float, default=0.0)
    parser.add_argument("--worst-set-fraction", type=float, default=0.10)

    # Stage-one regularization is zero by default: first obtain a complete fit.
    parser.add_argument("--stage1-length-lambda", type=float, default=0.0)
    parser.add_argument("--stage1-curvature-lambda", type=float, default=0.0)
    parser.add_argument("--reg-start-frac", type=float, default=0.35)
    parser.add_argument("--reg-full-frac", type=float, default=0.75)

    # A very low-LR finishing stage adds mild smoothness while preserving the fit.
    parser.add_argument("--finetune-epochs", type=int, default=600)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-point-fraction", type=float, default=0.20)
    parser.add_argument("--length-lambda", type=float, default=0.05)
    parser.add_argument("--curvature-lambda", type=float, default=0.001)

    parser.add_argument("--reg-grid-points", type=int, default=81)
    parser.add_argument("--reg-phi-min", type=float, default=-math.pi)
    parser.add_argument("--reg-phi-max", type=float, default=math.pi)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=300)
    parser.add_argument("--finetune-print-every", type=int, default=200)

    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--plot-full-phi", action="store_true")
    parser.add_argument("--phi-grid-points", type=int, default=400)
    parser.add_argument("--html-dpi", type=int, default=95)
    parser.add_argument("--plot-print-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, out, metrics, per_set = train(args)
    if not args.no_html:
        make_html_report(model, out, metrics, per_set, args)
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)
    print(f"Wrote {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
