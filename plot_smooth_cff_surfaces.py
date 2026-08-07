#!/usr/bin/env python3
"""Plot smooth ReH and ImH surfaces from the trained simultaneous CFF DNN.

The CFF model is three-dimensional,

    (Q2, xB, t) -> (ReH, ImH),

so a conventional 3D surface plot can display only a two-dimensional slice at
one fixed value of the third kinematic coordinate.  This script therefore makes
three complementary slice families:

  1. (xB, -t) at fixed Q2,
  2. (Q2, -t) at fixed xB,
  3. (Q2, xB) at fixed -t.

For every slice it plots ReH and ImH side by side.  The surface is shown only in
a conservative local-support region defined from the selected common CFF
kinematics in the same standardized feature coordinates used by the DNN.  This
prevents the rectangular plotting grid from being mistaken for a fully measured
phase-space box.

The black points are nearby local two-CFF fits.  Their vertical bars are local
curvature/Hessian errors and are included only as diagnostics; they are not the
final experimental CFF uncertainty.  The final uncertainty surfaces require the
matched observable-replica -> CFF-replica propagation.

Outputs
-------
One self-contained HTML file with all plots; no per-plot PNG files are written.
A separate preview image can optionally be requested.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
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


def cff_features(q2: np.ndarray, xb: np.ndarray, t: np.ndarray) -> np.ndarray:
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
    """Architecture stored by dvcs_joint_cff_extraction.py."""

    def __init__(self, checkpoint: Dict) -> None:
        super().__init__()
        self.output_limit = float(checkpoint["output_limit"])
        self.register_buffer("x_mean", torch.tensor(checkpoint["feature_mean"], dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(checkpoint["feature_std"], dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(checkpoint["output_mean"], dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(checkpoint["output_std"], dtype=torch.float32))
        hidden = int(checkpoint["hidden"])
        depth = int(checkpoint["depth"])
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


def load_model(path: Path) -> Tuple[CFFNet, Dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CFFNet(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict(model: CFFNet, q2, xb, t, batch: int = 65536) -> np.ndarray:
    features = cff_features(q2, xb, t)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            tensor = torch.tensor(features[start:start + batch], dtype=torch.float32)
            out.append(model(tensor).cpu().numpy())
    return np.concatenate(out, axis=0)


def parse_values(text: str | None, data: pd.Series, quantiles=(0.2, 0.5, 0.8)) -> List[float]:
    if text:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    return [float(data.quantile(q)) for q in quantiles]


def figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=165, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def support_setup(cff_sets: pd.DataFrame, checkpoint: Dict, neighbors: int, quantile: float, scale: float):
    features = cff_features(
        cff_sets["q_squared"].to_numpy(float),
        cff_sets["x_b"].to_numpy(float),
        cff_sets["t"].to_numpy(float),
    )
    standardized = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    tree = cKDTree(standardized)
    k_ref = min(neighbors + 1, len(standardized))  # includes the point itself
    reference_distances = tree.query(standardized, k=k_ref)[0]
    if reference_distances.ndim == 1:
        reference_radius = reference_distances
    else:
        reference_radius = reference_distances[:, -1]
    threshold = float(np.quantile(reference_radius, quantile) * scale)
    return standardized, tree, threshold


def support_mask(q2, xb, t, checkpoint: Dict, tree: cKDTree, threshold: float, neighbors: int):
    features = cff_features(q2, xb, t)
    standardized = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    k = min(neighbors, tree.n)
    distances = tree.query(standardized, k=k)[0]
    kth = distances if np.ndim(distances) == 1 else distances[:, -1]
    return kth <= threshold, kth


def fixed_feature_z(kind: str, value: float, checkpoint: Dict) -> Tuple[int, float]:
    if kind == "q2":
        index, raw = 0, math.log(value)
    elif kind == "xb":
        index, raw = 1, math.log(value / (1.0 - value))
    elif kind == "minus_t":
        index, raw = 2, math.log(value)
    else:
        raise ValueError(kind)
    z = (raw - float(checkpoint["feature_mean"][0, index])) / float(checkpoint["feature_std"][0, index])
    return index, z


def nearby_local_points(
    local: pd.DataFrame,
    kind: str,
    fixed_value: float,
    checkpoint: Dict,
    slice_band: float,
) -> pd.DataFrame:
    features = cff_features(local["q_squared"], local["x_b"], local["t"])
    standardized = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    index, z0 = fixed_feature_z(kind, fixed_value, checkpoint)
    return local[np.abs(standardized[:, index] - z0) <= slice_band].copy()


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


def local_xy(kind: str, local: pd.DataFrame):
    if kind == "q2":
        return local["x_b"].to_numpy(), (-local["t"]).to_numpy()
    if kind == "xb":
        return local["q_squared"].to_numpy(), (-local["t"]).to_numpy()
    return local["q_squared"].to_numpy(), local["x_b"].to_numpy()


def plot_slice(
    kind: str,
    value: float,
    model: CFFNet,
    checkpoint: Dict,
    cff_sets: pd.DataFrame,
    local: pd.DataFrame,
    tree: cKDTree,
    threshold: float,
    args,
    norms: Tuple[Normalize, Normalize],
):
    A, B, q2, xb, t, xlabel, ylabel, title = make_slice_grid(kind, value, cff_sets, args.grid_size)
    pred = predict(model, q2, xb, t)
    mask, distance = support_mask(q2, xb, t, checkpoint, tree, threshold, args.support_neighbors)
    reh = pred[:, 0].reshape(A.shape)
    imh = pred[:, 1].reshape(A.shape)
    mask2 = mask.reshape(A.shape)
    reh = np.where(mask2, reh, np.nan)
    imh = np.where(mask2, imh, np.nan)

    points = nearby_local_points(local, kind, value, checkpoint, args.slice_band)
    px, py = local_xy(kind, points) if len(points) else (np.array([]), np.array([]))

    # A combined view is used deliberately.  The contour maps make the support
    # mask and smooth interpolation easy to read, while the 3D panels show the
    # actual surface geometry and the local-fit vertical uncertainty bars.
    fig = plt.figure(figsize=(13.6, 10.0))
    cax_re = fig.add_subplot(221)
    cax_im = fig.add_subplot(222)
    sax_re = fig.add_subplot(223, projection="3d")
    sax_im = fig.add_subplot(224, projection="3d")

    contour_re = cax_re.contourf(A, B, reh, levels=30, cmap="coolwarm", norm=norms[0])
    contour_im = cax_im.contourf(A, B, imh, levels=30, cmap="viridis", norm=norms[1])
    if len(points):
        cax_re.scatter(px, py, s=15, facecolors="none", edgecolors="black", linewidths=0.7)
        cax_im.scatter(px, py, s=15, facecolors="none", edgecolors="black", linewidths=0.7)
    for ax, name in [(cax_re, r"$\mathrm{Re}\,\mathcal{H}_{\mathrm{eff}}$"), (cax_im, r"$\mathrm{Im}\,\mathcal{H}_{\mathrm{eff}}$")]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(rf"{name} contour | {title}")
        ax.grid(alpha=0.15)
    fig.colorbar(contour_re, ax=cax_re, pad=0.02)
    fig.colorbar(contour_im, ax=cax_im, pad=0.02)

    surfaces = []
    for ax, Z, zcol, ecol, zlabel, cmap, norm in [
        (sax_re, reh, "ReH_local", "ReH_local_curvature_err", r"$\mathrm{Re}\,\mathcal{H}_{\mathrm{eff}}$", "coolwarm", norms[0]),
        (sax_im, imh, "ImH_local", "ImH_local_curvature_err", r"$\mathrm{Im}\,\mathcal{H}_{\mathrm{eff}}$", "viridis", norms[1]),
    ]:
        surf = ax.plot_surface(A, B, Z, cmap=cmap, norm=norm, linewidth=0, antialiased=True, alpha=0.93)
        surfaces.append(surf)
        if len(points):
            pz = points[zcol].to_numpy(float)
            pe = points[ecol].to_numpy(float)
            ax.scatter(px, py, pz, s=18, c="black", depthshade=False)
            for xx, yy, zz, ee in zip(px, py, pz, pe):
                if np.isfinite(ee):
                    ax.plot([xx, xx], [yy, yy], [zz - ee, zz + ee], color="black", alpha=0.42, linewidth=0.65)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.grid(True, alpha=0.2)
    sax_re.set_title(rf"$\mathrm{{Re}}\,\mathcal{{H}}_{{\mathrm{{eff}}}}$ surface | {title}")
    sax_im.set_title(rf"$\mathrm{{Im}}\,\mathcal{{H}}_{{\mathrm{{eff}}}}$ surface | {title}")

    fig.suptitle(
        f"Smooth central CFF DNN slice; conservative common-support mask | "
        f"near-slice local points: {len(points)}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, int(mask.sum()), len(points), float(np.nanmin(reh)), float(np.nanmax(reh)), float(np.nanmin(imh)), float(np.nanmax(imh))

def support_figure(cff_sets: pd.DataFrame):
    fig = plt.figure(figsize=(12.3, 5.5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    sc1 = ax1.scatter(cff_sets["x_b"], -cff_sets["t"], cff_sets["q_squared"], c=cff_sets["ReH"], s=26)
    sc2 = ax2.scatter(cff_sets["x_b"], -cff_sets["t"], cff_sets["q_squared"], c=cff_sets["ImH"], s=26)
    for ax, title in [(ax1, "Selected support colored by ReH"), (ax2, "Selected support colored by ImH")]:
        ax.set_xlabel(r"$x_B$")
        ax.set_ylabel(r"$-t\,[\mathrm{GeV}^2]$")
        ax.set_zlabel(r"$Q^2\,[\mathrm{GeV}^2]$")
        ax.set_title(title)
    fig.colorbar(sc1, ax=ax1, shrink=0.65, pad=0.08)
    fig.colorbar(sc2, ax=ax2, shrink=0.65, pad=0.08)
    fig.tight_layout()
    return fig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="joint_cff_extraction/cff_surrogate.pt")
    p.add_argument("--cff-sets", default="joint_cff_extraction/cff_sets.csv")
    p.add_argument("--local-cff", default="joint_cff_extraction/local_h2_extraction.csv")
    p.add_argument("--out", default="cff_smooth_surfaces.html")
    p.add_argument("--preview", default="", help="Optional path for one representative PNG preview")
    p.add_argument("--q2-slices", default="")
    p.add_argument("--xb-slices", default="")
    p.add_argument("--minus-t-slices", default="")
    p.add_argument("--grid-size", type=int, default=78)
    p.add_argument("--support-neighbors", type=int, default=5)
    p.add_argument("--support-quantile", type=float, default=0.80)
    p.add_argument("--support-scale", type=float, default=1.0)
    p.add_argument("--slice-band", type=float, default=0.25, help="Near-slice point band in standardized DNN feature units")
    p.add_argument("--elev", type=float, default=28.0)
    p.add_argument("--azim", type=float, default=-58.0)
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    cff_sets_path = Path(args.cff_sets)
    local_path = Path(args.local_cff)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(checkpoint_path)
    cff_sets = pd.read_csv(cff_sets_path)
    local = pd.read_csv(local_path)
    local = local[local["extraction_selected"].astype(bool)].copy()
    selected = set(int(x) for x in checkpoint.get("selected_sets", cff_sets["set"].tolist()))
    cff_sets = cff_sets[cff_sets["set"].isin(selected)].copy()
    local = local[local["set"].isin(selected)].copy()

    _, tree, threshold = support_setup(
        cff_sets, checkpoint, args.support_neighbors, args.support_quantile, args.support_scale
    )

    q2_slices = parse_values(args.q2_slices or None, cff_sets["q_squared"])
    xb_slices = parse_values(args.xb_slices or None, cff_sets["x_b"])
    mt_slices = parse_values(args.minus_t_slices or None, -cff_sets["t"])

    # Use robust common color ranges so the same CFF has a consistent scale in
    # every slice.  The z axis itself remains free and is not numerically clipped.
    re_lo, re_hi = np.quantile(cff_sets["ReH"], [0.02, 0.98])
    im_lo, im_hi = np.quantile(cff_sets["ImH"], [0.02, 0.98])
    re_span = max(abs(re_lo), abs(re_hi), 1e-6)
    norms = (
        TwoSlopeNorm(vmin=-re_span, vcenter=0.0, vmax=re_span),
        Normalize(vmin=float(im_lo), vmax=float(im_hi)),
    )

    images: List[Tuple[str, str]] = []
    records: List[Dict] = []
    images.append(("Selected strict common CFF support", figure_to_base64(support_figure(cff_sets))))

    preview_written = False
    for kind, values, heading in [
        ("q2", q2_slices, r"$(x_B,-t)$ surfaces at fixed $Q^2$"),
        ("xb", xb_slices, r"$(Q^2,-t)$ surfaces at fixed $x_B$"),
        ("minus_t", mt_slices, r"$(Q^2,x_B)$ surfaces at fixed $-t$"),
    ]:
        for value in values:
            fig, nmask, npts, rmin, rmax, imin, imax = plot_slice(
                kind, value, model, checkpoint, cff_sets, local, tree, threshold, args, norms
            )
            if args.preview and not preview_written and kind == "q2" and value == q2_slices[len(q2_slices)//2]:
                fig.savefig(args.preview, dpi=180, bbox_inches="tight")
                preview_written = True
            encoded = figure_to_base64(fig)
            label = {"q2": "Q2", "xb": "xB", "minus_t": "-t"}[kind]
            images.append((f"{heading}: {label}={value:.5g}", encoded))
            records.append({
                "slice_kind": kind,
                "slice_value": value,
                "supported_grid_points": nmask,
                "nearby_local_points": npts,
                "ReH_min": rmin,
                "ReH_max": rmax,
                "ImH_min": imin,
                "ImH_max": imax,
            })

    config = {
        "checkpoint": str(checkpoint_path),
        "cff_sets": str(cff_sets_path),
        "local_cff": str(local_path),
        "selected_sets": int(len(cff_sets)),
        "support_neighbors": args.support_neighbors,
        "support_quantile": args.support_quantile,
        "support_scale": args.support_scale,
        "support_threshold_standardized_feature_space": threshold,
        "slice_band_standardized_feature_units": args.slice_band,
        "q2_slices": q2_slices,
        "xb_slices": xb_slices,
        "minus_t_slices": mt_slices,
        "uncertainty_status": "local curvature bars and central-seed surface only; matched observable replicas still required for experimental CFF surfaces",
    }

    rows_html = pd.DataFrame(records).to_html(index=False, float_format=lambda x: f"{x:.4g}")
    html = [
        "<html><head><meta charset='utf-8'><title>Smooth DVCS CFF surfaces</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1250px;margin:2em auto;line-height:1.5} img{max-width:100%;height:auto} table{border-collapse:collapse} th,td{border:1px solid #bbb;padding:5px 8px} code{background:#eee;padding:2px 4px}</style>",
        "</head><body>",
        "<h1>Smooth effective ReH and ImH surfaces</h1>",
        "<p>The central simultaneous BKM DNN is evaluated continuously in <code>(Q2,xB,t)</code>. Because the CFF is a function of three kinematic variables, the report uses complementary two-dimensional slices at fixed values of the third variable.</p>",
        f"<p><b>Common-support mask:</b> a point is displayed only when its {args.support_neighbors}-neighbor distance in the standardized DNN feature space lies below the {100*args.support_quantile:.0f}th-percentile support radius ({threshold:.4g}). This is deliberately more restrictive than plotting the full rectangular min/max box.</p>",
        "<p><b>Points and bars:</b> black markers are nearby local H-dominance fits. Their vertical bars are local curvature/Hessian diagnostics, not final experimental uncertainties. Experimental CFF error surfaces require matched cross-section/difference surrogate replicas propagated through the simultaneous BKM fit.</p>",
        "<p><b>Interpretation:</b> both surfaces are effective H-dominance CFFs; the remaining CFF components were fixed to zero because the larger local parameterizations were ill-conditioned with only the two present observables.</p>",
        "<h2>Plot configuration</h2>",
        f"<pre>{json.dumps(config, indent=2)}</pre>",
        "<h2>Slice diagnostics</h2>",
        rows_html,
    ]
    for title, image in images:
        html.extend([f"<h2>{title}</h2>", f"<img src='data:image/png;base64,{image}'>"])
    html.extend(["</body></html>"])
    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote {out_path}")
    if args.preview:
        print(f"Wrote {args.preview}")


if __name__ == "__main__":
    main()
