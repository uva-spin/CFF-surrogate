#!/usr/bin/env python3
"""Plot CFF mean surfaces with experimental uncertainty encoded by color.

The vertical coordinate is the replica-ensemble mean CFF value.  Surface color
is the pointwise experimental 68% half-width,

    half_width = (q84 - q16) / 2,

so geometry and color carry different information:

  * surface height: ReH_eff or ImH_eff central/mean value;
  * surface color: local experimental uncertainty magnitude.

The script reads the compact tables produced by
``dvcs_cff_experimental_replica_surfaces.py`` and writes one self-contained HTML
report.  It creates no per-plot PNG files unless ``--preview`` is requested.
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def parse_csv_values(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def nearest_available(requested: Iterable[float], available: np.ndarray) -> List[float]:
    available = np.asarray(sorted(np.unique(available)), dtype=float)
    if len(available) == 0:
        return []
    requested = list(requested)
    if not requested:
        return available.tolist()
    selected: List[float] = []
    for value in requested:
        nearest = float(available[np.argmin(np.abs(available - value))])
        if nearest not in selected:
            selected.append(nearest)
    return selected


def figure_to_base64(fig, dpi: int) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pivot_grid(frame: pd.DataFrame, xcol: str, ycol: str, zcol: str):
    xvals = np.sort(frame[xcol].unique())
    yvals = np.sort(frame[ycol].unique())
    pivot = frame.pivot(index=ycol, columns=xcol, values=zcol)
    array = pivot.reindex(index=yvals, columns=xvals).to_numpy(dtype=float)
    X, Y = np.meshgrid(xvals, yvals)
    return X, Y, array


def slice_coordinates(kind: str, frame: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str, str]:
    out = frame.copy()
    if kind == "q2":
        out["plot_x"] = out["x_b"]
        out["plot_y"] = -out["t"]
        return out, r"$x_B$", r"$-t\,[\mathrm{GeV}^2]$", "x_b", "minus_t"
    if kind == "xb":
        out["plot_x"] = out["q_squared"]
        out["plot_y"] = -out["t"]
        return out, r"$Q^2\,[\mathrm{GeV}^2]$", r"$-t\,[\mathrm{GeV}^2]$", "q_squared", "minus_t"
    if kind == "minus_t":
        out["plot_x"] = out["q_squared"]
        out["plot_y"] = out["x_b"]
        return out, r"$Q^2\,[\mathrm{GeV}^2]$", r"$x_B$", "q_squared", "x_b"
    raise ValueError(f"Unknown slice kind: {kind}")


def nearby_points(points: pd.DataFrame, kind: str, value: float, q2_band: float, xb_band: float, t_band: float):
    if kind == "q2":
        p = points[np.abs(points["q_squared"] - value) <= q2_band].copy()
        p["plot_x"] = p["x_b"]
        p["plot_y"] = -p["t"]
        fixed = rf"$Q^2={value:.3g}\,\mathrm{{GeV}}^2$"
    elif kind == "xb":
        p = points[np.abs(points["x_b"] - value) <= xb_band].copy()
        p["plot_x"] = p["q_squared"]
        p["plot_y"] = -p["t"]
        fixed = rf"$x_B={value:.3g}$"
    else:
        p = points[np.abs((-points["t"]) - value) <= t_band].copy()
        p["plot_x"] = p["q_squared"]
        p["plot_y"] = p["x_b"]
        fixed = rf"$-t={value:.3g}\,\mathrm{{GeV}}^2$"
    return p, fixed


def make_figure(
    surface: pd.DataFrame,
    points: pd.DataFrame,
    kind: str,
    value: float,
    component: str,
    cmap: str,
    norm: colors.Normalize,
    elev: float,
    azim: float,
    q2_band: float,
    xb_band: float,
    t_band: float,
):
    d = surface[(surface["slice_kind"] == kind) & np.isclose(surface["slice_value"], value)].copy()
    d = d[d["inside_support"].astype(bool)].copy()
    d, xlabel, ylabel, _, _ = slice_coordinates(kind, d)

    mean_col = f"{component}_mean"
    width_col = f"{component}_half_68"
    X, Y, Z = pivot_grid(d, "plot_x", "plot_y", mean_col)
    _, _, W = pivot_grid(d, "plot_x", "plot_y", width_col)

    invalid = ~np.isfinite(Z) | ~np.isfinite(W)
    Zm = np.ma.array(Z, mask=invalid)
    facecolors = matplotlib.colormaps[cmap](norm(W))
    facecolors[invalid, 3] = 0.0

    p, fixed_label = nearby_points(points, kind, value, q2_band, xb_band, t_band)

    fig = plt.figure(figsize=(9.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        Zm,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.96,
    )
    if len(p):
        ax.scatter(
            p["plot_x"],
            p["plot_y"],
            p[f"{component}_mean"],
            s=24,
            c="black",
            depthshade=False,
            label="CFF extraction points",
        )

    math_name = r"\mathrm{Re}\,\mathcal{H}_{\mathrm{eff}}" if component == "ReH" else r"\mathrm{Im}\,\mathcal{H}_{\mathrm{eff}}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(rf"${math_name}$")
    ax.set_title(
        rf"Mean ${math_name}$ surface; color = experimental 68% half-width\n{fixed_label}",
        pad=15,
    )
    ax.view_init(elev=elev, azim=azim)

    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, shrink=0.70, pad=0.09)
    cbar.set_label(rf"${math_name}$ experimental 68% half-width")
    if len(p):
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--surface-csv",
        default="cff_experimental_replica_surfaces/cff_surface_experimental_bands.csv",
    )
    p.add_argument(
        "--point-csv",
        default="cff_experimental_replica_surfaces/cff_set_experimental_bands.csv",
    )
    p.add_argument("--out", default="cff_uncertainty_colored_surfaces.html")
    p.add_argument("--preview", default="", help="Optional representative PNG path.")
    p.add_argument("--components", default="ReH,ImH")
    p.add_argument("--q2-slices", default="", help="Comma-separated; blank uses all stored q2 slices.")
    p.add_argument("--xb-slices", default="", help="Comma-separated; blank uses all stored xB slices.")
    p.add_argument("--minus-t-slices", default="", help="Comma-separated; blank uses all stored -t slices.")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--global-color-scale", action="store_true", default=True)
    p.add_argument("--per-slice-color-scale", dest="global_color_scale", action="store_false")
    p.add_argument("--q2-point-band", type=float, default=0.10)
    p.add_argument("--xb-point-band", type=float, default=0.015)
    p.add_argument("--minus-t-point-band", type=float, default=0.03)
    p.add_argument("--elev", type=float, default=28.0)
    p.add_argument("--azim", type=float, default=-58.0)
    p.add_argument("--dpi", type=int, default=165)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    surface = pd.read_csv(args.surface_csv)
    points = pd.read_csv(args.point_csv)
    components = [item.strip() for item in args.components.split(",") if item.strip()]
    for component in components:
        if component not in {"ReH", "ImH"}:
            raise ValueError("--components entries must be ReH and/or ImH")

    available = {
        kind: np.sort(surface.loc[surface["slice_kind"] == kind, "slice_value"].unique())
        for kind in ("q2", "xb", "minus_t")
    }
    chosen = {
        "q2": nearest_available(parse_csv_values(args.q2_slices), available["q2"]),
        "xb": nearest_available(parse_csv_values(args.xb_slices), available["xb"]),
        "minus_t": nearest_available(parse_csv_values(args.minus_t_slices), available["minus_t"]),
    }

    global_norms = {}
    for component in components:
        values = surface.loc[surface["inside_support"].astype(bool), f"{component}_half_68"].to_numpy(float)
        values = values[np.isfinite(values)]
        global_norms[component] = colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))

    html = [
        "<html><head><meta charset='utf-8'><title>CFF uncertainty-colored surfaces</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1450px;margin:auto;padding:24px;line-height:1.45}img{max-width:100%;height:auto}code{background:#eee;padding:2px 4px}</style></head><body>",
        "<h1>CFF mean surfaces with experimental uncertainty encoded by color</h1>",
        "<p>The vertical coordinate is the replica-ensemble mean CFF. The color is the pointwise experimental 68% half-width, <code>(q84-q16)/2</code>. Black points are nearby extracted CFF values. Only rows marked <code>inside_support</code> are displayed.</p>",
    ]

    preview_written = False
    for kind in ("q2", "xb", "minus_t"):
        for value in chosen[kind]:
            for component in components:
                if args.global_color_scale:
                    norm = global_norms[component]
                else:
                    subset = surface[
                        (surface["slice_kind"] == kind)
                        & np.isclose(surface["slice_value"], value)
                        & surface["inside_support"].astype(bool)
                    ]
                    vals = subset[f"{component}_half_68"].to_numpy(float)
                    vals = vals[np.isfinite(vals)]
                    norm = colors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
                fig = make_figure(
                    surface,
                    points,
                    kind,
                    value,
                    component,
                    args.cmap,
                    norm,
                    args.elev,
                    args.azim,
                    args.q2_point_band,
                    args.xb_point_band,
                    args.minus_t_point_band,
                )
                if args.preview and not preview_written:
                    fig.savefig(args.preview, dpi=args.dpi, bbox_inches="tight")
                    preview_written = True
                encoded = figure_to_base64(fig, args.dpi)
                html.append(f"<h2>{component}: {kind} slice at {value:.6g}</h2>")
                html.append(f"<img src='data:image/png;base64,{encoded}'>")

    html.append("</body></html>")
    Path(args.out).write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.preview:
        print(f"Wrote {args.preview}")


if __name__ == "__main__":
    main()
