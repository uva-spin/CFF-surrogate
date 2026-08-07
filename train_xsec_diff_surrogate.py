#!/usr/bin/env python3
"""One-command production training for the DVCS xsec-difference surrogate.

Run this script after the unpolarized cross-section surrogate has been trained.
It orchestrates the companion ``dvcs_xsec_diff_direct_dnn_optimized.py`` model
through two conceptually distinct steps:

1. Fit the measured beam-helicity cross-section difference with uncertainty-
   weighted experiment-balanced chi-square.
2. Apply increasingly strict smoothness passes.  The later passes penalize both
   the average BSA curvature and the worst small subset of phase-space curves.
   This is important because an average regularizer can hide a few locally
   wiggly curves among more than one thousand otherwise smooth curves.

The final observable representation is

    A_LU = tanh[sin(phi) * DNN(k,Q2,xB,t,u)],
    Delta_sigma_LU = sigma_UU_surrogate * A_LU,
    u = (1-cos(phi))/2.

Thus the difference is exactly odd in phi, the cross-section model and
xsec-difference model share the same numerical coordinates, and the DNN—not a
Fourier or Bernstein expansion—learns the phase-space dependence.

Example
-------
python train_xsec_diff_surrogate.py \
  --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
  --xsec-checkpoint xsec_surrogate/xsec_surrogate.pt \
  --outdir xsec_diff_surrogate

Only the final output directory is retained by default.  It contains one
self-contained ``index.html`` rather than hundreds of separate PNG files.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--xsec-checkpoint", default="xsec_surrogate/xsec_surrogate.pt")
    p.add_argument("--outdir", default="xsec_diff_surrogate")
    p.add_argument(
        "--model-script",
        default=str(here / "dvcs_xsec_diff_direct_dnn_optimized.py"),
    )
    p.add_argument(
        "--xsec-model-script",
        default=str(here / "dvcs_xsec_direct_dnn_optimized.py"),
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-stages", action="store_true")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Short diagnostic run; not intended for production uncertainty work.",
    )
    return p.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    model_script = Path(args.model_script).resolve()
    xsec_model_script = Path(args.xsec_model_script).resolve()
    csv_path = Path(args.csv).resolve()
    xsec_checkpoint = Path(args.xsec_checkpoint).resolve()
    outdir = Path(args.outdir).resolve()

    for path in [model_script, xsec_model_script, csv_path, xsec_checkpoint]:
        if not path.exists():
            raise FileNotFoundError(path)

    work = outdir.parent / f".{outdir.name}_stages"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if outdir.exists():
        shutil.rmtree(outdir)

    if args.quick:
        base_epochs, base_finish, polish_epochs = 300, 25, 20
        html_dpi = 70
    else:
        base_epochs, base_finish, polish_epochs = 1800, 100, 120
        html_dpi = 85

    common = [
        sys.executable,
        str(model_script),
        "--csv", str(csv_path),
        "--xsec-checkpoint", str(xsec_checkpoint),
        "--xsec-model-script", str(xsec_model_script),
        "--threads", str(args.threads),
        "--seed", str(args.seed),
        "--reg-grid-points", "61",
        "--no-html",
    ]

    # Initial data fit with a gentle first smoothness pass.
    previous = work / "stage0_fit"
    run(common + [
        "--outdir", str(previous),
        "--epochs", str(base_epochs),
        "--finetune-epochs", str(base_finish),
        "--length-lambda", "0.02",
        "--curvature-lambda", "0.0005",
    ])

    # Gradual continuation avoids forcing a very strong geometric prior onto an
    # untrained network.  Each stage begins from the preceding checkpoint.
    stages = [
        ("stage1_global_smooth", 0.10, 0.005, 0.00, 0.0),
        ("stage2_global_strong", 0.30, 0.020, 0.00, 0.0),
        ("stage3_worst_5pct",    0.10, 0.010, 0.05, 3.0),
        ("stage4_worst_10pct",   0.20, 0.020, 0.10, 5.0),
        ("final",                0.30, 0.040, 0.10, 10.0),
    ]

    for name, length_lam, curve_lam, worst_frac, worst_weight in stages:
        current = outdir if name == "final" else work / name
        run(common + [
            "--outdir", str(current),
            "--init-checkpoint", str(previous / "xsec_diff_surrogate.pt"),
            "--epochs", "0",
            "--finetune-epochs", str(polish_epochs),
            "--finetune-lr", "2e-5",
            "--length-lambda", str(length_lam),
            "--curvature-lambda", str(curve_lam),
            "--worst-curve-fraction", str(worst_frac),
            "--worst-curve-weight", str(worst_weight),
            "--keep-final-state",
        ])
        previous = current

    # Build the single requested all-set HTML after training is complete.
    run([
        sys.executable,
        str(model_script),
        "--csv", str(csv_path),
        "--xsec-checkpoint", str(xsec_checkpoint),
        "--xsec-model-script", str(xsec_model_script),
        "--outdir", str(outdir),
        "--report-only",
        "--max-html-sets", "0",
        "--html-dpi", str(html_dpi),
        "--threads", str(args.threads),
    ])

    if not args.keep_stages:
        shutil.rmtree(work)
    print(f"\nFinal surrogate written to {outdir}", flush=True)


if __name__ == "__main__":
    main()
