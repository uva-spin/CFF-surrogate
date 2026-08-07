#!/usr/bin/env python3
"""Prepare a common DVCS cross-section / beam-helicity-difference data table.

The input file contains both published central observables

  * ``exp d4sig (nb/Gev^4)``      : helicity-averaged unpolarized cross section
  * ``exp del4sig (nb/GeV^4)``    : beam-helicity cross-section difference

plus statistical and asymmetric systematic errors.  It also contains ``dsig``
and ``delsig`` columns that behave like one sampled pseudo-data realization.
Those sampled columns are deliberately *not* used as the central data here.

The output is restricted to rows for which both observables are usable, because
the intended later BKM fit will use the two surrogate surfaces together.

Cleaning choices
----------------
1. Keep only finite physical kinematics: k>0, Q2>0, 0<xB<1, t<0.
2. Require a positive cross section and a positive total cross-section error.
   (The positive cross-section surrogate predicts log(sigma).)
3. Permit a zero cross-section difference when its uncertainty is positive;
   zero is a valid measured fluctuation for an odd observable.
4. Require a positive cross-section-difference error.
5. Symmetrize asymmetric systematics conservatively with

       syst = max(abs(syst_plus), abs(syst_minus))
       total = sqrt(stat**2 + syst**2).

   This reproduces the supplied ``dsig_err`` / ``delsig_err`` values to the
   precision in the file for ordinary rows.
6. Exclude clearly malformed uncertainty entries using an auditable ratio test:

       syst / max(abs(central), stat, tiny) > max_systematic_ratio.

   The default threshold 20 removes the obvious decimal/format failures while
   retaining ordinary large-uncertainty points.  It is configurable.
7. Remove exact duplicate published rows, keeping the first occurrence.
8. Convert phi from degrees on [0,360) to centered Trento-style radians on
   [-pi,pi):

       phi_rad = wrap(phi_deg) about 0 degrees.

   Thus 15 deg -> +0.262 rad and 345 deg -> -0.262 rad.  This is the convention
   needed for sigma(phi)=sigma(-phi) and Delta-sigma(phi)=-Delta-sigma(-phi).

Outputs
-------
``dvcs_xsec_diff_common_clean.csv``
    Canonically named common-support table used by both surrogate scripts.
``dvcs_xsec_diff_cleaning_audit.csv``
    Every original row, all flags, and the exact exclusion reason.
``dvcs_xsec_diff_cleaning_summary.json``
    Counts and the cleaning configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ORIGINAL_COLUMNS = {
    "xsec": "exp d4sig (nb/Gev^4)",
    "xsec_stat": "d4sig_stat",
    "xsec_sys_plus": "d4sig_sys+",
    "xsec_sys_minus": "d4sig_sys-",
    "xsec_err_provided": "dsig_err",
    "xsec_diff": "exp del4sig (nb/GeV^4)",
    "xsec_diff_stat": "del4sig_stat",
    "xsec_diff_sys_plus": "del4sig_sys+",
    "xsec_diff_sys_minus": "del4sig_sys-",
    "xsec_diff_err_provided": "delsig_err",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Raw combined DVCS CSV.")
    parser.add_argument(
        "--outdir",
        default="dvcs_xsec_diff_prepared",
        help="Directory for clean table, audit table, and summary.",
    )
    parser.add_argument(
        "--max-systematic-ratio",
        type=float,
        default=20.0,
        help=(
            "Flag a row when max(|sys+|,|sys-|) divided by "
            "max(|central|,stat,tiny) exceeds this value."
        ),
    )
    parser.add_argument("--tiny", type=float, default=1e-12)
    parser.add_argument(
        "--max-local-xsec-pull", type=float, default=10.0,
        help="Extreme within-set smooth-even diagnostic threshold."
    )
    parser.add_argument(
        "--max-local-xsec-ratio", type=float, default=3.0,
        help="Also require an extreme point/prediction ratio before excluding it."
    )
    parser.add_argument(
        "--max-local-diff-pull", type=float, default=8.0,
        help="Extreme within-set smooth-odd diagnostic threshold."
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args()


def joined_reasons(frame: pd.DataFrame, flag_columns: Iterable[str]) -> pd.Series:
    """Return a semicolon-separated reason string for every row."""
    names = list(flag_columns)
    values = frame[names].to_numpy(bool)
    out = []
    for flags in values:
        out.append(";".join(name.removeprefix("flag_") for name, state in zip(names, flags) if state))
    return pd.Series(out, index=frame.index, dtype="object")


def robust_weighted_fit(
    design: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    huber_c: float = 1.5,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Small IRLS fit used only to identify catastrophic transcription spikes.

    This is not the production surrogate.  It is a conservative within-set
    quality-control fit using a low-order smooth basis and Huber downweighting.
    """
    base_weight = 1.0 / np.maximum(errors, 1e-12) ** 2
    robust_weight = np.ones(len(values), dtype=float)
    beta = np.zeros(design.shape[1], dtype=float)
    ridge = 1e-10 * np.eye(design.shape[1])
    for _ in range(iterations):
        weight = base_weight * robust_weight
        lhs = design.T @ (weight[:, None] * design) + ridge
        rhs = design.T @ (weight * values)
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        pull = (values - design @ beta) / np.maximum(errors, 1e-12)
        updated = np.ones_like(pull)
        mask = np.abs(pull) > huber_c
        updated[mask] = huber_c / np.abs(pull[mask])
        if np.max(np.abs(updated - robust_weight)) < 1e-5:
            robust_weight = updated
            break
        robust_weight = updated
    prediction = design @ beta
    pull = (values - prediction) / np.maximum(errors, 1e-12)
    return prediction, pull


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv)
    raw.insert(0, "source_row", np.arange(len(raw), dtype=np.int64))

    required = [
        "set", "bin", "k", "Q2", "xB", "t", "phi", "experiment", "link",
        *ORIGINAL_COLUMNS.values(),
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce every numerical field used by the cleaning logic.  The raw file has
    # one missing systematic value; it is treated as zero below because the
    # supplied total error for that row equals its statistical error.
    numeric = [
        "set", "bin", "k", "Q2", "xB", "t", "phi",
        *ORIGINAL_COLUMNS.values(),
        "ReH", "ReHt", "ReE", "ReEt", "ImH", "ImHt", "ImE", "ImEt", "DVCS",
    ]
    for column in numeric:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    xsec = raw[ORIGINAL_COLUMNS["xsec"]]
    xsec_stat = raw[ORIGINAL_COLUMNS["xsec_stat"]]
    xsec_sys_plus_raw = raw[ORIGINAL_COLUMNS["xsec_sys_plus"]]
    xsec_sys_minus_raw = raw[ORIGINAL_COLUMNS["xsec_sys_minus"]]

    xsec_diff = raw[ORIGINAL_COLUMNS["xsec_diff"]]
    xsec_diff_stat = raw[ORIGINAL_COLUMNS["xsec_diff_stat"]]
    diff_sys_plus_raw = raw[ORIGINAL_COLUMNS["xsec_diff_sys_plus"]]
    diff_sys_minus_raw = raw[ORIGINAL_COLUMNS["xsec_diff_sys_minus"]]

    raw["flag_xsec_systematic_missing"] = xsec_sys_plus_raw.isna() | xsec_sys_minus_raw.isna()
    raw["flag_xsec_diff_systematic_missing"] = diff_sys_plus_raw.isna() | diff_sys_minus_raw.isna()

    # Systematic columns occasionally carry the sign of the observable.  Errors
    # are magnitudes, so use absolute values.  A missing systematic is set to 0,
    # while the corresponding flag remains in the audit table.
    xsec_sys_plus = xsec_sys_plus_raw.fillna(0.0).abs()
    xsec_sys_minus = xsec_sys_minus_raw.fillna(0.0).abs()
    diff_sys_plus = diff_sys_plus_raw.fillna(0.0).abs()
    diff_sys_minus = diff_sys_minus_raw.fillna(0.0).abs()

    xsec_syst = np.maximum(xsec_sys_plus, xsec_sys_minus)
    diff_syst = np.maximum(diff_sys_plus, diff_sys_minus)
    xsec_err = np.sqrt(xsec_stat**2 + xsec_syst**2)
    diff_err = np.sqrt(xsec_diff_stat**2 + diff_syst**2)

    xsec_ratio_denom = np.maximum.reduce(
        [xsec.abs().to_numpy(), xsec_stat.abs().to_numpy(), np.full(len(raw), args.tiny)]
    )
    diff_ratio_denom = np.maximum.reduce(
        [xsec_diff.abs().to_numpy(), xsec_diff_stat.abs().to_numpy(), np.full(len(raw), args.tiny)]
    )
    xsec_syst_ratio = xsec_syst.to_numpy() / xsec_ratio_denom
    diff_syst_ratio = diff_syst.to_numpy() / diff_ratio_denom

    raw["xsec_total_err_recomputed"] = xsec_err
    raw["xsec_diff_total_err_recomputed"] = diff_err
    raw["xsec_systematic_ratio"] = xsec_syst_ratio
    raw["xsec_diff_systematic_ratio"] = diff_syst_ratio

    finite_kinematics = np.isfinite(raw[["k", "Q2", "xB", "t", "phi"]]).all(axis=1)
    physical_kinematics = (
        finite_kinematics
        & (raw["k"] > 0)
        & (raw["Q2"] > 0)
        & (raw["xB"] > 0)
        & (raw["xB"] < 1)
        & (raw["t"] < 0)
    )

    raw["flag_bad_kinematics"] = ~physical_kinematics
    raw["flag_xsec_nonpositive_or_missing"] = ~(np.isfinite(xsec) & (xsec > 0))
    raw["flag_xsec_error_nonpositive_or_missing"] = ~(np.isfinite(xsec_err) & (xsec_err > 0))
    raw["flag_xsec_systematic_anomaly"] = ~(np.isfinite(xsec_syst_ratio)) | (
        xsec_syst_ratio > args.max_systematic_ratio
    )
    raw["flag_xsec_diff_missing"] = ~np.isfinite(xsec_diff)
    raw["flag_xsec_diff_error_nonpositive_or_missing"] = ~(
        np.isfinite(diff_err) & (diff_err > 0)
    )
    raw["flag_xsec_diff_systematic_anomaly"] = ~(np.isfinite(diff_syst_ratio)) | (
        diff_syst_ratio > args.max_systematic_ratio
    )

    # Duplicate definition uses the published central values and errors, not the
    # sampled dsig/delsig columns, which differ randomly even for duplicated rows.
    duplicate_subset = [
        "k", "Q2", "xB", "t", "phi", "experiment",
        ORIGINAL_COLUMNS["xsec"], ORIGINAL_COLUMNS["xsec_stat"],
        ORIGINAL_COLUMNS["xsec_sys_plus"], ORIGINAL_COLUMNS["xsec_sys_minus"],
        ORIGINAL_COLUMNS["xsec_diff"], ORIGINAL_COLUMNS["xsec_diff_stat"],
        ORIGINAL_COLUMNS["xsec_diff_sys_plus"], ORIGINAL_COLUMNS["xsec_diff_sys_minus"],
    ]
    raw["flag_exact_duplicate"] = raw.duplicated(subset=duplicate_subset, keep="first")

    # Correct angular convention for the symmetry constraints.  Do NOT subtract
    # 180 degrees: the reflection partner of +15 deg is 345 deg, not 165 deg.
    phi_deg_wrapped = ((raw["phi"] + 180.0) % 360.0) - 180.0
    raw["phi_centered_rad"] = np.deg2rad(phi_deg_wrapped)

    # Conservative second-stage QC for isolated transcription/decimal spikes.
    # A robust low-order even/odd fit is made *within each set*.  A point is
    # removed only at a very large pull threshold; ordinary noisy deviations are
    # left to the experimental uncertainty and production surrogate.
    preliminary_flags = [
        "flag_bad_kinematics",
        "flag_xsec_nonpositive_or_missing",
        "flag_xsec_error_nonpositive_or_missing",
        "flag_xsec_systematic_anomaly",
        "flag_xsec_diff_missing",
        "flag_xsec_diff_error_nonpositive_or_missing",
        "flag_xsec_diff_systematic_anomaly",
        "flag_exact_duplicate",
    ]
    preliminary_valid = ~raw[preliminary_flags].any(axis=1)
    raw["local_xsec_prediction"] = np.nan
    raw["local_xsec_pull"] = np.nan
    raw["local_xsec_diff_prediction"] = np.nan
    raw["local_xsec_diff_pull"] = np.nan

    for _, group in raw.loc[preliminary_valid].groupby("set"):
        index = group.index
        phi_local = group["phi_centered_rad"].to_numpy(float)
        u_local = (1.0 - np.cos(phi_local)) / 2.0
        sin_local = np.sin(phi_local)
        n_unique_u = len(np.unique(np.round(u_local, 6)))
        degree = min(4, max(1, n_unique_u - 2))

        x_design = np.column_stack([u_local ** power for power in range(degree + 1)])
        x_prediction, x_pull_local = robust_weighted_fit(
            x_design,
            group[ORIGINAL_COLUMNS["xsec"]].to_numpy(float),
            group["xsec_total_err_recomputed"].to_numpy(float),
        )
        d_design = np.column_stack(
            [sin_local * (u_local ** power) for power in range(degree + 1)]
        )
        d_prediction, d_pull_local = robust_weighted_fit(
            d_design,
            group[ORIGINAL_COLUMNS["xsec_diff"]].to_numpy(float),
            group["xsec_diff_total_err_recomputed"].to_numpy(float),
        )
        raw.loc[index, "local_xsec_prediction"] = x_prediction
        raw.loc[index, "local_xsec_pull"] = x_pull_local
        raw.loc[index, "local_xsec_diff_prediction"] = d_prediction
        raw.loc[index, "local_xsec_diff_pull"] = d_pull_local

    safe_x_prediction = np.maximum(raw["local_xsec_prediction"].to_numpy(float), args.tiny)
    x_value = np.maximum(raw[ORIGINAL_COLUMNS["xsec"]].to_numpy(float), args.tiny)
    x_multiplicative_ratio = np.maximum(x_value / safe_x_prediction, safe_x_prediction / x_value)
    raw["local_xsec_multiplicative_ratio"] = x_multiplicative_ratio
    raw["flag_local_xsec_catastrophic_outlier"] = (
        raw["local_xsec_pull"].abs() > args.max_local_xsec_pull
    ) & (raw["local_xsec_multiplicative_ratio"] > args.max_local_xsec_ratio)
    raw["flag_local_xsec_diff_catastrophic_outlier"] = (
        raw["local_xsec_diff_pull"].abs() > args.max_local_diff_pull
    )

    exclusion_flags = preliminary_flags + [
        "flag_local_xsec_catastrophic_outlier",
        "flag_local_xsec_diff_catastrophic_outlier",
    ]
    raw["exclude_reason"] = joined_reasons(raw, exclusion_flags)
    raw["common_valid"] = ~raw[exclusion_flags].any(axis=1)

    valid = raw[raw["common_valid"]].copy()

    clean = pd.DataFrame(
        {
            "source_row": valid["source_row"].astype(np.int64),
            "set": valid["set"].astype(np.int64),
            "bin": valid["bin"].astype(np.int64),
            "k": valid["k"],
            "q_squared": valid["Q2"],
            "x_b": valid["xB"],
            "t": valid["t"],
            "phi_deg": valid["phi"],
            "phi": valid["phi_centered_rad"],
            "unp_beam_unp_target_xsec": valid[ORIGINAL_COLUMNS["xsec"]],
            "unp_beam_unp_target_xsec_err": valid["xsec_total_err_recomputed"],
            "unp_beam_unp_target_xsec_errstat": valid[ORIGINAL_COLUMNS["xsec_stat"]],
            "unp_beam_unp_target_xsec_errsyst_plus": xsec_sys_plus.loc[valid.index],
            "unp_beam_unp_target_xsec_errsyst_minus": xsec_sys_minus.loc[valid.index],
            "xsec_diff": valid[ORIGINAL_COLUMNS["xsec_diff"]],
            "xsec_diff_err": valid["xsec_diff_total_err_recomputed"],
            "xsec_diff_errstat": valid[ORIGINAL_COLUMNS["xsec_diff_stat"]],
            "xsec_diff_errsyst_plus": diff_sys_plus.loc[valid.index],
            "xsec_diff_errsyst_minus": diff_sys_minus.loc[valid.index],
            # A useful diagnostic only.  The production BSA uncertainty should
            # be obtained from common replicas, not this uncorrelated formula.
            "bsa_from_central_values": (
                valid[ORIGINAL_COLUMNS["xsec_diff"]]
                / valid[ORIGINAL_COLUMNS["xsec"]]
            ),
            "experiment_year": valid["experiment"].astype(str),
            "experiment": valid["experiment"].astype(str),
            "link": valid["link"].astype(str),
        }
    )

    # Retain the supplied CFF/model columns for traceability, but neither
    # surrogate training script uses them as inputs or targets.
    for column in ["ReH", "ReHt", "ReE", "ReEt", "ImH", "ImHt", "ImE", "ImEt", "DVCS"]:
        if column in valid.columns:
            clean[column] = valid[column].to_numpy()

    clean = clean.sort_values(["set", "bin", "source_row"]).reset_index(drop=True)

    clean_path = outdir / "dvcs_xsec_diff_common_clean.csv"
    audit_path = outdir / "dvcs_xsec_diff_cleaning_audit.csv"
    excluded_path = outdir / "dvcs_xsec_diff_excluded_rows.csv"
    summary_path = outdir / "dvcs_xsec_diff_cleaning_summary.json"
    clean.to_csv(clean_path, index=False)
    raw.to_csv(audit_path, index=False)
    raw.loc[~raw["common_valid"]].to_csv(excluded_path, index=False)

    all_sets = set(pd.to_numeric(raw["set"], errors="coerce").dropna().astype(int))
    kept_sets = set(clean["set"].astype(int))
    summary = {
        "input_file": str(Path(args.csv).resolve()),
        "input_rows": int(len(raw)),
        "input_sets": int(raw["set"].nunique()),
        "clean_rows": int(len(clean)),
        "clean_sets": int(clean["set"].nunique()),
        "excluded_rows": int((~raw["common_valid"]).sum()),
        "sets_removed_entirely": sorted(all_sets - kept_sets),
        "max_systematic_ratio": float(args.max_systematic_ratio),
        "max_local_xsec_pull": float(args.max_local_xsec_pull),
        "max_local_xsec_ratio": float(args.max_local_xsec_ratio),
        "max_local_diff_pull": float(args.max_local_diff_pull),
        "phi_conversion": "phi=deg2rad(((phi_deg+180)%360)-180); symmetry center is 0 deg",
        "total_error_definition": "sqrt(stat^2 + max(abs(sys_plus),abs(sys_minus))^2)",
        "flag_counts": {flag: int(raw[flag].sum()) for flag in exclusion_flags},
        "informational_flag_counts": {
            "flag_xsec_systematic_missing": int(raw["flag_xsec_systematic_missing"].sum()),
            "flag_xsec_diff_systematic_missing": int(raw["flag_xsec_diff_systematic_missing"].sum()),
        },
        "clean_rows_by_experiment": {
            str(key): int(value) for key, value in clean.groupby("experiment").size().items()
        },
        "clean_sets_by_experiment": {
            str(key): int(value) for key, value in clean.groupby("experiment")["set"].nunique().items()
        },
        "outputs": {
            "clean_csv": str(clean_path),
            "audit_csv": str(audit_path),
            "excluded_rows_csv": str(excluded_path),
            "summary_json": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
