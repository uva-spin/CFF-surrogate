#!/usr/bin/env python3
"""Small integrity and compatibility check for the unzipped pipeline bundle."""
from pathlib import Path
import json
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
required = [
    "data/raw_dvcs_xsec_diff.csv",
    "dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv",
    "xsec_surrogate/xsec_surrogate.pt",
    "xsec_diff_surrogate/xsec_diff_surrogate.pt",
    "joint_cff_extraction/cff_surrogate.pt",
    "cff_experimental_replica_surfaces/cff_replica_ensemble.pt",
    "cff_experimental_replica_surfaces/cff_surface_experimental_bands.csv",
]
missing = [name for name in required if not (ROOT / name).exists()]
if missing:
    raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))

clean = pd.read_csv(ROOT / "dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv")
cff_sets = pd.read_csv(ROOT / "joint_cff_extraction/cff_sets.csv")
surfaces = pd.read_csv(ROOT / "cff_experimental_replica_surfaces/cff_surface_experimental_bands.csv")
ensemble = torch.load(ROOT / "cff_experimental_replica_surfaces/cff_replica_ensemble.pt", map_location="cpu", weights_only=False)

print(f"Clean observable rows: {len(clean):,}")
print(f"Clean observable sets: {clean['set'].nunique():,}")
print(f"Selected CFF sets: {len(cff_sets):,}")
print(f"Stored surface-grid rows: {len(surfaces):,}")
print(f"CFF replicas in checkpoint: {ensemble['n_replicas']}")
print(f"Calibration: {ensemble.get('calibration_formula', 'not recorded')}")
print("Bundle validation passed.")
