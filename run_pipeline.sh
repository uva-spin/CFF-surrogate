#!/usr/bin/env bash
set -euo pipefail

# Run from the top-level directory of the unzipped bundle.
RAW_DATA="${RAW_DATA:-data/raw_dvcs_xsec_diff.csv}"
THREADS="${THREADS:-4}"
N_REPLICAS="${N_REPLICAS:-100}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

python prepare_dvcs_xsec_diff_data.py \
  --csv "$RAW_DATA" \
  --outdir dvcs_xsec_diff_prepared

python dvcs_xsec_direct_dnn_optimized.py \
  --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
  --outdir xsec_surrogate \
  --threads "$THREADS"

python train_xsec_diff_surrogate.py \
  --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
  --xsec-checkpoint xsec_surrogate/xsec_surrogate.pt \
  --outdir xsec_diff_surrogate \
  --threads "$THREADS"

python dvcs_joint_cff_extraction.py \
  --predictions xsec_diff_surrogate/predictions_with_pulls.csv \
  --bkm-module bkm10_observables_corrected.py \
  --outdir joint_cff_extraction \
  --threads "$THREADS"

python dvcs_cff_experimental_replica_surfaces.py \
  --predictions xsec_diff_surrogate/predictions_with_pulls.csv \
  --domain-audit joint_cff_extraction/common_domain_audit.csv \
  --cff-sets joint_cff_extraction/cff_sets.csv \
  --xsec-checkpoint xsec_surrogate/xsec_surrogate.pt \
  --diff-checkpoint xsec_diff_surrogate/xsec_diff_surrogate.pt \
  --cff-checkpoint joint_cff_extraction/cff_surrogate.pt \
  --xsec-script dvcs_xsec_direct_dnn_optimized.py \
  --diff-script dvcs_xsec_diff_direct_dnn_optimized.py \
  --joint-script dvcs_joint_cff_extraction.py \
  --bkm-module bkm10_observables_corrected.py \
  --outdir cff_experimental_replica_surfaces \
  --n-replicas "$N_REPLICAS" \
  --threads "$THREADS"

python plot_smooth_cff_surfaces.py \
  --checkpoint joint_cff_extraction/cff_surrogate.pt \
  --cff-sets joint_cff_extraction/cff_sets.csv \
  --local-cff joint_cff_extraction/local_h2_extraction.csv \
  --out cff_smooth_surfaces.html

python plot_cff_uncertainty_colored_surfaces.py \
  --surface-csv cff_experimental_replica_surfaces/cff_surface_experimental_bands.csv \
  --point-csv cff_experimental_replica_surfaces/cff_set_experimental_bands.csv \
  --out cff_uncertainty_colored_surfaces.html

printf '\nPipeline complete. Open:\n'
printf '  joint_cff_extraction/index.html\n'
printf '  cff_experimental_replica_surfaces/index.html\n'
printf '  cff_uncertainty_colored_surfaces.html\n'
