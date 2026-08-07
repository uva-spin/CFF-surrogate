# Quick start

## Inspect without retraining

```bash
python validate_bundle.py
```

Then open these files in a browser:

```text
joint_cff_extraction/index.html
cff_experimental_replica_surfaces/index.html
reports/cff_uncertainty_colored_surfaces.html
```

## Regenerate only the uncertainty-colored visualization

```bash
python plot_cff_uncertainty_colored_surfaces.py \
  --surface-csv cff_experimental_replica_surfaces/cff_surface_experimental_bands.csv \
  --point-csv cff_experimental_replica_surfaces/cff_set_experimental_bands.csv \
  --out cff_uncertainty_colored_surfaces.html
```

## Short smoke-test rerun

The full central and replica training stages are deliberately substantial. For a
small functional test, prepare the data normally, use shortened epochs for the
cross-section model, pass `--quick` to the difference driver, shorten the CFF
stages, and use only a few replicas. These outputs are diagnostic only and must
not replace the included production checkpoints.

```bash
python prepare_dvcs_xsec_diff_data.py \
  --csv data/raw_dvcs_xsec_diff.csv \
  --outdir smoke_prepared

python dvcs_xsec_direct_dnn_optimized.py \
  --csv smoke_prepared/dvcs_xsec_diff_common_clean.csv \
  --outdir smoke_xsec \
  --epochs 80 --finetune-epochs 20 --no-html

python train_xsec_diff_surrogate.py \
  --csv smoke_prepared/dvcs_xsec_diff_common_clean.csv \
  --xsec-checkpoint smoke_xsec/xsec_surrogate.pt \
  --outdir smoke_diff --quick

python dvcs_joint_cff_extraction.py \
  --predictions smoke_diff/predictions_with_pulls.csv \
  --bkm-module bkm10_observables_corrected.py \
  --outdir smoke_cff \
  --seeds 11 \
  --pretrain-epochs 30 \
  --physics-stages 30:1e-3,20:3e-4
```

A smoke test may fail the strict domain cuts or produce poor physics quality; its
purpose is only to verify installation and code paths.
