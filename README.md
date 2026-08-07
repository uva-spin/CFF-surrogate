# DVCS observable-surrogate to CFF-surface pipeline

This bundle contains the complete working pipeline used to go from paired DVCS
unpolarized cross-section and beam-helicity cross-section-difference data to
smooth effective CFF surfaces with propagated experimental uncertainty:

\[
(\sigma_{UU},\,\Delta\sigma_{LU})
\rightarrow
\text{symmetry-constrained observable surrogates}
\rightarrow
\text{simultaneous BKM fit}
\rightarrow
(\operatorname{Re}\mathcal H_{\rm eff},\,\operatorname{Im}\mathcal H_{\rm eff})
\rightarrow
\text{experimental replica surfaces}.
\]

The zip is intentionally self-contained. It includes the source CSV supplied for
this analysis, the cleaned common-support table and audit, all production
scripts, the current central checkpoints, the current 50-replica CFF ensemble,
and the final HTML/CSV surface products.

## Fastest way to inspect the current result

After unzipping, open:

```text
cff_experimental_replica_surfaces/index.html
```

That report shows the replica-mean ReH and ImH surfaces together with their
pointwise 16th/84th percentile surfaces. A second report,

```text
reports/cff_uncertainty_colored_surfaces.html
```

uses surface height for the mean CFF value and surface color for the local
experimental 68% half-width.

## Software requirements

Python 3.10 or newer is recommended. Create an isolated environment and install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch may be installed separately with a CUDA-specific command on GPU systems.
The supplied scripts also run on CPU; the replica stage is the expensive part.

## End-to-end production workflow

Run all commands from the top-level directory of this bundle.

### 1. Prepare the common observable table

```bash
python prepare_dvcs_xsec_diff_data.py \
  --csv data/raw_dvcs_xsec_diff.csv \
  --outdir dvcs_xsec_diff_prepared
```

The clean table is

```text
dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv
```

The preparation stage keeps only rows with usable cross section and
cross-section difference, converts azimuth from degrees to centered radians,
constructs total errors from the statistical and asymmetric systematic errors,
removes exact duplicates, and records every exclusion in an audit table.

### 2. Train the positive, hard-even cross-section surrogate

```bash
python dvcs_xsec_direct_dnn_optimized.py \
  --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
  --outdir xsec_surrogate
```

The model is

\[
u=\frac{1-\cos\phi}{2},\qquad
\log\widehat\sigma_{UU}=f_\theta(k,Q^2,x_B,t,u),
\]

so \(\widehat\sigma_{UU}>0\) and
\(\widehat\sigma_{UU}(\phi)=\widehat\sigma_{UU}(-\phi)\) exactly.

### 3. Train the hard-odd cross-section-difference companion

```bash
python train_xsec_diff_surrogate.py \
  --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
  --xsec-checkpoint xsec_surrogate/xsec_surrogate.pt \
  --outdir xsec_diff_surrogate
```

The companion head learns

\[
\widehat A_{LU}
=\tanh\!\left[\sin\phi\;g_\psi(k,Q^2,x_B,t,u)\right],
\qquad
\widehat{\Delta\sigma}_{LU}
=\widehat\sigma_{UU}\widehat A_{LU}.
\]

It therefore satisfies

\[
\widehat{\Delta\sigma}_{LU}(-\phi)
=-\widehat{\Delta\sigma}_{LU}(\phi)
\]

and \(|\widehat A_{LU}|<1\) by construction.

### 4. Perform the central simultaneous BKM CFF extraction

```bash
python dvcs_joint_cff_extraction.py \
  --predictions xsec_diff_surrogate/predictions_with_pulls.csv \
  --bkm-module bkm10_observables_corrected.py \
  --outdir joint_cff_extraction \
  --threads 4
```

The CFF DNN has inputs \((Q^2,x_B,t)\), not \((k,\phi)\):

\[
(Q^2,x_B,t)
\longmapsto
(\operatorname{Re}\mathcal H_{\rm eff},\operatorname{Im}\mathcal H_{\rm eff}).
\]

Beam energy and azimuth enter only through the BKM observable layer. The current
parameterization is an explicitly tested two-CFF, H-dominance reduction; the six
other real/imaginary CFF components are fixed to zero. The script constructs the
strict common interpolation and extraction domain, performs local
identifiability checks, trains multiple seeds, and writes the selected central
checkpoint and diagnostics.

### 5. Propagate experimental errors into CFF surfaces

For a publication-oriented run, use at least 100 replicas and check convergence
against a larger ensemble:

```bash
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
  --n-replicas 100 \
  --threads 8
```

Each matched data replica is carried through both observable surrogates and one
warm-started simultaneous CFF fit. The domain is frozen from the central
analysis. The script applies a null-replica calibration to remove deterministic
finite-training drift:

\[
\mathcal F_{\rm cal}^{(r)}(x)
=\mathcal F_{\rm central}(x)
+\mathcal F_{\rm raw}^{(r)}(x)
-\mathcal F_{\rm null}(x).
\]

The final pointwise experimental interval is reported through the 16th and 84th
replica percentiles rather than assuming Gaussian CFF distributions.

### 6. Plot central or uncertainty-colored CFF surfaces

Central smooth surfaces:

```bash
python plot_smooth_cff_surfaces.py \
  --checkpoint joint_cff_extraction/cff_surrogate.pt \
  --cff-sets joint_cff_extraction/cff_sets.csv \
  --local-cff joint_cff_extraction/local_h2_extraction.csv \
  --out cff_smooth_surfaces.html
```

Mean elevation with experimental uncertainty encoded by color:

```bash
python plot_cff_uncertainty_colored_surfaces.py \
  --surface-csv cff_experimental_replica_surfaces/cff_surface_experimental_bands.csv \
  --point-csv cff_experimental_replica_surfaces/cff_set_experimental_bands.csv \
  --out cff_uncertainty_colored_surfaces.html
```

## Included current artifacts

The current bundle contains a complete 50-replica result for immediate use:

```text
xsec_surrogate/xsec_surrogate.pt
xsec_diff_surrogate/xsec_diff_surrogate.pt
joint_cff_extraction/cff_surrogate.pt
cff_experimental_replica_surfaces/cff_replica_ensemble.pt
```

The compact numerical products are also included, especially

```text
cff_experimental_replica_surfaces/cff_set_experimental_bands.csv
cff_experimental_replica_surfaces/cff_surface_experimental_bands.csv
```

The 50-replica result is sufficient to establish the band structure and scale.
The included metrics show split-half differences of order 17--18% in the 68%
widths, so a larger ensemble is recommended for final percentile precision.

## Scientific interpretation

The extracted quantities are

\[
\operatorname{Re}\mathcal H_{\rm eff},\qquad
\operatorname{Im}\mathcal H_{\rm eff}
\]

within the two-CFF H-dominance BKM reduction. They should not be described as an
unconditional extraction of all eight twist-2 CFF components. The final bands in
this bundle are the propagated experimental-replica component. Optimizer
variability, model reduction, higher-twist choices, domain variations, and
missing covariance information are separate algorithmic or methodological
uncertainties.

See `METHODS.md`, `KNOWN_LIMITATIONS.md`, and `DATA_DICTIONARY.md` for details.
