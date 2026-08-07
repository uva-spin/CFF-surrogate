# DVCS cross-section and beam-helicity-difference surrogate workflow

This workflow uses the published central columns in `data(20260807-003945).csv`:

- `exp d4sig (nb/Gev^4)` for the helicity-averaged unpolarized cross section;
- `exp del4sig (nb/GeV^4)` for the beam-helicity cross-section difference.

The `dsig` and `delsig` columns are not treated as central measurements. They
behave like one sampled pseudo-data realization of the published columns.

## 1. Prepare the common clean data table

```bash
python prepare_dvcs_xsec_diff_data.py \
  --csv 'data(20260807-003945).csv' \
  --outdir dvcs_xsec_diff_prepared
```

The preparation script:

- recomputes total uncertainties as
  `sqrt(stat^2 + max(abs(sys+), abs(sys-))^2)`;
- converts degrees to centered radians using
  `phi = deg2rad(((phi_deg + 180) % 360) - 180)`;
- retains only rows with usable cross section and cross-section difference;
- removes one exact duplicate;
- excludes obvious malformed systematic-error entries;
- excludes only catastrophic isolated transcription-like spikes found by a
  robust low-order within-set quality-control fit;
- writes a row-by-row audit rather than silently correcting source values.

For the supplied file the final common table contains 6,824 rows in 369 sets.

## 2. Train the common-support cross-section surrogate

```bash
python dvcs_xsec_direct_dnn_optimized.py \
  --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
  --outdir xsec_surrogate
```

The cross-section model is positive and exactly even in phi:

```text
u = (1-cos(phi))/2
log(model_xsec) = DNN(k,Q2,xB,t,u)
```

## 3. Train the companion cross-section-difference surrogate

The one-command production driver performs the data fit and gradual smoothness
polish used for the supplied checkpoint:

```bash
python train_xsec_diff_surrogate.py \
  --csv dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv \
  --xsec-checkpoint xsec_surrogate/xsec_surrogate.pt \
  --outdir xsec_diff_surrogate
```

The companion representation is

```text
u = (1-cos(phi))/2
amplitude = DNN(k,Q2,xB,t,u)
model_bsa = tanh(sin(phi)*amplitude)
model_xsec_diff = model_xsec * model_bsa
```

Consequently:

- `model_bsa(-phi) = -model_bsa(phi)` exactly;
- `model_xsec_diff(-phi) = -model_xsec_diff(phi)` exactly;
- `|model_bsa| < 1` by construction;
- the cross-section and difference models use the same input normalization and
  are directly compatible for the later BKM/CFF fit.

The final output directory contains only the useful artifacts:

```text
xsec_diff_surrogate/
  index.html
  xsec_diff_surrogate.pt
  predictions_with_pulls.csv
  set_metrics.csv
  metrics.json
  training_history.csv
```

`index.html` is self-contained and embeds all set plots. No per-set PNG files or
zip archive are produced.

## 4. Main prediction columns

The final prediction CSV uses simple names:

- `model_xsec`
- `model_bsa`
- `model_xsec_diff`
- `pull`

The measured central-value ratio `xsec_diff / unp_beam_unp_target_xsec` is kept
as a diagnostic only. Its uncertainty should ultimately come from matched
cross-section/difference replicas rather than an independent ratio formula.
