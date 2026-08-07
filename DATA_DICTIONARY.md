# Data dictionary

## Raw input

`data/raw_dvcs_xsec_diff.csv` is a renamed copy of the uploaded combined file.
Important source columns are:

| Source column | Meaning |
|---|---|
| `set`, `bin` | Published kinematic-set and angular-bin identifiers |
| `k` | Beam energy in GeV |
| `Q2` | Photon virtuality in GeV^2 |
| `xB` | Bjorken x |
| `t` | Momentum transfer in GeV^2; physical rows are negative |
| `phi` | Source azimuth in degrees |
| `exp d4sig (nb/Gev^4)` | Published unpolarized cross section |
| `d4sig_stat`, `d4sig_sys+`, `d4sig_sys-` | Cross-section uncertainties |
| `exp del4sig (nb/GeV^4)` | Published beam-helicity cross-section difference |
| `del4sig_stat`, `del4sig_sys+`, `del4sig_sys-` | Difference uncertainties |
| `experiment`, `link` | Source metadata |

The raw `dsig` and `delsig` fields are not used as central values.

## Clean common table

`dvcs_xsec_diff_prepared/dvcs_xsec_diff_common_clean.csv` uses canonical names:

| Clean column | Meaning |
|---|---|
| `q_squared`, `x_b`, `phi` | Canonical kinematics; `phi` is centered radians |
| `unp_beam_unp_target_xsec` | Central sigma_UU |
| `unp_beam_unp_target_xsec_err` | Total sigma_UU uncertainty |
| `xsec_diff` | Central Delta-sigma_LU |
| `xsec_diff_err` | Total Delta-sigma_LU uncertainty |
| `bsa_from_central_values` | Diagnostic ratio `xsec_diff/xsec`; not the fit target |
| `experiment_year`, `experiment` | Group labels used for balanced training |

The audit file contains boolean flags and exclusion reasons for every original
row.

## Observable-surrogate prediction table

`xsec_diff_surrogate/predictions_with_pulls.csv` contains

| Column | Meaning |
|---|---|
| `model_xsec` | Positive hard-even cross-section prediction |
| `model_bsa` | Bounded hard-odd beam-spin asymmetry prediction |
| `model_xsec_diff` | `model_xsec * model_bsa` |
| `pull` | `(model_xsec_diff - xsec_diff)/xsec_diff_err` |

## Central CFF products

`joint_cff_extraction/cff_sets.csv` contains one row per selected kinematic set,
including central `ReH`, `ImH`, and algorithmic seed-spread diagnostics.

`joint_cff_extraction/common_domain_audit.csv` records why each clean set was or
was not admitted to the common interpolation and strict CFF extraction domains.

`joint_cff_extraction/cff_predictions.csv` contains the BKM reconstruction of
both observables for each selected angular row.

## Experimental CFF products

`cff_experimental_replica_surfaces/cff_set_experimental_bands.csv` reports the
CFF ensemble at the 155 selected set centers. Important columns include

```text
ReH_central ReH_mean ReH_std ReH_q16 ReH_q50 ReH_q84 ReH_half_68
ImH_central ImH_mean ImH_std ImH_q16 ImH_q50 ImH_q84 ImH_half_68
cov_ReH_ImH corr_ReH_ImH
```

`cff_experimental_replica_surfaces/cff_surface_experimental_bands.csv` contains
all stored slice grids. `slice_kind` is one of `q2`, `xb`, or `minus_t`;
`inside_support` is the irregular-domain mask; `support_distance` is the local
standardized-feature distance used by that mask.
