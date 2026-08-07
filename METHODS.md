# Methods and design choices

## 1. Paired observable data

The raw table contains the published central values

- `exp d4sig (nb/Gev^4)`: helicity-averaged unpolarized cross section;
- `exp del4sig (nb/GeV^4)`: beam-helicity cross-section difference.

The columns `dsig` and `delsig` behave like one sampled pseudo-data realization
and are not used as the central measurements.

Asymmetric systematics are reduced to a conservative symmetric scale,

\[
\delta_{\rm sys}=\max(|\delta_{\rm sys+}|,|\delta_{\rm sys-}|),
\qquad
\delta_{\rm tot}=\sqrt{\delta_{\rm stat}^2+\delta_{\rm sys}^2}.
\]

Azimuth is mapped from degrees on \([0,360)\) to centered radians on
\([-\pi,\pi)\):

\[
\phi=\frac{\pi}{180}
\left[((\phi_{\rm deg}+180)\bmod 360)-180\right].
\]

The included clean result has 6,824 common rows in 369 sets. The audit and
excluded-row tables preserve the exact decisions.

## 2. Unpolarized cross-section surrogate

The model directly predicts standardized log cross section from
\((k,Q^2,x_B,t,u)\), where

\[
u=\frac{1-\cos\phi}{2}.
\]

This makes the cross section exactly even in \(\phi\) and positive after
exponentiation. A staged training loss combines a log-scale warmup, quoted-error
chi-square, experiment balancing, and mild arc-length/curvature penalties on
\(\log\sigma(\phi)\).

## 3. Cross-section-difference/BSA surrogate

A second DNN predicts an even amplitude in \(u\). The physical odd observable is
constructed analytically:

\[
A_{LU}=\tanh[\sin\phi\,g_\psi(k,Q^2,x_B,t,u)],
\qquad
\Delta\sigma_{LU}=\sigma_{UU}A_{LU}.
\]

The loss is applied to the measured cross-section difference, not to a noisy
pointwise ratio. The BSA is a derived model output.

## 4. Common interpolation and extraction domain

The central CFF analysis first identifies sets with adequate common angular
coverage and acceptable surrogate reconstruction for both observables. The
current defaults require at least 12 common points, both signs of centered
\(\phi\), a circular angular gap no larger than 90 degrees, and set-level
surrogate chi-square no larger than 2 for each observable.

The H-dominance extraction domain additionally requires local BKM reconstruction
chi-square no larger than 2 and a two-CFF Jacobian condition number no larger
than 10. The current central analysis retains 166 interpolation sets and 155 CFF
extraction sets.

## 5. Simultaneous BKM CFF extraction

The CFF DNN depends only on hadronic kinematics:

\[
(Q^2,x_B,t)\rightarrow(\operatorname{Re}\mathcal H,\operatorname{Im}\mathcal H).
\]

For each row, the BKM layer uses \(k\) and \(\phi\) to predict both
\(\sigma_{UU}\) and \(\Delta\sigma_{LU}\). The BKM observable is represented as
an exact quadratic polynomial in the CFF components for efficient PyTorch
training; the implementation was validated against direct numerical evaluation
to approximately \(10^{-13}\) in the cross section and \(10^{-14}\) in the
helicity difference.

Local Jacobian studies showed that the two-component H-only parameterization is
well conditioned, while four- and six-component alternatives are strongly
degenerate on the present common observable set. The outputs are therefore
labeled effective H-dominance CFFs.

## 6. Experimental replica propagation

For each replica index, the cross section and signed difference are sampled on
the same retained rows. The baseline uses row-wise Gaussian sampling. Cross
sections are redrawn if a sample is nonpositive; the signed difference is not
clipped. The current source file provides no bin-to-bin or cross-observable
covariance matrix, so the default cross-observable correlation is zero.

Each observable replica is warm-started from its central surrogate. A CFF DNN is
then warm-started from the central CFF model and refitted to both replica
surfaces simultaneously. The strict central domain is frozen across replicas.

A null-replica run with the same finite training schedule is used to calibrate
away deterministic optimizer drift. The pointwise experimental bands are then
computed from the calibrated CFF-surface ensemble:

\[
q_{16}(x),\quad q_{50}(x),\quad q_{84}(x).
\]

The tables also retain standard deviation and the ReH--ImH replica correlation.

## 7. Support mask and visualization

The selected CFF points form an irregular cloud in transformed coordinates

\[
(\log Q^2,\operatorname{logit}x_B,\log(-t)).
\]

Surface plots are masked using a local nearest-neighbor distance criterion. A
point inside the independent min/max ranges of all variables is not automatically
inside the trusted joint support.

The standard uncertainty report plots the mean together with translucent q16
and q84 surfaces. `plot_cff_uncertainty_colored_surfaces.py` instead uses height
for the mean CFF and color for the local 68% half-width.
