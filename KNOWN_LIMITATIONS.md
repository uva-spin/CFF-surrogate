# Known limitations and interpretation boundaries

1. **Effective H-dominance extraction.** The central CFF model frees only
   ReH and ImH. ReE, ImE, ReHtilde, ImHtilde, ReEtilde, and ImEtilde are fixed
   to zero. The outputs are therefore effective CFFs under this reduction.

2. **Missing full covariance.** The supplied data contain pointwise statistical
   and systematic scales but no full bin-to-bin or cross-observable covariance
   matrix. The baseline experimental replicas are row-wise independent and use
   zero cross-section/difference correlation unless explicitly changed.

3. **Experimental component only.** The q16/q84 surfaces in the included final
   report propagate experimental replica variation. Algorithmic seed variation
   and methodological alternatives are not folded into those bands.

4. **Finite replica count.** The included checkpoint contains 50 replicas. Its
   split-half comparison changes the median 68% widths by about 17--18%. Use
   100--300 replicas for more stable production percentiles.

5. **Irregular support.** The DNN can return a value anywhere numerically, but
   only points inside the stored common-support mask should be treated as
   interpolation. A rectangular min/max box is not a reliability domain.

6. **Cleaning decisions are auditable, not invisible.** The preparation script
   removes unusable placeholders, malformed error entries, one exact duplicate,
   and a very small number of catastrophic isolated spikes. The complete audit
   and excluded-row tables are included so these decisions can be varied.

7. **BKM implementation choice.** `bkm10_observables_corrected.py` uses the
   Ktilde/K kinematic definition adopted from `BHDVCS_tf_modified.SetKinematics`.
   The unmodified reference implementation is included under `reference/` for
   comparison but is not imported by the production pipeline.

8. **Higher twist and model-systematic effects.** The current extraction does
   not claim an exhaustive treatment of twist-3, target-mass, finite-t, radiative,
   or CFF-prior systematics. These belong in a separate methodological ensemble.
