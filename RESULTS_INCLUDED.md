# Included trained results

This bundle preserves the currently selected central and experimental-replica
artifacts so the results can be inspected without retraining.

## Prepared common data

- 7,055 input rows in 378 sets;
- 6,824 retained common rows in 369 sets;
- 231 excluded rows;
- completely removed sets: 19--22 and 133--137.

## Cross-section-difference companion

The included selected model has 6 hidden layers of 256 SiLU units.

- chi2 per point: 1.13195;
- pull RMS: 1.06393;
- 67.44% of points within 1 sigma;
- 93.63% within 2 sigma;
- exact odd-symmetry residual: zero to numerical precision.

## Central simultaneous CFF extraction

- 166 common interpolation sets;
- 155 strict CFF extraction sets;
- 3,363 selected angular rows;
- central simultaneous BKM point chi2/N: 0.3593;
- H-only Jacobian median condition number: 1.98;
- larger four- and six-component parameterizations were strongly ill-conditioned.

## Experimental CFF ensemble

- 50 matched observable replicas;
- median ReH 68% half-width at selected centers: 0.1202;
- median ImH 68% half-width: 0.0934;
- median ReH--ImH replica correlation: 0.353;
- median replica CFF reconstruction chi2/N: 0.380;
- null-calibration RMS shifts: 0.0300 in ReH and 0.0226 in ImH.

The exact machine-readable values are in each output directory's `metrics.json`.
