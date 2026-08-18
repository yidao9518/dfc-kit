# Paired inference

The inference layer operates on one paired difference per participant. It does
not infer pairing, subtraction direction, permutation unit, comparison family,
or covariate meaning from a table.

## Sign-flip test

```python
from dfckit.inference import paired_sign_flip

result = paired_sign_flip(
    on_minus_off,
    subject_ids,
    n_permutations=10_000,
    seed=20260818,
    alternative="two-sided",
)
```

Every participant receives one independently sampled sign. Monte Carlo tests
use the plus-one correction and require an explicit seed. Exact enumeration is
available for at most 20 participants. Duplicate subject IDs are rejected.

## Participant bootstrap

```python
from dfckit.inference import paired_bootstrap_mean_ci

interval = paired_bootstrap_mean_ci(
    on_minus_off,
    subject_ids,
    n_resamples=10_000,
    seed=20260818,
    confidence=0.95,
)
```

The current implementation is a percentile interval for the paired mean and
resamples participants, not sessions, frames, or windows.

## HC3 conditional paired effect

```python
from dfckit.inference import paired_hc3

result = paired_hc3(
    on_minus_off,
    covariates=np.column_stack([delta_fd, delta_censor]),
    subject_ids=subject_ids,
    covariate_names=("delta_mean_fd", "delta_censor"),
    difference_direction="ON minus OFF",
)
```

The intercept is the paired difference when every supplied covariate equals
zero. Covariates are not silently centered. The implementation rejects
rank-deficient designs, non-positive residual degrees of freedom, and unit
leverage. It uses HC3 sandwich standard errors and finite-sample Student-t
tests without requiring statsmodels.

## BH-FDR

```python
from dfckit.inference import benjamini_hochberg

fdr = benjamini_hochberg(
    pvalues,
    family="six prespecified visual-motor circuit endpoints",
)
```

The family description is mandatory and travels with the adjusted values.
Missing p values remain missing and are not counted as tests.
