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

`infer_paired_endpoints` combines the sign-flip test, bootstrap interval, and
declared-family BH correction for named endpoints. Each testable endpoint also
reports paired Cohen `dz`, defined as the mean complete-pair difference divided
by its sample standard deviation; it is missing when all differences are
identical.

One source artifact may contain endpoints that answer different scientific
questions. Pass an explicit `endpoint_names` sequence to test only one declared
family:

```python
from dfckit.inference import infer_paired_endpoints

result = infer_paired_endpoints(
    endpoint_payload,
    condition_a="on",
    condition_b="off",
    fdr_family="low-rank reconstruction across declared ranks",
    endpoint_names=(
        "heldout_r2.rank_4.mean",
        "heldout_r2.rank_5.mean",
        "heldout_r2.rank_6.mean",
    ),
    n_permutations=10_000,
    n_bootstrap=10_000,
    seed=20260905,
)
```

BH correction then includes exactly those selected, testable endpoints. An
omitted selection retains the original all-endpoint behavior. An explicitly
empty selection, duplicate name, or name absent from the source artifact is an
error. The result records the selection mode, requested and selected names,
and the number of endpoints available in the source artifact.

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

## Independent endpoint groups

Two independent endpoint cohorts can be compared with
`infer_independent_endpoints`. The model uses a declared group indicator and
optional numeric covariates with HC3 standard errors. Subject identity is the
pair of group and subject, so textual identifiers may overlap between cohorts.
The result includes adjusted group coefficients, unadjusted Hedges' g, full
model R2, group incremental in-sample R2, and q values inside the declared
family. See [Independent group inference](group_inference.md) for the endpoint
artifact and covariate contracts.
