# Subject-balanced references

Reference models are learned transformations. They must record their fitted
participants, give each participant equal total influence, and keep reference
participants out of ordinary held-out scoring. `dfckit.reference` implements
these rules for feature templates and low-rank subspaces.

The utilities are cohort-neutral. A caller may use a healthy cohort, a control
condition, or another declared training set; the library does not assign
clinical meaning to the reference.

## Feature templates

`fit_feature_reference` first averages all observations belonging to the same
participant and then averages those participant means:

```text
participant_mean_s = mean(observations from participant s)
template = mean(participant_mean_s over participants)
```

Consequently, one participant with many sessions or windows cannot dominate a
participant with one session. Feature identity and order are stored in the
model.

```python
from dfckit.reference import fit_feature_reference, score_feature_reference

reference = fit_feature_reference(
    hc_window_features,
    hc_window_subjects,
    feature_keys,
)
similarity = score_feature_reference(
    reference,
    heldout_features,
    heldout_subjects,
    feature_keys,
)
```

Scores are row-wise Pearson spatial similarities. Constant rows or templates
are rejected. Any overlap between scoring subjects and `fit_subjects` is also
rejected unless the caller explicitly enables it.

For observations from the fitted reference participants,
`leave_one_subject_out_feature_similarity` uses a different template for each
participant and removes every observation from that participant before scoring
them. `subject_balanced_quantiles` gives each participant total weight one and
is suitable for thresholds derived from unequal numbers of windows.
`hierarchical_balanced_quantiles` adds explicit nested levels. For example,
levels `(subject, acquisition)` give subjects equal total weight, acquisitions
equal weight within subject, and windows equal weight within acquisition; its
quantiles use the left-continuous inverse empirical CDF.

## Subspace references

`fit_subspace_reference` accepts one or more orthonormal ROI-by-rank bases per
participant. It performs two levels of projection averaging:

1. Average `B @ B.T` across observations within each participant and recover a
   participant basis.
2. Average participant projectors with equal participant weight and recover the
   final reference basis.

At least three participants are required. Every reference participant is
compared with a template formed from all other participants, producing a
leave-one-subject-out distance distribution. The model stores its mean and
sample standard deviation.

```python
from dfckit.reference import fit_subspace_reference, score_subspace_reference

reference = fit_subspace_reference(
    hc_run_bases,
    hc_subjects,
    roi_names,
)
scores = score_subspace_reference(
    reference,
    pd_run_bases,
    pd_subjects,
    roi_names,
)
```

Raw distance is `sqrt(1 - projection_similarity)`. Standardized distance is

```text
(heldout_distance - mean(HC LOO distances)) / SD(HC LOO distances)
```

Positive standardized distance means farther from the reference than the mean
LOO reference participant. This sign has no automatic interpretation as
improvement, impairment, normalization, or compensation.

ROI identity, order, rank, and fitted participant IDs are checked before every
score. Reference scoring does not refit or update the template with held-out
observations.
