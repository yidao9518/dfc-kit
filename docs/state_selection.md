# State-count selection

`K` is the number of dynamic states: cluster centers for KMeans and hidden
states for a Gaussian HMM. `dfc-kit` selects it by comparing held-out scores
for a small candidate set.

For each subject-disjoint fold, fit every candidate `K` on the training
subjects and score it on the same validation subjects. Scores are averaged
within subject, then across folds. KMeans uses mean squared distance to the
nearest frozen center (lower is better); HMM uses mean log likelihood per
sample (higher is better). Repeated seeds are averaged within subject and are
not counted as extra participants.

```python
from dfckit.states import compare_state_model_scores

comparison = compare_state_model_scores(score_reports)
print(comparison.selection.best_n_states)
print(comparison.selection.one_standard_error_n_states)
```

The comparison requires one model family and feature setting, identical
candidate and seed grids in every fold, identical held-out runs within a fold,
and no subject appearing in more than one validation fold.

The returned `StateCountComparison` contains fold scores, candidate means and
standard errors, ranks, and both the best-K and one-standard-error choices.
`state_count_comparison_payload` and `write_state_count_comparison` provide a
small JSON summary for reports.
