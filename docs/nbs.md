# Paired NBS

`dfckit.inference.paired_nbs` implements the network-based statistic for paired
edge differences. The strict edge-level threshold, component statistic,
contrast direction, tested tail, and permutation count are explicit inputs and
should be declared before examining the result.

The CLI supports two explicit upstream definitions. `static-fc-endpoints`
computes a complete Fisher-z edge vector from all retained frames in each
XCP-D acquisition. Alternatively, `summarize-store` can reduce a dynamic edge
FeatureStore to an acquisition-level statistic before `infer-paired-nbs`.
Whole-acquisition static FC and a mean over sliding-window FC are not treated
as interchangeable inputs; the output retains its source contract.

## Statistical unit and contrast

The input has one complete edge vector per participant, normally formed by
subtracting two paired conditions. `difference_direction` records that
subtraction explicitly. A classical intercept t statistic is computed for each
edge. With nuisance variables, each variable is centered and the intercept is
the adjusted paired difference at the observed mean nuisance values.

Every permutation multiplies a participant's complete edge vector by one sign;
edges are never permuted independently. With confounds, the module fits the
centered nuisance-only reduced model, sign-flips each participant's complete
residual edge vector, and adds the nuisance fit back before recomputing t.

`confounds` are participant-level nuisance variables that may explain part of
the paired edge difference without being the effect under test. For example,
an ON-minus-OFF analysis may supply ON-minus-OFF mean framewise displacement
and censor fraction. Centering makes the intercept the adjusted mean paired
difference at the sample-average nuisance values. The unadjusted paired test is
the standard NBS path; nuisance-residual sign flips are an extension for a
prespecified sensitivity model.

## Tails, component definitions, and FWE

`component_sign_mode` exposes two definitions of a two-sided network
component:

- `separate` constructs positive (`t > threshold`) and negative
  (`t < -threshold`) components independently, even when opposite-sign edges
  share nodes. The two-sided permutation null contains the larger of the
  positive and negative maximum components. One-sided `greater` and `less`
  analyses use the corresponding tail only.
- `pooled` thresholds `abs(t) > threshold` and constructs a single unsigned
  support graph, so adjacent positive and negative edges may belong to one
  component. Its permutation null contains the maximum pooled component from
  each permutation. This mode requires `alternative="two-sided"`.

These definitions answer different statistical questions. `separate` detects
connected effects that share a direction; `pooled` detects connected departures
from zero regardless of whether individual edges increase or decrease. Neither
definition has inherent biological priority. `separate` is the default for
backward compatibility with existing `dfc-kit` analyses. In pooled results,
each component has `direction="pooled"`; use `observed_t` and
`observed_mean_difference` to inspect the sign of its individual edges.
`null_maximum` always stores the active family-wise null. In pooled mode it is
identical to `null_pooled`, while `null_positive` and `null_negative` are
`None`.

Two component statistics are explicit:

- `edge_extent`: number of supra-threshold edges in the component;
- `sum_abs_statistic`: sum of absolute edge t statistics in the component.

The Monte Carlo p value compares each observed component with the permutation
maximum and uses the plus-one correction. It therefore controls component-level
family-wise error across components and tested tails **within one threshold**.
It does not make every edge inside a significant component an independently
corrected discovery.

## Fixed-threshold sensitivity

Several thresholds can be evaluated in one call. They share the same
participant sign flips, which removes avoidable Monte Carlo differences. Each
threshold nevertheless has its own maximum-component null distribution. The
thresholds form a sensitivity analysis; they are not independent replications,
and the API does not correct post hoc selection of whichever threshold looks
most favorable.

## Validation and implementation choice

The unadjusted one-sided implementation was compared with BCTPy 0.6.1
`nbs_bct`, a Python port of the Brain Connectivity Toolbox algorithm. With
identical participant sign flips, positive and negative component edge sets and
the complete maximum-component null arrays matched exactly. BCTPy reports
`hits / permutations`; `dfc-kit` uses the plus-one Monte Carlo correction
`(hits + 1) / (permutations + 1)`.

The native BCTPy two-sided mode thresholds `abs(t)` and can join positive and
negative edges that share nodes. This corresponds to
`component_sign_mode="pooled"`; with identical sign flips, the observed pooled
edge set and complete maximum-component null array matched BCTPy. The
`separate` mode is a direction-separated two-sided extension. BCTPy has no
interface for the centered-confound model used here. Independent regression
tests reconstruct the OLS statistics with NumPy and the graph components with
SciPy, and compare observed components and complete null arrays for both sign
modes and for unadjusted and confound-adjusted analyses.

## References

- Zalesky A, Fornito A, Bullmore ET. Network-based statistic: identifying
  differences in brain networks. *NeuroImage*. 2010;53(4):1197-1207.
  <https://doi.org/10.1016/j.neuroimage.2010.06.041>
- Winkler AM, Ridgway GR, Webster MA, Smith SM, Nichols TE. Permutation
  inference for the general linear model. *NeuroImage*. 2014;92:381-397.
  <https://doi.org/10.1016/j.neuroimage.2014.01.060>

```python
from dfckit.inference import paired_nbs

result = paired_nbs(
    differences,
    subject_ids,
    edge_i,
    edge_j,
    n_nodes=32,
    thresholds=(2.0, 2.5, 3.0),
    n_permutations=10_000,
    seed=20260725,
    difference_direction="condition B minus condition A",
    alternative="two-sided",
    component_statistic="edge_extent",
    component_sign_mode="separate",
)

for component in result.at_threshold(3.0).components:
    print(component.direction, component.edge_extent, component.fwe_pvalue)
```
