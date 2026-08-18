# Experimental paired NBS

`dfckit.inference.paired_nbs` implements the network-based statistic for paired
edge differences. It is marked experimental because the result depends on a
fixed edge-level threshold and component statistic that must be declared before
examining the result.

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

## Tails, components, and FWE

Positive and negative supra-threshold edges are connected separately, even when
they share nodes. `alternative="two-sided"` takes the maximum component across
both tails in every permutation. `greater` and `less` use only their specified
tail.

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
)

for component in result.at_threshold(3.0).components:
    print(component.direction, component.edge_extent, component.fwe_pvalue)
```
