# Multiplication of temporal derivatives

For every ROI `i`, MTD begins with first differences between genuinely
adjacent retained frames:

```text
d_i(t) = x_i(t + 1) - x_i(t)
```

No difference is formed across an XCP-D censor gap. All valid derivative rows
from one run are concatenated and standardized once for each ROI using the
population standard deviation:

```text
z_i(t) = (d_i(t) - mean(d_i)) / sd_population(d_i)
```

The instantaneous MTD value for edge `(i, j)` is:

```text
MTD_ij(t) = z_i(t) * z_j(t)
```

`MTD().transform(run)` returns all upper-triangular edge products together with
the original start/end frame of every valid derivative. It does not smooth the
products over a secondary temporal window. Run-level or block-level averaging
is a separate summary decision.

```python
from dfckit.connectivity import MTD, cross_block_mtd

result = MTD().transform(run)
visual_motor = cross_block_mtd(
    result.standardized_derivatives,
    left=[0, 1, 2, 3],
    right=[4, 5, 6, 7],
)
run_mean = visual_motor.mean()
```

The global run-level standardization is deliberate. It matches the audited
reference definition and must not be replaced by separate standardization of
each censor-bounded segment.
