# Instantaneous edges

ETS and MTD use the same edge representation:

```text
E_ij(k) = z_i(k) * z_j(k),  i < j
```

They differ only in what one sample `z(k)` represents.

| Method | Input row | Row timing | Standardization |
| --- | --- | --- | --- |
| `ETS` | retained BOLD values | one frame | separately in each retained segment |
| `MTD` | adjacent BOLD differences | one frame interval | across all valid derivatives in a run |

Neither method crosses a censor gap. ETS omits a retained segment with only one
frame because it cannot be standardized; MTD omits it because it has no
derivative.

## Compute edges

```python
from dfckit.connectivity import ETS, MTD

ets = ETS().transform(run)
mtd = MTD().transform(run)

visual_motor = ets.cross_block(left=[0, 1, 2], right=[3, 4, 5])
within_motor = mtd.within_block(nodes=[3, 4, 5])
```

Both calls return an `InstantaneousEdgeResult`. `features` is a
samples-by-edge matrix in upper-triangular ROI order. It also records the
original frame interval and retained segment for every row, so the same result
can enter state analysis without crossing censor gaps.

Use `rss()` when only the instantaneous whole-network amplitude is needed:

```python
amplitude = ETS().rss(run).rss
```

This avoids constructing the edge matrix and uses

```text
RSS(k)^2 = ((sum_i z_i(k)^2)^2 - sum_i z_i(k)^4) / 2
```

## State analysis

```python
from dfckit.states import instantaneous_edge_sequences, fit_kmeans_states

results = [ETS().transform(run) for run in dataset.runs]
sequences = instantaneous_edge_sequences(results)
fit = fit_kmeans_states(sequences, n_states=4, seed=17)
```

Frame rows from ETS have identical start and end indices. MTD rows keep both
ends of the derivative interval. The state model treats either representation
as a sequence of edge-feature rows and always keeps retained segments separate.

## Chunked storage

```python
from dfckit.storage import write_instantaneous_edge_store

store = write_instantaneous_edge_store(
    "/path/to/features.store",
    dataset.runs,
    ETS(),
    chunk_size=128,
)
```

The writer builds edge rows in bounded chunks and preserves their sample
intervals and segment boundaries.
