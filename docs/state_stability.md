# Repeated-fit state stability

State labels are arbitrary across optimization runs. Stability summaries are
therefore calculated only after each candidate fit is mapped to one declared
reference numbering. `dfc-kit` decodes a common cohort, aligns each candidate
with Hungarian matching, and then summarizes occupancy, dwell, switch, and
transition measures.

```bash
dfc-kit summarize-stability \
  features/window-fc.store \
  models/k4-seed-17.model \
  results/k4-stability.json \
  --candidate-model models/k4-seed-29.model \
  --candidate-model models/k4-seed-41.model \
  --subject sub-010 \
  --subject sub-011
```

The command requires one model family and state count, the same feature keys
and model settings apart from the seed, the same training participants, and a
common decoded cohort. Candidate seeds must be distinct. Training participants
are rejected by default; `--allow-fit-subjects` is reserved for diagnostics.

The JSON contains the seed order, candidate-to-reference permutations, cost
matrices, alignment metric, run identities, and per-fit/mean/standard-
deviation/valid-count summaries. Missing dwell or transition values are written
as JSON `null`. The numerical summary is independent of any cryptographic or
content identity mechanism.

For assignments already aligned by the caller:

```python
from dfckit.states import summarize_state_stability

runs = summarize_state_stability(
    (reference_assignments, candidate_1_aligned, candidate_2_aligned)
)
```
