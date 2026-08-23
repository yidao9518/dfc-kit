# Engineering plan

`dfc-kit` is a numerical library, not an experiment scheduler. The design is
organized around a short path:

```text
XCP-D derivatives
    -> censor-safe feature estimator
    -> FeatureStore or in-memory result
    -> state model / information estimator
    -> explicit score, summary, or inference result
```

## Design rules

1. Preserve original frame indices and segment identities.
2. Never calculate a temporal derivative, window, phase feature, or state
   transition across a retained-frame gap.
3. Keep fit and held-out subjects explicit in function arguments.
4. Validate shapes, feature keys, finite values, model parameters, and duplicate
   acquisition identities at the point of use.
5. Prefer one direct computation path over resumable intermediate state.
6. Store model parameters and human-readable metadata; do not add cryptographic
   identities to numerical results.
7. Keep MI/CMI in `dfckit.information`, separate from connectivity kernels.
8. Keep the agent package separate from this toolkit.

## State-count workflow

For each candidate K and subject-disjoint fold, fit on the training subjects and
score on the held-out subjects. Average repeated seeds inside each participant,
then average participants inside each fold. The validation module returns the
candidate table, best K, and one-standard-error K. A final refit is explicit and
does not require a workflow directory.

## Verification

```bash
python -m unittest discover
ruff check src tests
mkdocs build --strict
```
