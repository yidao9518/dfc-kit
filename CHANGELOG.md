# Changelog

## 1.0.0 - 2026-08-24

- Moved MI/CMI estimators and fixed-window orchestration into
  `dfckit.information`.
- Unified random sampling and frozen schedules through one fixed-window
  estimation path.
- Simplified state-count selection to subject-disjoint held-out score
  comparison with compact JSON output.
- Removed cryptographic data/model identities, workflow-directory loaders,
  nested checkpointing, and legacy compatibility layers from the development
  codebase.
- Kept model, prediction, alignment, and information artifacts pickle-free and
  validated by explicit shapes, parameters, feature keys, and participant
  metadata.
- ETS and MTD share the same segment-safe instantaneous-edge framework while
  retaining their distinct sample generators.
- Paired NBS, HC3 inference, information estimates, and all censor-gap-safe
  connectivity methods retain their numerical APIs.
- FeatureStore summaries support streaming acquisition-level mean, population
  variance, standard deviation, minimum, and maximum endpoints.
