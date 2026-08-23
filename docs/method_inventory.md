# Method inventory

This inventory maps the method families available in `dfc-kit` to their public
API and user guide. Application-specific hypotheses, cohort definitions, and
clinical endpoints are configured by callers rather than encoded in the method
implementations.

| Method family | Public API | Guide |
|---|---|---|
| XCP-D loading and censor-bounded topology | `dfckit.io` | [XCP-D input](xcpd_input.md) |
| Weighted correlation and sliding-window FC | `dfckit.connectivity` | [Correlation and sliding-window FC](correlation.md) |
| Instantaneous edges | `dfckit.connectivity.ETS`, `dfckit.connectivity.MTD` | [ETS and MTD](instantaneous_edges.md) |
| LEiDA and phase summaries | `dfckit.connectivity.LEiDA` | [LEiDA](leida.md) |
| Low-rank covariance geometry | `dfckit.connectivity.LowRankCovariance` | [Low-rank covariance](lowrank.md) |
| Fixed-length MI/CMI | `dfckit.information` | [Information](information.md) |
| Partition and graph metrics | `dfckit.connectivity` | [Partition graphs](partition_graphs.md) |
| CAP, KMeans, and Gaussian HMM states | `dfckit.states` | [State models](states.md) |
| State alignment and repeated-fit stability | `dfckit.states` | [State alignment](state_alignment.md), [stability](state_stability.md) |
| Held-out scoring and state-count selection | `dfckit.states`, `dfckit.artifacts` | [State scoring](state_scoring.md), [selection](state_selection.md) |
| Paired inference and multiple testing | `dfckit.inference` | [Paired inference](inference.md) |
| Covariate summaries and within-subject matching | `dfckit.inference` | [Covariate matching](inference_matching.md) |
| FeatureStores and portable model artifacts | `dfckit.storage`, `dfckit.artifacts` | [Storage](storage.md), [model artifacts](model_artifacts.md) |
| Paired NBS | `dfckit.inference` | [NBS](nbs.md) |

NBS tail handling, component statistics, permutation units, and threshold
sensitivity are explicit parts of the result contract.
