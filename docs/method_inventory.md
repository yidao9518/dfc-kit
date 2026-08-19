# Method inventory

This inventory maps the method families available in `dfc-kit` to their public
API and user guide. Application-specific hypotheses, cohort definitions, and
clinical endpoints are configured by callers rather than encoded in the method
implementations.

| Method family | Public API | Guide |
|---|---|---|
| XCP-D loading and censor-bounded topology | `dfckit.io`, `dfckit.segments` | [XCP-D input](xcpd_input.md) |
| Weighted correlation and sliding-window FC | `dfckit.connectivity.correlation`, `SlidingWindowFC` | [Correlation and sliding-window FC](correlation.md) |
| MTD | `dfckit.connectivity.MTD` | [MTD](mtd.md) |
| ETS | `dfckit.connectivity.ETS` | [ETS](ets.md) |
| LEiDA and phase summaries | `dfckit.connectivity.LEiDA` | [LEiDA](leida.md) |
| Low-rank covariance geometry | `dfckit.connectivity.LowRankCovariance` | [Low-rank covariance](lowrank.md) |
| Fixed-length MI/CMI | `dfckit.connectivity.FixedLengthInformation`, `dfckit.io.information` | [Information](information.md) |
| Partition and graph metrics | `dfckit.networks` | [Partition graphs](partition_graphs.md) |
| CAP, KMeans, and Gaussian HMM states | `dfckit.states` | [State models](states.md) |
| State alignment and repeated-fit stability | `dfckit.states.alignment`, `dfckit.states.stability` | [State alignment](state_alignment.md), [stability](state_stability.md) |
| Held-out scoring and state-count selection | `dfckit.states.scoring`, `dfckit.states.selection`, `dfckit.io` | [State scoring](state_scoring.md), [selection](state_selection.md) |
| Nested participant-disjoint validation | `dfckit.io`, `dfckit.outofcore` | [Nested validation](nested_cross_validation.md) |
| Paired inference and multiple testing | `dfckit.inference` | [Paired inference](inference.md) |
| Motion summaries and within-subject matching | `dfckit.qc` | [Motion matching](qc_matching.md) |
| FeatureStores and portable model artifacts | `dfckit.storage`, `dfckit.io` | [Storage](storage.md), [model artifacts](model_artifacts.md) |
| Paired NBS | `dfckit.inference.nbs` | [NBS](nbs.md) (experimental) |

NBS remains explicitly experimental. Its tail handling, component statistic,
permutation unit, and threshold sensitivity now have independent unit tests.
