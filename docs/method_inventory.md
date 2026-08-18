# Method inventory

This inventory defines the intended library destination for methods already
implemented in the research workflows. It does not copy project-specific
contracts into the library.

| Method family | Planned module | Initial reference implementation |
|---|---|---|
| Censor-bounded windows | `dfckit.segments` | `dfc_sliding_window.py` |
| Weighted FC | `dfckit.connectivity.correlation` | `roi32_modular_dynamic_stability.py` |
| MTD | `dfckit.connectivity.mtd` (implemented) | `audited_mtd.py` |
| ETS | `dfckit.connectivity.ets` (implemented) | `dfc_ets.py` |
| LEiDA | `dfckit.connectivity.leida` (implemented) | `traditional_leida.py` |
| Low-rank subspaces | `dfckit.connectivity.lowrank` (implemented) | `system_lowrank_coordination.py` |
| MI/CMI | `dfckit.connectivity.information` (implemented) | `fixed_length_information_hc_audit.py` |
| Window KMeans | `dfckit.states.kmeans` (implemented) | `dfc_cluster_windows.py` |
| Out-of-core KMeans | `dfckit.outofcore` (implemented) | large feature-store state fits |
| Out-of-core PCA/HMM | `dfckit.outofcore`, `dfckit.outofcore_hmm` (implemented) | large sequence-state fits |
| CAP | `dfckit.states.cap` (implemented) | `dfc_cap.py` |
| Gaussian HMM | `dfckit.states.hmm` (implemented) | `dfc_gaussian_hmm.py` |
| State metrics | `dfckit.states.metrics` (implemented) | CAP/HMM/KMeans runners |
| State alignment | `dfckit.states.alignment` (implemented) | `audit_sliding_kmeans_state_stability.py` |
| Held-out state scoring and state-count selection | `dfckit.states.scoring`, `dfckit.states.selection`, `dfckit.states.cross_validation`, `dfckit.io.state_selection`, `dfckit.outofcore`, `dfckit.outofcore_hmm` (implemented) | deterministic subject-disjoint validation, repeated-fit and participant/fold-balanced candidate comparison |
| Partition metrics | `dfckit.networks.partition` (implemented) | `roi32_modular_dynamic_stability.py` |
| Paired inference | `dfckit.inference` (implemented) | repeated audited implementations |
| Motion matching | `dfckit.qc.matching` (implemented) | `sliding_state_motion_matched_windows.py` |
| Subject-balanced reference | `dfckit.reference` (implemented) | `system_lowrank_coordination.py`, `roi32_modular_dynamic_stability.py` |
| Paired NBS | `dfckit.inference.nbs` (experimental) | `roi32_medstate_nbs_standard.py` |
| XCP-D input | `dfckit.io.xcpd` | audited ROI32/ROI40 loaders; implemented as the primary file input |

NBS remains explicitly experimental. Its tail handling, component statistic,
permutation unit, and threshold sensitivity now have independent unit tests.
