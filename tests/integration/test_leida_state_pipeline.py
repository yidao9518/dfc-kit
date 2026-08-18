import importlib.util
import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import LEiDA
from dfckit.states import fit_kmeans_states, leida_sequences, summarize_state_assignments

OPTIONAL_DEPENDENCIES = (
    importlib.util.find_spec("scipy") is not None
    and importlib.util.find_spec("sklearn") is not None
)


@unittest.skipUnless(OPTIONAL_DEPENDENCIES, "requires dfc-kit[phase,states]")
class LEiDAStatePipelineTests(unittest.TestCase):
    def test_gap_safe_leida_to_state_metrics(self):
        rng = np.random.default_rng(55)
        runs = []
        for subject in ("sub-001", "sub-002"):
            latent = rng.normal(size=(50, 2))
            loadings = rng.normal(size=(2, 6))
            values = latent @ loadings + 0.1 * rng.normal(size=(50, 6))
            runs.append(
                TimeSeriesRun(
                    values=values,
                    original_indices=np.r_[np.arange(25), np.arange(30, 55)],
                    roi_names=tuple(f"roi-{index}" for index in range(6)),
                    subject=subject,
                    session="off",
                    tr=0.8,
                )
            )

        sequences = leida_sequences(
            [LEiDA(minimum_segment_length=20).transform(run) for run in runs]
        )
        fitted = fit_kmeans_states(
            sequences,
            n_states=3,
            seed=55,
            n_init=10,
            algorithm="minibatch",
            standardize_features=False,
        )
        metrics = summarize_state_assignments(fitted.assignments)

        self.assertEqual(fitted.model.fit_subjects, ("sub-001", "sub-002"))
        self.assertFalse(fitted.model.standardize_features)
        self.assertEqual(len(fitted.assignments.sequences), 4)
        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(metric.n_sequences == 2 for metric in metrics))


if __name__ == "__main__":
    unittest.main()
