import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.artifacts import load_state_alignment, save_state_alignment
from dfckit.states import StateAlignment


def _alignment() -> StateAlignment:
    return StateAlignment(
        candidate_to_reference=np.asarray([1, 0]),
        matched_costs=np.asarray([0.05, 0.1]),
        cost_matrix=np.asarray([[1.2, 0.05], [0.1, 1.4]]),
        reference_seed=17,
        candidate_seed=29,
        feature_keys=(("visual", "motor"), ("visual", "putamen")),
        source_contract="alignment-io-test:v1",
        sample_interval_seconds=0.8,
    )


class StateAlignmentIOTests(unittest.TestCase):
    def test_roundtrip_preserves_mapping_and_contract(self):
        with TemporaryDirectory() as temporary:
            target = save_state_alignment(_alignment(), Path(temporary) / "alignment")
            restored = load_state_alignment(target)
            np.testing.assert_array_equal(restored.candidate_to_reference, [1, 0])
            np.testing.assert_allclose(restored.matched_costs, [0.05, 0.1])
            np.testing.assert_allclose(
                restored.cost_matrix,
                [[1.2, 0.05], [0.1, 1.4]],
            )
            self.assertEqual(restored.reference_seed, 17)
            self.assertEqual(restored.candidate_seed, 29)
            self.assertEqual(restored.metric, "euclidean")
            self.assertFalse(restored.cost_matrix.flags.writeable)

    def test_overwrite_and_bad_version_are_rejected(self):
        with TemporaryDirectory() as temporary:
            target = save_state_alignment(_alignment(), Path(temporary) / "alignment")
            with self.assertRaises(FileExistsError):
                save_state_alignment(_alignment(), target)
            manifest = target / "manifest.json"
            text = manifest.read_text(encoding="utf-8").replace(
                '"format_version": 2',
                '"format_version": 99',
            )
            manifest.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_state_alignment(target)


if __name__ == "__main__":
    unittest.main()
