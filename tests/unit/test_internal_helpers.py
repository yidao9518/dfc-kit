import unittest

import numpy as np

from dfckit._arrays import readonly_copy
from dfckit._validation import (
    validated_integer,
    validated_nonnegative_integer,
    validated_positive_integer,
    validated_roi_indices,
    validated_seed,
    validated_subject_ids,
    validated_subject_labels,
)


class InternalArrayHelperTests(unittest.TestCase):
    def test_readonly_copy_owns_data_and_rejects_mutation(self):
        source = np.asarray([1.0, 2.0])
        frozen = readonly_copy(source)

        source[0] = 9.0
        np.testing.assert_array_equal(frozen, [1.0, 2.0])
        self.assertFalse(frozen.flags.writeable)
        with self.assertRaises(ValueError):
            frozen[0] = 3.0


class InternalValidationHelperTests(unittest.TestCase):
    def test_integer_validation_accepts_numpy_integers_but_not_booleans(self):
        self.assertEqual(validated_integer(np.int64(3), label="count", minimum=1), 3)
        with self.assertRaisesRegex(TypeError, "count must be an integer"):
            validated_integer(True, label="count")
        with self.assertRaisesRegex(ValueError, "at least 2"):
            validated_integer(1, label="count", minimum=2)

    def test_seed_and_subject_validation_share_boundary_rules(self):
        self.assertEqual(validated_seed(np.int32(0), label="seed"), 0)
        with self.assertRaisesRegex(ValueError, "at least 0"):
            validated_seed(-1, label="seed")
        self.assertEqual(
            validated_subject_ids(["sub-001", "sub-002"], 2),
            ("sub-001", "sub-002"),
        )

    def test_legacy_integer_message_contracts_are_shared(self):
        self.assertEqual(validated_positive_integer(np.int64(2), "count"), 2)
        self.assertEqual(validated_nonnegative_integer(np.int32(0), "seed"), 0)
        with self.assertRaisesRegex(ValueError, "count must be positive"):
            validated_positive_integer(0, "count")
        with self.assertRaisesRegex(ValueError, "seed must be non-negative"):
            validated_nonnegative_integer(-1, "seed")

    def test_roi_indices_are_unique_integer_positions_within_bounds(self):
        np.testing.assert_array_equal(
            validated_roi_indices([0, np.int64(2)], n_rois=3, label="nodes"),
            [0, 2],
        )
        with self.assertRaisesRegex(TypeError, "integer ROI indices"):
            validated_roi_indices([False], n_rois=3, label="nodes")
        with self.assertRaisesRegex(ValueError, "duplicate ROI indices"):
            validated_roi_indices([1, 1], n_rois=3, label="nodes")
        with self.assertRaisesRegex(ValueError, r"outside \[0, 3\)"):
            validated_roi_indices([3], n_rois=3, label="nodes")

    def test_subject_labels_allow_repeated_observations(self):
        self.assertEqual(
            validated_subject_labels(["sub-001", "sub-001"], n_observations=2),
            ("sub-001", "sub-001"),
        )
        with self.assertRaisesRegex(ValueError, "one identifier per observation"):
            validated_subject_labels(["sub-001"], n_observations=2)


if __name__ == "__main__":
    unittest.main()
