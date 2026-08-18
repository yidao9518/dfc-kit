import unittest
from dataclasses import replace

from dfckit.states import SubjectValidationFold, make_subject_validation_folds


class SubjectValidationFoldTests(unittest.TestCase):
    def test_hash_split_is_balanced_complete_and_order_independent(self):
        subjects = tuple(f"sub-{index:03d}" for index in range(11))
        folds = make_subject_validation_folds(subjects, n_folds=4, seed=17)
        reordered = make_subject_validation_folds(
            tuple(reversed(subjects)),
            n_folds=4,
            seed=17,
        )

        self.assertEqual([len(fold.evaluation_subjects) for fold in folds], [3, 3, 3, 2])
        self.assertEqual(
            {frozenset(fold.evaluation_subjects) for fold in folds},
            {frozenset(fold.evaluation_subjects) for fold in reordered},
        )
        self.assertEqual(
            {subject for fold in folds for subject in fold.evaluation_subjects},
            set(subjects),
        )
        for fold in folds:
            self.assertFalse(set(fold.fit_subjects).intersection(fold.evaluation_subjects))
            self.assertEqual(set(fold.fit_subjects).union(fold.evaluation_subjects), set(subjects))

    def test_seed_changes_assignment_and_repeated_calls_match(self):
        subjects = tuple(f"sub-{index:03d}" for index in range(12))
        first = make_subject_validation_folds(subjects, n_folds=3, seed=17)
        repeated = make_subject_validation_folds(subjects, n_folds=3, seed=17)
        changed = make_subject_validation_folds(subjects, n_folds=3, seed=29)

        self.assertEqual(first, repeated)
        self.assertNotEqual(
            tuple(fold.evaluation_subjects for fold in first),
            tuple(fold.evaluation_subjects for fold in changed),
        )

    def test_invalid_cohort_fold_count_seed_and_object_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            make_subject_validation_folds(("sub-001", "sub-001"), n_folds=2, seed=1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            make_subject_validation_folds(("sub-001", "sub-002"), n_folds=1, seed=1)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            make_subject_validation_folds(("sub-001", "sub-002"), n_folds=3, seed=1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_subject_validation_folds(("sub-001", "sub-002"), n_folds=2, seed=-1)

        fold = SubjectValidationFold(0, ("sub-001",), ("sub-002",))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            replace(fold, evaluation_subjects=("sub-001",))


if __name__ == "__main__":
    unittest.main()
