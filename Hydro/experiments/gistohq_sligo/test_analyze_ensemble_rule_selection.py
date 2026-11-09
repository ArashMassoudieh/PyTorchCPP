#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyze_ensemble_rule_selection import (
    EnsembleRuleSelectionError,
    analyze_group,
    ensemble_of_predictions_nse,
    mean_of_seed_nse,
    nse,
)


def _write_job(root, name, rows, approach="lstm_pinn"):
    """rows: list of (split, x, observed, predicted)."""
    job_dir = root / name
    job_dir.mkdir(parents=True)
    with open(job_dir / "predictions.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["approach", "index", "split", "x", "observed", "predicted", "residual"])
        for i, (split, x, observed, predicted) in enumerate(rows):
            writer.writerow([approach, i, split, x, observed, predicted, observed - predicted])
    return job_dir


class EnsembleRuleSelectionTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_mean_of_seed_and_ensemble_are_different_estimands(self):
        """Documents the estimand distinction directly: mean-of-seed-NSE and
        ensemble-of-predictions-NSE are not the same number in general."""
        rows = {
            "42": {1: (10.0, 8.0), 2: (1.0, 1.2)},
            "123": {1: (10.0, 12.0), 2: (1.0, 0.8)},
        }
        support = [1, 2]
        m = mean_of_seed_nse(rows, support)
        e = ensemble_of_predictions_nse(rows, support)
        self.assertNotAlmostEqual(m, e, places=3)

    def test_cross_seed_observed_disagreement_is_rejected(self):
        job42 = _write_job(self.tmp_path, "s42", [("validation", 1, 5.0, 5.0), ("test", 2, 1.0, 1.0)])
        job123 = _write_job(self.tmp_path, "s123", [("validation", 1, 999.0, 5.0), ("test", 2, 1.0, 1.0)])
        with self.assertRaisesRegex(EnsembleRuleSelectionError, "disagree on observed value"):
            analyze_group({"42": job42, "123": job123}, approach="lstm_pinn")

    def test_reserved_period_x_is_rejected(self):
        job = _write_job(self.tmp_path, "s42", [("validation", 77016, 1.0, 1.0)])
        from analyze_ensemble_rule_selection import read_split_rows
        with self.assertRaisesRegex(EnsembleRuleSelectionError, "reserved-period seal"):
            read_split_rows(job / "predictions.csv", "lstm_pinn", "validation")

    def test_selection_uses_validation_not_test(self):
        """Validation and test are deliberately built so the two rules disagree on which
        looks better in EACH split, and validation prefers the opposite rule from what
        test alone would prefer. The selected test value must follow the
        VALIDATION-chosen rule, not silently pick whichever rule is better on test
        (that would be test-outcome selection, exactly what's forbidden)."""
        # Validation: one seed (27182) is a wild outlier. The ensemble's per-x MEDIAN
        # suppresses that outlier almost entirely, so ensemble-of-predictions wins on
        # validation; mean-of-seed-NSE is dragged down by averaging in the bad seed's NSE.
        val_rows_42 = [("validation", 1, 10.0, 10.0), ("validation", 2, 2.0, 2.0)]
        val_rows_123 = [("validation", 1, 10.0, 9.9), ("validation", 2, 2.0, 1.9)]
        val_rows_27182 = [("validation", 1, 10.0, 0.0), ("validation", 2, 2.0, 0.0)]
        # Test: all three seeds now agree closely with each other but all three are
        # mediocre in the SAME direction, so the per-x median barely differs from any one
        # seed while each seed's own NSE is still reasonable -- here mean-of-seed-NSE is
        # the higher number, the opposite ranking from validation.
        test_rows_42 = [("test", 3, 5.0, 4.0), ("test", 4, 1.0, 0.8)]
        test_rows_123 = [("test", 3, 5.0, 4.1), ("test", 4, 1.0, 0.75)]
        test_rows_27182 = [("test", 3, 5.0, 3.9), ("test", 4, 1.0, 0.85)]

        job42 = _write_job(self.tmp_path, "s42", val_rows_42 + test_rows_42)
        job123 = _write_job(self.tmp_path, "s123", val_rows_123 + test_rows_123)
        job27182 = _write_job(self.tmp_path, "s27182", val_rows_27182 + test_rows_27182)

        result = analyze_group({"42": job42, "123": job123, "27182": job27182}, approach="lstm_pinn")
        # Validation must prefer ensemble (median suppresses the one bad seed).
        self.assertGreater(result["validation_ensemble_of_predictions_nse"],
                            result["validation_mean_of_seed_nse"])
        self.assertEqual(result["selected_rule_from_validation"], "ensemble_of_predictions")
        # The selected test value must equal the ENSEMBLE test number, matching the
        # validation-time choice -- even where that's not the higher of the two test numbers.
        self.assertAlmostEqual(
            result["test_value_under_selected_rule_EXPLORATORY"],
            result["test_ensemble_of_predictions_nse"],
        )

    def test_missing_split_rejected(self):
        job = _write_job(self.tmp_path, "s42", [("validation", 1, 1.0, 1.0)])  # no test rows
        job2 = _write_job(self.tmp_path, "s123", [("validation", 1, 1.0, 1.0)])
        with self.assertRaisesRegex(EnsembleRuleSelectionError, "No 'test' rows"):
            analyze_group({"42": job, "123": job2}, approach="lstm_pinn")


if __name__ == "__main__":
    unittest.main()
