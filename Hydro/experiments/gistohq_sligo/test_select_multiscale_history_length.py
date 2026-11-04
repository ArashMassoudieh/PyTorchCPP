#!/usr/bin/env python3
import csv
import tempfile
import unittest
from pathlib import Path

from select_multiscale_history_length import (
    HistoryLengthSelectionError,
    nse,
    select_history_length,
)


def _write_predictions_csv(path, approach, rows):
    """rows: {x: (observed, predicted)}, all written as split='validation'."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["approach", "index", "split", "x", "observed", "predicted", "residual"])
        for index, (x, (observed, predicted)) in enumerate(sorted(rows.items())):
            writer.writerow([approach, index, "validation", x, observed, predicted, observed - predicted])


class SelectMultiscaleHistoryLengthTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_naive_own_support_comparison_falsely_favors_the_longer_history(self):
        """Documents the bug this module exists to prevent, computed directly
        (not via select_history_length) so the contrast with the fix is explicit."""
        # D=14 must face one hard, high-variance spike (x=1) that D=30/D=60's
        # own eligibility masks exclude; D=14 misses it, D=30/D=60 are never
        # even tested on it.
        observed = {1: 10.0, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}
        predicted_d14 = {1: 2.0, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}
        predicted_d30 = {x: observed[x] for x in (2, 3, 4, 5, 6)}  # perfect, never saw x=1
        predicted_d60 = {x: observed[x] for x in (3, 4, 5, 6)}    # perfect, never saw x=1 or 2

        nse_d14_own = nse([observed[x] for x in predicted_d14], [predicted_d14[x] for x in predicted_d14])
        nse_d30_own = nse([observed[x] for x in predicted_d30], [predicted_d30[x] for x in predicted_d30])
        nse_d60_own = nse([observed[x] for x in predicted_d60], [predicted_d60[x] for x in predicted_d60])

        # The bug: D30/D60 score a perfect 1.0 purely by never facing the
        # hard point, while D14 -- which handled everything else perfectly
        # too -- looks far worse purely because it was judged on more ground.
        self.assertAlmostEqual(nse_d30_own, 1.0)
        self.assertAlmostEqual(nse_d60_own, 1.0)
        self.assertLess(nse_d14_own, 0.1)
        self.assertGreater(nse_d30_own, nse_d14_own)
        self.assertGreater(nse_d60_own, nse_d14_own)

    def test_common_support_restriction_reveals_no_real_difference(self):
        observed = {1: 10.0, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}
        predicted_d14 = {1: 2.0, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}
        predicted_d30 = {x: observed[x] for x in (2, 3, 4, 5, 6)}
        predicted_d60 = {x: observed[x] for x in (3, 4, 5, 6)}

        paths = {}
        for d, predicted in ((14, predicted_d14), (30, predicted_d30), (60, predicted_d60)):
            path = self.tmp_path / f"d{d}_predictions.csv"
            _write_predictions_csv(path, "lstm_multiscale", {x: (observed[x], p) for x, p in predicted.items()})
            paths[d] = path

        report = select_history_length(paths)

        # Common support must be exactly D=60's own (smallest/most
        # restrictive) eligible set: {3, 4, 5, 6}.
        self.assertEqual(report["common_support_size"], 4)
        # On that shared ground, D14 is perfect too -- the apparent D30/D60
        # advantage from the naive comparison above was entirely an
        # artifact of excluded hard examples, not real skill.
        for d in (14, 30, 60):
            self.assertAlmostEqual(report["per_candidate"][d]["common_support_nse"], 1.0)
        # Ties break toward the smaller (simpler) model, never arbitrarily.
        self.assertEqual(report["selected_d"], 14)
        # Coverage/retained counts are recorded per the handoff's requirement.
        self.assertEqual(report["per_candidate"][14]["total_eligible_validation_windows"], 6)
        self.assertEqual(report["per_candidate"][14]["common_support_windows"], 4)
        self.assertAlmostEqual(report["per_candidate"][14]["coverage_fraction"], 4 / 6)
        self.assertEqual(report["per_candidate"][60]["coverage_fraction"], 1.0)

    def test_common_support_restriction_can_actually_flip_the_winner(self):
        """A genuine, not merely flattened, reversal: the longer history is
        naively ahead only because of an uneven test population."""
        observed = {1: 10.0, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}
        predicted_d14 = {1: 9.5, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}  # near-perfect even on the hard point
        predicted_d60 = {3: 1.2, 4: 0.8, 5: 1.1, 6: 3.5}  # one real miss on its own shared ground

        path14 = self.tmp_path / "d14.csv"
        path60 = self.tmp_path / "d60.csv"
        _write_predictions_csv(path14, "lstm_multiscale", {x: (observed[x], p) for x, p in predicted_d14.items()})
        _write_predictions_csv(path60, "lstm_multiscale", {x: (observed[x], p) for x, p in predicted_d60.items()})

        report = select_history_length({14: path14, 60: path60})
        self.assertEqual(report["selected_d"], 14)
        self.assertGreater(
            report["per_candidate"][14]["common_support_nse"],
            report["per_candidate"][60]["common_support_nse"],
        )

    def test_non_monotonic_eligibility_is_rejected_not_assumed(self):
        # D=30 (smaller D, should be a superset of D=60) is missing a window
        # that D=60 (larger D) has -- eligibility is not actually monotonic
        # here, so the module must refuse rather than silently proceed.
        path30 = self.tmp_path / "d30.csv"
        path60 = self.tmp_path / "d60.csv"
        _write_predictions_csv(path30, "lstm_multiscale", {1: (1.0, 1.0), 2: (2.0, 2.0)})
        _write_predictions_csv(path60, "lstm_multiscale", {1: (1.0, 1.0), 3: (3.0, 3.0)})

        with self.assertRaisesRegex(HistoryLengthSelectionError, "not monotonic"):
            select_history_length({30: path30, 60: path60})

    def test_requires_at_least_two_candidates(self):
        path = self.tmp_path / "only.csv"
        _write_predictions_csv(path, "lstm_multiscale", {1: (1.0, 1.0)})
        with self.assertRaisesRegex(HistoryLengthSelectionError, "at least two"):
            select_history_length({14: path})


if __name__ == "__main__":
    unittest.main()
