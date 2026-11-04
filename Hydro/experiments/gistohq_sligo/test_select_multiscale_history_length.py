#!/usr/bin/env python3
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from select_multiscale_history_length import (
    TIE_BREAK_RULE,
    HistoryLengthSelectionError,
    nse,
    select_history_length,
)


def _write_job_dir(root, name, approach, rows, config=None):
    """rows: {x: (observed, predicted)}, written as split='validation'.
    Returns the job directory path. Also writes a minimal config.json and a
    fake checkpoint file so the hash-recording path has real files to hash.
    """
    job_dir = root / name
    (job_dir / "models").mkdir(parents=True)
    with open(job_dir / "predictions.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["approach", "index", "split", "x", "observed", "predicted", "residual"])
        for index, (x, (observed, predicted)) in enumerate(sorted(rows.items())):
            writer.writerow([approach, index, "validation", x, observed, predicted, observed - predicted])
    (job_dir / "experiment_config.json").write_text(json.dumps(config or {"experiment_id": name}))
    checkpoint_name = "".join(c.lower() if c.isalnum() else "_" for c in approach) + ".pt"
    (job_dir / "models" / checkpoint_name).write_bytes(f"fake-checkpoint-{name}".encode())
    return job_dir


def _sha256_text(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SelectMultiscaleHistoryLengthTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_naive_own_support_comparison_falsely_favors_the_longer_history(self):
        """Documents the bug this module exists to prevent, computed directly
        (not via select_history_length) so the contrast with the fix is explicit."""
        observed = {1: 10.0, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}
        predicted_d14 = {1: 2.0, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}
        predicted_d30 = {x: observed[x] for x in (2, 3, 4, 5, 6)}  # perfect, never saw x=1
        predicted_d60 = {x: observed[x] for x in (3, 4, 5, 6)}    # perfect, never saw x=1 or 2

        nse_d14_own = nse([observed[x] for x in predicted_d14], [predicted_d14[x] for x in predicted_d14])
        nse_d30_own = nse([observed[x] for x in predicted_d30], [predicted_d30[x] for x in predicted_d30])
        nse_d60_own = nse([observed[x] for x in predicted_d60], [predicted_d60[x] for x in predicted_d60])

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

        job_dirs = {}
        for d, predicted in ((14, predicted_d14), (30, predicted_d30), (60, predicted_d60)):
            job_dirs[d] = _write_job_dir(
                self.tmp_path, f"d{d}", "lstm_multiscale",
                {x: (observed[x], p) for x, p in predicted.items()},
                config={"experiment_id": f"d{d}", "multiscale_daily_history_days": d},
            )

        report = select_history_length(job_dirs)

        self.assertEqual(report["common_support_size"], 4)
        self.assertEqual(report["common_support_x_values"], [3.0, 4.0, 5.0, 6.0])
        self.assertEqual(report["tie_break_rule"], TIE_BREAK_RULE)
        for d in (14, 30, 60):
            self.assertAlmostEqual(report["per_candidate"][d]["common_support_nse"], 1.0)
        self.assertEqual(report["selected_d"], 14)
        self.assertEqual(report["per_candidate"][14]["total_eligible_validation_windows"], 6)
        self.assertEqual(report["per_candidate"][14]["common_support_windows"], 4)
        self.assertAlmostEqual(report["per_candidate"][14]["coverage_fraction"], 4 / 6)
        self.assertEqual(report["per_candidate"][60]["coverage_fraction"], 1.0)

        # Config/checkpoint hashes are real, verifiable, and tied to the
        # selected candidate's own on-disk files -- not fabricated strings.
        selected_dir = job_dirs[14]
        self.assertEqual(
            report["selected_config_sha256"],
            _sha256_text(selected_dir / "experiment_config.json"),
        )
        self.assertEqual(
            report["selected_checkpoint_sha256"],
            _sha256_text(selected_dir / "models" / "lstm_multiscale.pt"),
        )
        for d in (14, 30, 60):
            self.assertEqual(
                report["per_candidate"][d]["config_sha256"],
                _sha256_text(job_dirs[d] / "experiment_config.json"),
            )

    def test_common_support_restriction_can_actually_flip_the_winner(self):
        """A genuine, not merely flattened, reversal: the longer history is
        naively ahead only because of an uneven test population."""
        observed = {1: 10.0, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}
        predicted_d14 = {1: 9.5, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.1, 6: 0.9}  # near-perfect even on the hard point
        predicted_d60 = {3: 1.2, 4: 0.8, 5: 1.1, 6: 3.5}  # one real miss on its own shared ground

        job_dirs = {
            14: _write_job_dir(self.tmp_path, "d14", "lstm_multiscale",
                                {x: (observed[x], p) for x, p in predicted_d14.items()}),
            60: _write_job_dir(self.tmp_path, "d60", "lstm_multiscale",
                                {x: (observed[x], p) for x, p in predicted_d60.items()}),
        }

        report = select_history_length(job_dirs)
        self.assertEqual(report["selected_d"], 14)
        self.assertGreater(
            report["per_candidate"][14]["common_support_nse"],
            report["per_candidate"][60]["common_support_nse"],
        )

    def test_ignores_test_split_rows_entirely(self):
        """A test-split row with a wildly different value must never affect
        the selection -- only validation rows are ever read."""
        job_dir = self.tmp_path / "d14_with_test_rows"
        (job_dir / "models").mkdir(parents=True)
        with open(job_dir / "predictions.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["approach", "index", "split", "x", "observed", "predicted", "residual"])
            writer.writerow(["lstm_multiscale", 0, "validation", 3, 1.2, 1.2, 0.0])
            writer.writerow(["lstm_multiscale", 1, "validation", 4, 0.8, 0.8, 0.0])
            # A test row that would wreck the metric if it leaked in.
            writer.writerow(["lstm_multiscale", 2, "test", 999, 500.0, 0.0, 500.0])
        (job_dir / "experiment_config.json").write_text("{}")
        (job_dir / "models" / "lstm_multiscale.pt").write_bytes(b"x")
        other = _write_job_dir(self.tmp_path, "d60", "lstm_multiscale", {3: (1.2, 1.2), 4: (0.8, 0.8)})

        report = select_history_length({14: job_dir, 60: other})
        self.assertEqual(report["common_support_x_values"], [3.0, 4.0])
        self.assertAlmostEqual(report["per_candidate"][14]["common_support_nse"], 1.0)

    def test_non_monotonic_eligibility_is_rejected_not_assumed(self):
        job30 = _write_job_dir(self.tmp_path, "d30", "lstm_multiscale", {1: (1.0, 1.0), 2: (2.0, 2.0)})
        job60 = _write_job_dir(self.tmp_path, "d60", "lstm_multiscale", {1: (1.0, 1.0), 3: (3.0, 3.0)})

        with self.assertRaisesRegex(HistoryLengthSelectionError, "not monotonic"):
            select_history_length({30: job30, 60: job60})

    def test_requires_at_least_two_candidates(self):
        job = _write_job_dir(self.tmp_path, "only", "lstm_multiscale", {1: (1.0, 1.0)})
        with self.assertRaisesRegex(HistoryLengthSelectionError, "at least two"):
            select_history_length({14: job})

    def test_missing_checkpoint_file_is_rejected(self):
        job_dir = self.tmp_path / "incomplete"
        (job_dir / "models").mkdir(parents=True)
        (job_dir / "predictions.csv").write_text(
            "approach,index,split,x,observed,predicted,residual\n"
            "lstm_multiscale,0,validation,1,1.0,1.0,0.0\n"
        )
        (job_dir / "experiment_config.json").write_text("{}")
        # No checkpoint file written under models/.
        other = _write_job_dir(self.tmp_path, "d60", "lstm_multiscale", {1: (1.0, 1.0)})

        with self.assertRaisesRegex(HistoryLengthSelectionError, "missing"):
            select_history_length({14: job_dir, 60: other})


if __name__ == "__main__":
    unittest.main()
