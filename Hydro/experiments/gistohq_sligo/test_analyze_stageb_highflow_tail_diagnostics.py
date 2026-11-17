#!/usr/bin/env python3
"""Regression tests for analyze_stageb_highflow_tail_diagnostics.py."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import compute_stageb_matched_support_metrics as base
from analyze_stageb_highflow_tail_diagnostics import (
    _quantile,
    _sse,
    high_flow_tail_diagnostics,
)
from test_compute_stageb_matched_support_metrics import _write_job_dir, _frozen_manifest_hashes


def _add_training_rows(job_dir: Path, approach: str, train_rows: dict) -> None:
    """Append split='train' rows to an existing job_dir's predictions.csv
    (written by _write_job_dir, which only writes split='test' rows)."""
    csv_path = job_dir / "predictions.csv"
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["approach", "index", "split", "x", "observed", "predicted", "residual"]
        )
        for index, (x, observed) in enumerate(sorted(train_rows.items())):
            writer.writerow({
                "approach": approach, "index": 1000 + index, "split": "train", "x": x,
                "observed": observed, "predicted": observed, "residual": 0.0,
            })


class QuantileTest(unittest.TestCase):
    def test_known_values(self):
        values = [float(i) for i in range(1, 11)]  # 1..10
        # 90th percentile of 1..10 (linear interpolation, numpy convention) = 9.1
        self.assertAlmostEqual(_quantile(values, 0.90), 9.1, places=9)
        self.assertEqual(_quantile(values, 0.0), 1.0)
        self.assertEqual(_quantile(values, 1.0), 10.0)

    def test_single_value(self):
        self.assertEqual(_quantile([5.0], 0.90), 5.0)


class SseTest(unittest.TestCase):
    def test_partition_sums_to_total(self):
        obs = [1.0, 2.0, 3.0, 4.0]
        pred = [1.1, 1.9, 3.2, 3.8]
        total = _sse(obs, pred)
        high = _sse(obs[2:], pred[2:])
        low = _sse(obs[:2], pred[:2])
        self.assertAlmostEqual(high + low, total, places=12)


class HighFlowTailDiagnosticsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _build_fold(self, year: str, *, training_offset_by_seed=None):
        native_support = set(range(0, 100000))
        bounds = {"train_end_x": 1000, "validation_end_x": 2000, "test_end_x": 3000}
        test_rows = {
            2500: (1.0, 1.1), 2501: (2.0, 1.9), 2502: (3.0, 3.2),
            2503: (10.0, 9.5), 2504: (20.0, 18.0),  # high-flow tail
        }
        training_rows = {100 + i: float(v) for i, v in enumerate(range(1, 21))}  # 1..20
        for seed in base.SEEDS:
            offset = (training_offset_by_seed or {}).get(seed, 0.0)
            control_dir = _write_job_dir(
                self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}",
                "lstm_pinn", test_rows, bounds=bounds, seed=int(seed),
            )
            multiscale_dir = _write_job_dir(
                self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}",
                "lstm_multiscale", test_rows, bounds=bounds, seed=int(seed),
            )
            seed_training_rows = {x: v + offset for x, v in training_rows.items()}
            _add_training_rows(control_dir, "lstm_pinn", seed_training_rows)
            _add_training_rows(multiscale_dir, "lstm_multiscale", seed_training_rows)
        frozen = _frozen_manifest_hashes(self.tmp_path)
        return frozen, native_support

    def test_end_to_end_partitions_and_checks_sse(self):
        orig_seeds, orig_output = base.SEEDS, base.STAGE_B_OUTPUT
        try:
            base.SEEDS = ("42", "123", "27182")
            base.STAGE_B_OUTPUT = self.tmp_path
            frozen, native_support = self._build_fold("9997")
            result = high_flow_tail_diagnostics("9997", 30, "test", frozen, native_support)
        finally:
            base.SEEDS, base.STAGE_B_OUTPUT = orig_seeds, orig_output

        # Training observed is 1..20; Q90 (linear interp) = 18.1.
        self.assertAlmostEqual(result["q90_threshold_mm_h"], 18.1, places=9)
        # Only observed=20.0 (x=2504) is >= 18.1 among the 5 test rows.
        entry = result["per_seed"]["42"]
        self.assertEqual(entry["n_high"], 1)
        self.assertEqual(entry["n_remaining"], 4)
        self.assertTrue(entry["control"]["sse_partition_check_passed"])
        self.assertTrue(entry["multiscale"]["sse_partition_check_passed"])
        self.assertEqual(entry["control"]["high"]["n"], 1)
        self.assertEqual(entry["control"]["remaining"]["n"], 4)

    def test_differing_training_targets_across_seeds_is_rejected(self):
        """If two seeds' own training-split observed discharge genuinely
        differ (should never happen for the real Stage B data -- same raw
        discharge regardless of training seed), a single shared Q90
        threshold cannot be computed; must refuse, not silently pick one
        seed's training data arbitrarily."""
        orig_seeds, orig_output = base.SEEDS, base.STAGE_B_OUTPUT
        try:
            base.SEEDS = ("42", "123", "27182")
            base.STAGE_B_OUTPUT = self.tmp_path
            frozen, native_support = self._build_fold("9996", training_offset_by_seed={"123": 5.0})
            with self.assertRaisesRegex(base.MatchedSupportError, "training-split observed discharge differs"):
                high_flow_tail_diagnostics("9996", 30, "test", frozen, native_support)
        finally:
            base.SEEDS, base.STAGE_B_OUTPUT = orig_seeds, orig_output


if __name__ == "__main__":
    unittest.main()
