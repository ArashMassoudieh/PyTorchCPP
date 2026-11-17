#!/usr/bin/env python3
"""Regression tests for analyze_stageb_event_diagnostics.py."""
import tempfile
import unittest
from pathlib import Path

import compute_stageb_matched_support_metrics as base
from analyze_stageb_event_diagnostics import (
    event_arm_metrics,
    event_cores,
    event_diagnostics,
    expanded_events,
)
from test_analyze_stageb_highflow_tail_diagnostics import _add_training_rows
from test_compute_stageb_matched_support_metrics import _write_job_dir, _frozen_manifest_hashes


class EventCoresTest(unittest.TestCase):
    def test_merges_nearby_peaks_keeps_distant_ones_separate(self):
        observed = {10: 5.0, 11: 6.0, 12: 5.5, 100: 7.0, 101: 6.5}
        cores = event_cores(observed, threshold=5.0, merge_gap=24)
        self.assertEqual(cores, [(10, 12), (100, 101)])

    def test_no_values_above_threshold_returns_empty(self):
        self.assertEqual(event_cores({10: 1.0, 20: 2.0}, threshold=5.0, merge_gap=24), [])


class ExpandedEventsTest(unittest.TestCase):
    def test_padding_merges_overlapping_windows(self):
        cores = [(100, 100), (110, 110)]
        windows = expanded_events(cores, before=5, after=20)
        self.assertEqual(windows, [(95, 130)])

    def test_non_overlapping_windows_stay_separate(self):
        cores = [(100, 100), (500, 500)]
        windows = expanded_events(cores, before=5, after=20)
        self.assertEqual(windows, [(95, 120), (495, 520)])


class EventArmMetricsTest(unittest.TestCase):
    def test_peak_matched_within_radius_not_window_wide_max(self):
        """A larger value further away in x must NOT be picked as the
        'predicted peak' if it falls outside peak_match_radius_hours of the
        observed peak's position -- otherwise an unrelated later pulse
        could be mistaken for a timing/magnitude match."""
        xs = [0, 1, 2, 50]
        obs = [1.0, 10.0, 2.0, 1.0]  # observed peak at x=1
        pred = [1.0, 9.0, 2.0, 100.0]  # huge value at x=50 must be ignored (too far)
        result = event_arm_metrics(xs, obs, pred, peak_match_radius_hours=12)
        self.assertEqual(result["predicted_peak"], 9.0)
        self.assertEqual(result["peak_timing_error_hours"], 0)

    def test_missed_peak_flagged_with_disclosed_percentage(self):
        xs = [0, 1, 2]
        obs = [1.0, 10.0, 1.0]
        pred = [1.0, 2.0, 1.0]  # predicted peak is 80% below observed
        result = event_arm_metrics(xs, obs, pred, peak_match_radius_hours=12)
        self.assertTrue(result["missed_peak"])
        self.assertAlmostEqual(result["peak_magnitude_error_percent"], -80.0, places=9)

    def test_not_missed_when_within_threshold(self):
        xs = [0, 1, 2]
        obs = [1.0, 10.0, 1.0]
        pred = [1.0, 6.0, 1.0]  # 40% below -- not missed (threshold is -50%)
        result = event_arm_metrics(xs, obs, pred, peak_match_radius_hours=12)
        self.assertFalse(result["missed_peak"])


class EventDiagnosticsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_identical_windows_applied_to_both_arms_control_misses_multiscale_catches(self):
        """Builds one synthetic event where control badly misses the peak
        but multiscale catches it, and confirms: (a) the SAME window is
        used to score both arms (built from observed data only); (b)
        control's miss is correctly flagged; (c) multiscale's catch is
        correctly not flagged."""
        year = "9995"
        native_support = set(range(0, 100000))
        bounds = {"train_end_x": 1000, "validation_end_x": 2000, "test_end_x": 3000}
        test_rows_control = {
            2500: (1.0, 1.1), 2501: (2.0, 1.9), 2502: (20.0, 3.0),  # control misses the peak badly
            2503: (3.0, 2.9), 2504: (1.0, 1.0),
        }
        test_rows_multiscale = {
            2500: (1.0, 1.1), 2501: (2.0, 1.9), 2502: (20.0, 19.0),  # multiscale catches it
            2503: (3.0, 2.9), 2504: (1.0, 1.0),
        }
        training_rows = {100 + i: float(v) for i, v in enumerate(range(1, 21))}  # 1..20, Q90=18.1
        orig_seeds, orig_output = base.SEEDS, base.STAGE_B_OUTPUT
        try:
            base.SEEDS = ("42", "123", "27182")
            base.STAGE_B_OUTPUT = self.tmp_path
            for seed in base.SEEDS:
                control_dir = _write_job_dir(
                    self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}",
                    "lstm_pinn", test_rows_control, bounds=bounds, seed=int(seed),
                )
                multiscale_dir = _write_job_dir(
                    self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}",
                    "lstm_multiscale", test_rows_multiscale, bounds=bounds, seed=int(seed),
                )
                _add_training_rows(control_dir, "lstm_pinn", training_rows)
                _add_training_rows(multiscale_dir, "lstm_multiscale", training_rows)
            frozen = _frozen_manifest_hashes(self.tmp_path)
            result = event_diagnostics(year, 30, "test", frozen, native_support)
        finally:
            base.SEEDS, base.STAGE_B_OUTPUT = orig_seeds, orig_output

        self.assertEqual(result["n_events"], 1)
        event = result["events"][0]
        # Same window applied to both arms -- there is only one window definition per event.
        self.assertIn("window_start_x", event)
        self.assertIn("window_end_x", event)
        entry = event["per_seed"]["42"]
        self.assertTrue(entry["control"]["missed_peak"])
        self.assertFalse(entry["multiscale"]["missed_peak"])
        self.assertEqual(entry["control"]["observed_peak"], 20.0)
        self.assertEqual(entry["multiscale"]["observed_peak"], 20.0)

    def test_matched_x_mismatch_across_seeds_is_rejected(self):
        year = "9994"
        native_support = set(range(0, 100000))
        bounds = {"train_end_x": 1000, "validation_end_x": 2000, "test_end_x": 3000}
        training_rows = {100 + i: float(v) for i, v in enumerate(range(1, 21))}
        orig_seeds, orig_output = base.SEEDS, base.STAGE_B_OUTPUT
        try:
            base.SEEDS = ("42", "123")
            base.STAGE_B_OUTPUT = self.tmp_path
            rows_by_seed = {
                "42": {2500: (5.0, 5.1), 2501: (6.0, 5.9)},
                "123": {2501: (6.0, 6.0)},
            }
            for seed, rows in rows_by_seed.items():
                control_dir = _write_job_dir(
                    self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}",
                    "lstm_pinn", rows, bounds=bounds, seed=int(seed),
                )
                multiscale_dir = _write_job_dir(
                    self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}",
                    "lstm_multiscale", rows, bounds=bounds, seed=int(seed),
                )
                _add_training_rows(control_dir, "lstm_pinn", training_rows)
                _add_training_rows(multiscale_dir, "lstm_multiscale", training_rows)
            frozen = _frozen_manifest_hashes(self.tmp_path)
            with self.assertRaisesRegex(base.MatchedSupportError, "matched_x differs between seed"):
                event_diagnostics(year, 30, "test", frozen, native_support)
        finally:
            base.SEEDS, base.STAGE_B_OUTPUT = orig_seeds, orig_output


if __name__ == "__main__":
    unittest.main()
