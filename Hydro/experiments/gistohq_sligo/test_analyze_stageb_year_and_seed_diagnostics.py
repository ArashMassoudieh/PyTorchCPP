#!/usr/bin/env python3
"""Regression tests for analyze_stageb_year_and_seed_diagnostics.py."""
import json
import tempfile
import unittest
from pathlib import Path

import compute_stageb_matched_support_metrics as base
from analyze_stageb_year_and_seed_diagnostics import (
    _safe_metrics,
    per_year_diagnostics,
    x_to_year,
)
from test_compute_stageb_matched_support_metrics import _write_job_dir, _frozen_manifest_hashes

# Hours since 2015-01-01T00:00:00Z. 2017-12-31T23:00Z = 26303; 2018-01-01T00:00Z = 26304;
# 2018-12-31T23:00Z = 35063 (matches the real 2018 fold's actual test_end_x, independently
# confirmed against dataset_manifest.json's study_start/study_end -- see
# compute_stageb_matched_support_metrics.STUDY_START_UTC's docstring).
_DEC_31_2017_23 = 26303
_JAN_1_2018_00 = 26304
_DEC_31_2018_23 = 35063


class XToYearTest(unittest.TestCase):
    def test_known_boundaries(self):
        self.assertEqual(x_to_year(0), 2015)
        self.assertEqual(x_to_year(_DEC_31_2017_23), 2017)
        self.assertEqual(x_to_year(_JAN_1_2018_00), 2018)
        self.assertEqual(x_to_year(_DEC_31_2018_23), 2018)

    def test_matches_real_fold_test_end_x(self):
        """The real 2018 fold's test_end_x=35063 (from dataset_manifest.json via
        get_native_matched_rows) must land exactly on 2018-12-31T23:00Z -- this
        is the same cross-check performed manually before writing this module,
        now pinned as a regression test."""
        self.assertEqual(x_to_year(35063), 2018)
        self.assertEqual(x_to_year(35063 + 1), 2019)


class SafeMetricsTest(unittest.TestCase):
    def test_normal_case_matches_all_metrics(self):
        obs = [5.0, 6.0, 7.0]
        pred = [5.1, 5.9, 7.2]
        result = _safe_metrics(obs, pred)
        self.assertFalse(result["undefined"])
        self.assertIn("nse", result)

    def test_zero_variance_observed_is_disclosed_not_raised(self):
        """A degenerate per-year bin (e.g. constant observed discharge) would
        make NSE's denominator zero -- all_metrics() raises for this; the
        per-year report must disclose it as undefined instead of crashing
        the whole diagnostic run over one bad year."""
        obs = [5.0, 5.0, 5.0]
        pred = [5.1, 4.9, 5.2]
        result = _safe_metrics(obs, pred)
        self.assertTrue(result["undefined"])
        self.assertIn("reason", result)
        self.assertEqual(result["n"], 3)


class PerYearDiagnosticsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_matched_x_mismatch_across_seeds_is_rejected(self):
        """If the matched-x set somehow differs between two seeds of the same
        fold (bounds/native-support/eligibility are supposed to be
        seed-independent), per_year_diagnostics must refuse rather than bin
        each seed on a different, non-comparable set of timestamps."""
        year = "9999"
        native_support = set(range(0, 100000))
        # seed 42 and seed 123 get genuinely different row sets (seed 123's
        # control/multiscale share only x=1600, not x=1500 too).
        rows_by_seed = {
            "42": {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
            "123": {1600: (6.0, 6.0)},
            "27182": {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
        }
        orig_seeds = base.SEEDS
        try:
            base.SEEDS = ("42", "123", "27182")
            for seed, rows in rows_by_seed.items():
                _write_job_dir(
                    self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}",
                    "lstm_pinn", rows, bounds={"train_end_x": 0, "validation_end_x": 1000, "test_end_x": 2000},
                    seed=int(seed),
                )
                _write_job_dir(
                    self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}",
                    "lstm_multiscale", rows, bounds={"train_end_x": 0, "validation_end_x": 1000, "test_end_x": 2000},
                    seed=int(seed),
                )
            orig_output = base.STAGE_B_OUTPUT
            base.STAGE_B_OUTPUT = self.tmp_path
            try:
                frozen = _frozen_manifest_hashes(self.tmp_path)
                with self.assertRaisesRegex(base.MatchedSupportError, "matched_x differs between seed"):
                    per_year_diagnostics(year, 30, "test", frozen, native_support)
            finally:
                base.STAGE_B_OUTPUT = orig_output
        finally:
            base.SEEDS = orig_seeds

    def test_end_to_end_synthetic_fold_produces_per_year_and_cross_seed_stats(self):
        """Builds a synthetic fold spanning the 2017/2018 calendar-year
        boundary (x=26303 is 2017-12-31T23:00Z, x=26304 is 2018-01-01T00:00Z)
        with identical matched-x across all 3 seeds, and confirms the report
        splits rows into the correct calendar years and computes cross-seed
        stability stats."""
        year = "9998"
        native_support = set(range(0, 100000))
        rows = {
            26303: (5.0, 5.1),  # 2017
            26304: (6.0, 5.8),  # 2018
            26305: (7.0, 6.9),  # 2018
        }
        bounds = {"train_end_x": 0, "validation_end_x": 1000, "test_end_x": 30000}
        orig_seeds = base.SEEDS
        orig_output = base.STAGE_B_OUTPUT
        try:
            base.SEEDS = ("42", "123", "27182")
            base.STAGE_B_OUTPUT = self.tmp_path
            for seed in base.SEEDS:
                # Slightly different predictions per seed so cross-seed stdev is nonzero.
                offset = {"42": 0.0, "123": 0.05, "27182": -0.05}[seed]
                seed_rows = {x: (o, p + offset) for x, (o, p) in rows.items()}
                _write_job_dir(
                    self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}",
                    "lstm_pinn", seed_rows, bounds=bounds, seed=int(seed),
                )
                _write_job_dir(
                    self.tmp_path, f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}",
                    "lstm_multiscale", seed_rows, bounds=bounds, seed=int(seed),
                )
            frozen = _frozen_manifest_hashes(self.tmp_path)
            result = per_year_diagnostics(year, 30, "test", frozen, native_support)
        finally:
            base.SEEDS = orig_seeds
            base.STAGE_B_OUTPUT = orig_output

        self.assertEqual(result["n_matched_total"], 3)
        self.assertEqual(sorted(result["calendar_years_present"]), ["2017", "2018"])
        self.assertEqual(result["per_year"]["2017"]["42"]["n_matched"], 1)
        self.assertEqual(result["per_year"]["2018"]["42"]["n_matched"], 2)
        # 1-row bins have zero NSE/KGE denominator -- must be disclosed as undefined.
        self.assertTrue(result["per_year"]["2017"]["42"]["control"]["undefined"])
        self.assertIn("control", result["cross_seed_stability"])
        self.assertIn("nse", result["cross_seed_stability"]["control"])
        self.assertEqual(len(result["cross_seed_stability"]["control"]["nse"]["per_seed"]), 3)
        self.assertIn("delta_multiscale_minus_control", result["cross_seed_stability"])


if __name__ == "__main__":
    unittest.main()
