#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest import mock

from compute_shap_attribution import (
    N_CHANNELS,
    ShapAttributionError,
    apply_mask,
    build_window,
    exact_shapley_values,
    main,
    read_dumped_series,
    read_test_split_targets,
)


def _write_series_csv(path: Path, rows):
    """rows: list of (x, segment_id, f0..f5, target, target_valid)."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "segment_id", "feature0", "feature1", "feature2", "feature3",
                    "feature4", "feature5", "target", "target_valid"])
        for row in rows:
            w.writerow(row)


class ExactShapleyMathTest(unittest.TestCase):
    def test_reproduces_known_linear_attributions(self):
        """For an additive linear function v(S) = sum_{i in S} c_i * (real_i - baseline_i),
        the EXACT Shapley value of feature i is exactly c_i*(real_i-baseline_i), regardless
        of coalition order -- a textbook closed-form check independent of this program's
        own implementation."""
        coeffs = [1.0, -2.0, 0.5, 3.0, 0.0, -1.5]
        real = [10.0, 4.0, 2.0, 1.0, 5.0, 6.0]
        baseline = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        expected = [c * (r - b) for c, r, b in zip(coeffs, real, baseline)]

        def v_of(mask: int) -> float:
            total = 0.0
            for c in range(N_CHANNELS):
                value = real[c] if (mask >> c) & 1 else baseline[c]
                total += coeffs[c] * value
            return total

        v = {mask: v_of(mask) for mask in range(64)}
        phi = exact_shapley_values(v)
        for got, want in zip(phi, expected):
            self.assertAlmostEqual(got, want, places=9)

    def test_additivity_holds_for_nonlinear_function(self):
        """Even for a non-additive (nonlinear, interaction-having) value function, exact
        Shapley values must still satisfy the efficiency/additivity property exactly:
        sum(phi) == v(full) - v(empty)."""
        real = [2.0, 3.0, 1.0, 4.0, 0.5, 2.5]
        baseline = [0.0] * N_CHANNELS

        def v_of(mask: int) -> float:
            included = [real[c] if (mask >> c) & 1 else baseline[c] for c in range(N_CHANNELS)]
            return sum(included) + 0.1 * included[0] * included[1] - 0.05 * included[2] ** 2

        v = {mask: v_of(mask) for mask in range(64)}
        phi = exact_shapley_values(v)
        self.assertAlmostEqual(sum(phi), v[63] - v[0], places=9)


class WindowBuildingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_window_picks_correct_24_rows_ending_at_x(self):
        rows = [(x, 0, x, 0, 0, 0, 0, 0, 1.0, 1) for x in range(0, 30)]
        path = self.tmp_path / "series.csv"
        _write_series_csv(path, rows)
        series = read_dumped_series(path)
        window = build_window(series, end_x=29)
        # feature0 was set equal to x itself, so the window's feature0 column reveals
        # which x values were actually selected.
        got_xs = [hour[0] for hour in window]
        self.assertEqual(got_xs, list(range(6, 30)))

    def test_window_crossing_segment_boundary_rejected(self):
        rows = [(x, 0 if x < 15 else 1, 1.0, 0, 0, 0, 0, 0, 1.0, 1) for x in range(0, 40)]
        path = self.tmp_path / "series.csv"
        _write_series_csv(path, rows)
        series = read_dumped_series(path)
        with self.assertRaisesRegex(ShapAttributionError, "segment boundary"):
            build_window(series, end_x=30)

    def test_reserved_period_x_rejected(self):
        rows = [(77016, 0, 1.0, 0, 0, 0, 0, 0, 1.0, 1)]
        path = self.tmp_path / "series.csv"
        _write_series_csv(path, rows)
        with self.assertRaisesRegex(ShapAttributionError, "reserved-period seal"):
            read_dumped_series(path)

    def test_missing_hour_rejected(self):
        rows = [(x, 0, 1.0, 0, 0, 0, 0, 0, 1.0, 1) for x in range(0, 24) if x != 10]
        path = self.tmp_path / "series.csv"
        _write_series_csv(path, rows)
        series = read_dumped_series(path)
        with self.assertRaisesRegex(ShapAttributionError, "missing x"):
            build_window(series, end_x=23)


class MaskApplicationTest(unittest.TestCase):
    def test_apply_mask_selects_real_or_baseline_per_channel(self):
        real = [[float(h * 10 + c) for c in range(N_CHANNELS)] for h in range(24)]
        baseline = [[float(-1) for _ in range(N_CHANNELS)] for _ in range(24)]
        hybrid = apply_mask(real, baseline, mask=0b000101)  # channels 0 and 2 real
        for hour in range(24):
            for c in range(N_CHANNELS):
                expected = real[hour][c] if c in (0, 2) else baseline[hour][c]
                self.assertEqual(hybrid[hour][c], expected)


class ReadTestSplitTargetsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_filters_to_approach_and_test_split_below_seal(self):
        path = self.tmp_path / "predictions.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["approach", "index", "split", "x", "observed", "predicted", "residual"])
            w.writerow(["lstm", 0, "train", 1, 1.0, 1.0, 0.0])
            w.writerow(["lstm", 1, "test", 100, 2.0, 2.1, 0.1])
            w.writerow(["lstm", 2, "test", 77020, 3.0, 3.0, 0.0])  # reserved, must be excluded
            w.writerow(["ffn", 3, "test", 50, 4.0, 4.0, 0.0])  # wrong approach
        rows = read_test_split_targets(path, "lstm")
        self.assertEqual(rows, [(100, 2.0, 2.1)])


class EndToEndMainTest(unittest.TestCase):
    """Exercises main() end-to-end with a fully synthetic series/predictions and a
    stubbed CLI (no real binary, no real checkpoint) -- checks the full pipeline wiring
    (background construction, explanation sampling, mask generation, Shapley computation,
    additivity) without depending on the compiled C++ binary."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_end_to_end_with_stubbed_cli(self):
        n_rows = 60
        rows = [(x, 0, float(x), float(x) * 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1)
                for x in range(n_rows)]
        series_path = self.tmp_path / "series.csv"
        _write_series_csv(series_path, rows)

        predictions_path = self.tmp_path / "predictions.csv"
        with predictions_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["approach", "index", "split", "x", "observed", "predicted", "residual"])
            for x in range(23, n_rows):
                w.writerow(["lstm", x, "test", x, 1.0, 1.0, 0.0])

        output_path = self.tmp_path / "out.json"

        def fake_run(cmd, capture_output, text):
            input_path = Path(cmd[cmd.index("--input") + 1])
            output_csv = Path(cmd[cmd.index("--output") + 1])
            with input_path.open(newline="") as f, output_csv.open("w", newline="") as out:
                reader = csv.reader(f)
                next(reader)
                writer = csv.writer(out)
                writer.writerow(["row_id", "prediction"])
                for row in reader:
                    row_id = row[0]
                    values = [float(v) for v in row[1:]]
                    writer.writerow([row_id, sum(values)])  # trivial additive stand-in model
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("compute_shap_attribution.subprocess.run", side_effect=fake_run):
            rc = main([
                "--cli", "/bin/true", "--experiment-dir", str(self.tmp_path), "--approach", "lstm",
                "--dumped-series", str(series_path), "--predictions-csv", str(predictions_path),
                "--n-background", "10", "--n-explain", "5", "--seed", "1",
                "--output", str(output_path),
            ])
        self.assertEqual(rc, 0)
        report = json.loads(output_path.read_text())
        self.assertEqual(report["n_explained"], 5)
        self.assertLess(report["max_abs_additivity_residual"], 1e-6)


if __name__ == "__main__":
    unittest.main()
