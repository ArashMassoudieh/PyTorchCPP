#!/usr/bin/env python3
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reverify_annual_metrics import ReverifyError, main, nse, read_job_test_rows


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_job(root: Path, job_id: str, rows, test_start_x: float):
    """rows: list of (split, x, observed, predicted). Writes predictions.csv plus a
    VERIFIED.json recording its real hash, matching the on-disk layout
    runs/<id>/<id>/predictions.csv used by the real annual batch."""
    job_root = root / job_id
    inner = job_root / job_id
    inner.mkdir(parents=True)
    lines = ["approach,index,split,x,observed,predicted,residual"]
    for i, (split, x, o, p) in enumerate(rows):
        lines.append(f"ffn,{i},{split},{x},{o},{p},{o-p}")
    text = "\n".join(lines) + "\n"
    (inner / "predictions.csv").write_text(text)
    verified = {"id": job_id, "predictions_sha256": _sha256_text(text), "config_sha256": "x"}
    (job_root / "VERIFIED.json").write_text(json.dumps(verified))
    return test_start_x


class ReverifyAnnualMetricsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reproduces_hand_computed_nse(self):
        rows = {
            "42": [("test", 1, 10.0, 8.0), ("test", 2, 2.0, 2.5), ("test", 3, 4.0, 4.0)],
        }
        runs = self.tmp_path / "runs"
        _write_job(runs, "j1", rows["42"], test_start_x=0)
        manifest = {"jobs": [{"id": "j1", "year": "2019", "model": "ffn", "seed": 42, "test_start_x": 0}]}
        manifest_path = self.tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        native_path = self.tmp_path / "native.json"
        native_path.write_text(json.dumps({
            "strict_observed_hour_x": [1, 2, 3],
            "boundary_interpolation_checks": [],
        }))
        compare = self.tmp_path / "annual_metrics.csv"
        observed = [10.0, 2.0, 4.0]
        predicted = [8.0, 2.5, 4.0]
        expected_nse = nse(observed, predicted)
        with compare.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["year", "model", "seed", "nse"])
            w.writeheader()
            w.writerow({"year": "2019", "model": "ffn", "seed": "42", "nse": expected_nse})
        output = self.tmp_path / "out.json"
        rc = main([
            "--manifest", str(manifest_path), "--runs", str(runs),
            "--native-support", str(native_path), "--compare-against", str(compare),
            "--output", str(output),
        ])
        self.assertEqual(rc, 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["n_compared_against_annual_metrics_csv"], 1)

    def test_detects_real_mismatch(self):
        rows = [("test", 1, 10.0, 8.0), ("test", 2, 2.0, 2.5), ("test", 3, 4.0, 4.0)]
        runs = self.tmp_path / "runs"
        _write_job(runs, "j1", rows, test_start_x=0)
        manifest = {"jobs": [{"id": "j1", "year": "2019", "model": "ffn", "seed": 42, "test_start_x": 0}]}
        manifest_path = self.tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        native_path = self.tmp_path / "native.json"
        native_path.write_text(json.dumps({
            "strict_observed_hour_x": [1, 2, 3],
            "boundary_interpolation_checks": [],
        }))
        compare = self.tmp_path / "annual_metrics.csv"
        with compare.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["year", "model", "seed", "nse"])
            w.writeheader()
            w.writerow({"year": "2019", "model": "ffn", "seed": "42", "nse": 0.999})  # deliberately wrong
        output = self.tmp_path / "out.json"
        main([
            "--manifest", str(manifest_path), "--runs", str(runs),
            "--native-support", str(native_path), "--compare-against", str(compare),
            "--output", str(output),
        ])
        report = json.loads(output.read_text())
        self.assertEqual(len(report["mismatches"]), 1)
        self.assertIn("0.999", report["mismatches"][0])

    def test_hash_mismatch_rejected(self):
        rows = [("test", 1, 1.0, 1.0), ("test", 2, 2.0, 2.0)]
        runs = self.tmp_path / "runs"
        _write_job(runs, "j1", rows, test_start_x=0)
        # Corrupt the recorded hash so it no longer matches the file on disk.
        verified_path = runs / "j1" / "VERIFIED.json"
        verified = json.loads(verified_path.read_text())
        verified["predictions_sha256"] = "0" * 64
        verified_path.write_text(json.dumps(verified))
        manifest = {"jobs": [{"id": "j1", "year": "2019", "model": "ffn", "seed": 42, "test_start_x": 0}]}
        manifest_path = self.tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        native_path = self.tmp_path / "native.json"
        native_path.write_text(json.dumps({"strict_observed_hour_x": [1, 2], "boundary_interpolation_checks": []}))
        compare = self.tmp_path / "annual_metrics.csv"
        with compare.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["year", "model", "seed", "nse"])
            w.writeheader()
        with self.assertRaisesRegex(ReverifyError, "hash mismatch"):
            main([
                "--manifest", str(manifest_path), "--runs", str(runs),
                "--native-support", str(native_path), "--compare-against", str(compare),
                "--output", str(self.tmp_path / "out2.json"),
            ])

    def test_reserved_period_x_rejected(self):
        runs = self.tmp_path / "runs"
        job_root = runs / "j1"
        inner = job_root / "j1"
        inner.mkdir(parents=True)
        text = "approach,index,split,x,observed,predicted,residual\nffn,0,test,77016,1.0,1.0,0.0\n"
        (inner / "predictions.csv").write_text(text)
        (job_root / "VERIFIED.json").write_text(
            json.dumps({"predictions_sha256": _sha256_text(text)})
        )
        with self.assertRaisesRegex(ReverifyError, "reserved-period seal"):
            read_job_test_rows(inner / "predictions.csv", test_start_x=0)

    def test_observation_disagreement_rejected(self):
        runs = self.tmp_path / "runs"
        _write_job(runs, "j1", [("test", 1, 5.0, 5.0), ("test", 2, 7.0, 7.0)], test_start_x=0)
        _write_job(runs, "j2", [("test", 1, 999.0, 5.0), ("test", 2, 7.0, 7.0)], test_start_x=0)
        manifest = {"jobs": [
            {"id": "j1", "year": "2019", "model": "ffn", "seed": 42, "test_start_x": 0},
            {"id": "j2", "year": "2019", "model": "ffn", "seed": 123, "test_start_x": 0},
        ]}
        manifest_path = self.tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        native_path = self.tmp_path / "native.json"
        native_path.write_text(json.dumps({"strict_observed_hour_x": [1, 2], "boundary_interpolation_checks": []}))
        compare = self.tmp_path / "annual_metrics.csv"
        with compare.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["year", "model", "seed", "nse"])
            w.writeheader()
        with self.assertRaisesRegex(ReverifyError, "Observation disagreement"):
            main([
                "--manifest", str(manifest_path), "--runs", str(runs),
                "--native-support", str(native_path), "--compare-against", str(compare),
                "--output", str(self.tmp_path / "out3.json"),
            ])


if __name__ == "__main__":
    unittest.main()
