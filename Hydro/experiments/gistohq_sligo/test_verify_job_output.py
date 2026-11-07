#!/usr/bin/env python3
import csv
import io
import zipfile
import json
import tempfile
import unittest
from pathlib import Path

from verify_job_output import JobOutputVerificationError, verify_job_output

def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('archive/data.pkl', b'not-unpickled-by-this-verifier')
        z.writestr('archive/version', b'3\n')
        z.writestr('archive/data/0', b'\0' * 2048)
    return buf.getvalue()

_REAL_ZIP_HEADER = _zip_bytes()  # real CRC-valid ZIP; NOT semantic model evidence


def _write_predictions(path, rows):
    """rows: list of (approach, split, x, observed, predicted)."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["approach", "index", "split", "x", "observed", "predicted", "residual"])
        for i, (approach, split, x, observed, predicted) in enumerate(rows):
            writer.writerow([approach, i, split, x, observed, predicted, predicted - observed])


def _make_valid_job(root, name="job", approach="lstm_multiscale", d=14,
                     rows=None, checkpoint_bytes=_REAL_ZIP_HEADER, config_overrides=None):
    job_dir = root / name
    (job_dir / "models").mkdir(parents=True)
    if rows is None:
        rows = [
            (approach, "validation", 1, 1.0, 1.1),
            (approach, "validation", 2, 2.0, 2.1),
            (approach, "test", 3, 3.0, 3.1),
        ]
    _write_predictions(job_dir / "predictions.csv", rows)
    config = {"multiscale_daily_history_days": d, "train_end_x": 0, "validation_end_x": 2, "test_end_x": 3}
    if config_overrides:
        config.update(config_overrides)
    (job_dir / "experiment_config.json").write_text(json.dumps(config))
    checkpoint_name = "".join(c.lower() if c.isalnum() else "_" for c in approach) + ".pt"
    (job_dir / "models" / checkpoint_name).write_bytes(checkpoint_bytes)
    return job_dir


class VerifyJobOutputTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_valid_job_passes_and_reports_honestly(self):
        job_dir = _make_valid_job(self.tmp_path)
        report = verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)
        self.assertEqual(report["total_rows_for_approach"], 3)
        self.assertEqual(report["rows_by_split"], {"validation": 2, "test": 1})
        self.assertEqual(report["nonfinite_rows"], 0)
        self.assertTrue(report["checkpoint_structurally_valid_archive"])
        # The report must not claim something this script does not actually do.
        self.assertFalse(report["checkpoint_semantically_loaded_via_real_inference_path"])

    def test_padded_magic_blob_is_not_an_archive(self):
        job = _make_valid_job(self.tmp_path, checkpoint_bytes=b"PK\x03\x04" + b"x" * 2048)
        with self.assertRaisesRegex(JobOutputVerificationError, "invalid ZIP"):
            verify_job_output(job, approach="lstm_multiscale", expected_d=14)

    def test_crc_corruption_is_rejected(self):
        blob = bytearray(_REAL_ZIP_HEADER)
        offset = blob.index(b"not-unpickled-by-this-verifier")
        blob[offset] ^= 1
        job = _make_valid_job(self.tmp_path, checkpoint_bytes=bytes(blob))
        with self.assertRaisesRegex(JobOutputVerificationError, "ZIP"):
            verify_job_output(job, approach="lstm_multiscale", expected_d=14)

    def test_zip_without_tensor_layout_is_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("padding", b"x" * 2048)
        job = _make_valid_job(self.tmp_path, checkpoint_bytes=buf.getvalue())
        with self.assertRaisesRegex(JobOutputVerificationError, "data.pkl"):
            verify_job_output(job, approach="lstm_multiscale", expected_d=14)

    def test_invalid_timestamps_rejected_before_values(self):
        for i, x in enumerate([77016, -1, 1.5, "nan", "inf"]):
            with self.subTest(x=x):
                job = _make_valid_job(self.tmp_path, name=str(i), rows=[
                    ("lstm_multiscale", "test", x, float("nan"), float("nan"))])
                with self.assertRaisesRegex(JobOutputVerificationError, "prediction timestamp"):
                    verify_job_output(job, approach="lstm_multiscale", expected_d=14)

    def test_duplicate_and_unordered_timestamps(self):
        for i, xs in enumerate([[1, 1, 3], [2, 1, 3]]):
            job = _make_valid_job(self.tmp_path, name=str(i), rows=[
                ("lstm_multiscale", "test" if x == 3 else "validation", x, 1, 2) for x in xs])
            with self.assertRaisesRegex(JobOutputVerificationError, "duplicate or unordered"):
                verify_job_output(job, approach="lstm_multiscale", expected_d=14)

    def test_split_must_match_endpoint_interval(self):
        job = _make_valid_job(self.tmp_path, rows=[("lstm_multiscale", "test", 1, 1, 2)])
        with self.assertRaisesRegex(JobOutputVerificationError, "split interval"):
            verify_job_output(job, approach="lstm_multiscale", expected_d=14)

    def test_frozen_config_and_exact_support(self):
        job = _make_valid_job(self.tmp_path)
        report = verify_job_output(job, approach="lstm_multiscale", expected_d=14,
            expected_config={"train_end_x": 0}, expected_support={"validation": [1, 2], "test": [3]})
        self.assertTrue(report["exact_declared_support_verified"])
        self.assertFalse(report["full_scientific_verification"])
        self.assertFalse(report["checkpoint_semantically_loaded_via_real_inference_path"])
        with self.assertRaisesRegex(JobOutputVerificationError, "config identity"):
            verify_job_output(job, approach="lstm_multiscale", expected_d=14, expected_config={"train_end_x": 1})
        for support in [{"validation": [1], "test": [3]}, {"validation": [1,2], "test": [3,4]}]:
            with self.assertRaisesRegex(JobOutputVerificationError, "exact support mismatch"):
                verify_job_output(job, approach="lstm_multiscale", expected_d=14, expected_support=support)

    def test_reserved_config_bound_rejected(self):
        job = _make_valid_job(self.tmp_path, config_overrides={"test_end_x":77016})
        with self.assertRaisesRegex(JobOutputVerificationError, "test_end_x"):
            verify_job_output(job, approach="lstm_multiscale", expected_d=14)

    def test_missing_config_is_rejected(self):
        job_dir = _make_valid_job(self.tmp_path)
        (job_dir / "experiment_config.json").unlink()
        with self.assertRaisesRegex(JobOutputVerificationError, "missing config file"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)

    def test_wrong_declared_d_is_rejected(self):
        job_dir = _make_valid_job(self.tmp_path, d=30)
        with self.assertRaisesRegex(JobOutputVerificationError, "config identity"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)

    def test_undersized_checkpoint_stub_is_rejected(self):
        """A truncated/header-only checkpoint file -- the exact stub-output
        failure mode this check exists to catch."""
        job_dir = _make_valid_job(self.tmp_path, checkpoint_bytes=b"PK\x03\x04tiny")
        with self.assertRaisesRegex(JobOutputVerificationError, "too small"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)

    def test_checkpoint_without_zip_magic_is_rejected(self):
        job_dir = _make_valid_job(self.tmp_path, checkpoint_bytes=b"not-a-zip-archive" + b"\x00" * 1024)
        with self.assertRaisesRegex(JobOutputVerificationError, "invalid ZIP"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)

    def test_header_only_predictions_csv_is_rejected(self):
        """A schema-valid predictions.csv with a header row but zero data rows
        for the target approach -- the header-only stub-output case named
        explicitly in the governing instruction."""
        job_dir = _make_valid_job(self.tmp_path, rows=[])
        with self.assertRaisesRegex(JobOutputVerificationError, "header-only stub-output"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)

    def test_missing_required_column_is_rejected(self):
        job_dir = self.tmp_path / "bad_schema"
        (job_dir / "models").mkdir(parents=True)
        with open(job_dir / "predictions.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["approach", "index", "split", "x", "observed"])  # missing predicted, residual
            writer.writerow(["lstm_multiscale", 0, "validation", 1, 1.0])
        (job_dir / "experiment_config.json").write_text(json.dumps({"multiscale_daily_history_days": 14,"train_end_x":0,"validation_end_x":2,"test_end_x":3}))
        (job_dir / "models" / "lstm_multiscale.pt").write_bytes(_REAL_ZIP_HEADER)
        with self.assertRaisesRegex(JobOutputVerificationError, "missing required column"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)

    def test_nonfinite_value_is_rejected(self):
        job_dir = _make_valid_job(self.tmp_path, rows=[
            ("lstm_multiscale", "validation", 1, 1.0, 1.1),
            ("lstm_multiscale", "test", 3, 2.0, float("nan")),
        ])
        with self.assertRaisesRegex(JobOutputVerificationError, "non-finite or unparseable"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)

    def test_missing_required_split_is_rejected(self):
        """Only validation rows exported, no test rows -- the job did not
        export completely even though the file itself is non-empty."""
        job_dir = _make_valid_job(self.tmp_path, rows=[
            ("lstm_multiscale", "validation", 1, 1.0, 1.1),
            ("lstm_multiscale", "validation", 2, 2.0, 2.1),
        ])
        with self.assertRaisesRegex(JobOutputVerificationError, "required split"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)

    def test_caller_can_declare_validation_only_as_sufficient(self):
        """Phase-1 selection jobs are validation-focused; a caller that only
        requires the validation split must be able to say so explicitly,
        rather than this script imposing one universal split requirement."""
        job_dir = _make_valid_job(self.tmp_path, rows=[
            ("lstm_multiscale", "validation", 1, 1.0, 1.1),
            ("lstm_multiscale", "validation", 2, 2.0, 2.1),
        ])
        report = verify_job_output(
            job_dir, approach="lstm_multiscale", expected_d=14, required_splits=["validation"],
        )
        self.assertEqual(report["rows_by_split"], {"validation": 2})

    def test_wrong_approach_rows_are_not_counted(self):
        """Rows for a different approach in the same file must not count
        toward this job's own row/split coverage."""
        job_dir = _make_valid_job(self.tmp_path, rows=[
            ("some_other_approach", "validation", 1, 1.0, 1.1),
            ("some_other_approach", "test", 2, 2.0, 2.1),
        ])
        with self.assertRaisesRegex(JobOutputVerificationError, "header-only stub-output"):
            verify_job_output(job_dir, approach="lstm_multiscale", expected_d=14)


if __name__ == "__main__":
    unittest.main()
