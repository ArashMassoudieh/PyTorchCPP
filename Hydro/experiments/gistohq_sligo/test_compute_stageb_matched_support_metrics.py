#!/usr/bin/env python3
"""Regression tests for the real bugs found on independent review
(2026-11-16/17, CLAUDE_SUPERVISOR_TASKS.md action-checklist item 1) in
compute_stageb_matched_support_metrics.py, each reproduced directly before
being fixed:

1. Unequal control/multiscale eligible-support sets were silently scored on
   their intersection instead of being refused.
2. A boundary_provenance.json sidecar's declared hashes were trusted
   without re-verifying them against the actual current file bytes.
3. Non-finite/non-integer/sealed timestamp guards only ran for rows of the
   approach being scored, skipping them entirely for any other approach's
   rows sharing the same predictions.csv.
4. (checkpoint66) A sidecar's declared_from_staged_input bounds were never
   cross-checked against the staged file's own PARSED content, nor against
   the independent frozen-manifest hash -- only the sidecar's own
   self-reported hash of the staged file was checked, so a sidecar could
   declare fabricated bounds for an otherwise genuinely-unmodified file and
   still pass.
"""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from compute_stageb_matched_support_metrics import (
    MatchedSupportError,
    compare_fold_seed_pair,
    read_split_rows_fail_closed,
)

_BOUNDS = {"train_end_x": 0, "validation_end_x": 1000, "test_end_x": 2000}
_D = 30


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_CATCHMENT_AREA_M2 = 19.0893 * 1_000_000.0


def _write_job_dir(root: Path, name: str, approach: str, test_rows: dict,
                    *, bounds: dict = None, extra_rows: list = None,
                    matched_control: bool = None, d: int = _D, seed: int = 42,
                    declared_bounds: dict = None, experiment_id: str = None,
                    catchment_area_m2: float = _CATCHMENT_AREA_M2) -> Path:
    """test_rows: {x: (observed, predicted)}, written as split='test'.
    extra_rows: optional list of raw dicts for additional CSV rows (used to
    inject a different approach's own corrupted row). declared_bounds, if
    given, is what the sidecar CLAIMS (may deliberately differ from the
    staged file's real `bounds`, to reproduce a fabricated-declaration
    sidecar). experiment_id, if given, overrides the default (== name) --
    used to simulate a whole-directory swap where the files' own stated
    identity doesn't match the directory they're sitting in."""
    bounds = bounds or _BOUNDS
    declared_bounds = declared_bounds if declared_bounds is not None else bounds
    experiment_id = experiment_id if experiment_id is not None else name
    if matched_control is None:
        matched_control = approach == "lstm_pinn"
    job_dir = root / name
    (job_dir / "models").mkdir(parents=True)
    checkpoint_name = "".join(c.lower() if c.isalnum() else "_" for c in approach) + ".pt"
    (job_dir / "models" / checkpoint_name).write_bytes(f"fake-checkpoint-{name}".encode())
    (job_dir / "dataset_manifest.json").write_text(json.dumps({"catchment_area_m2": catchment_area_m2}))

    with open(job_dir / "predictions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["approach", "index", "split", "x", "observed", "predicted", "residual"]
        )
        writer.writeheader()
        for index, (x, (observed, predicted)) in enumerate(sorted(test_rows.items())):
            writer.writerow({
                "approach": approach, "index": index, "split": "test", "x": x,
                "observed": observed, "predicted": predicted, "residual": observed - predicted,
            })
        for extra in extra_rows or []:
            writer.writerow(extra)

    exported_config = job_dir / "experiment_config.json"
    exported_config.write_text(json.dumps({
        "experiment_id": experiment_id, "random_seed": seed,
        "multiscale_daily_history_days": d, "multiscale_matched_control": matched_control,
    }))

    staged_config = root / f"{name}_staged_input.json"
    staged_config.write_text(json.dumps({
        "experiment_id": experiment_id, "random_seed": seed,
        "multiscale_daily_history_days": d, "multiscale_matched_control": matched_control,
        **bounds,
    }))

    sidecar = {
        "exported_config_path": str(exported_config),
        "exported_config_sha256": _sha256_file(exported_config),
        "staged_input_config_path": str(staged_config),
        "staged_input_config_sha256": _sha256_file(staged_config),
        "declared_from_staged_input": declared_bounds,
    }
    (job_dir / "boundary_provenance.json").write_text(json.dumps(sidecar))

    manifest_path = root / "STAGED_MANIFEST_HASHES.txt"
    with open(manifest_path, "a") as handle:
        if manifest_path.stat().st_size == 0:
            handle.write(f"ROOT: {root}\n")
        handle.write(f"{_sha256_file(staged_config)}  {staged_config.resolve()}\n")

    return job_dir


def _frozen_manifest_hashes(root: Path) -> dict:
    from compute_stageb_matched_support_metrics import _parse_frozen_manifest_hashes
    return _parse_frozen_manifest_hashes(root / "STAGED_MANIFEST_HASHES.txt")


def _compare(control_dir, multiscale_dir, *, split="test", label="test-fold",
             expected_d=_D, expected_seed=42, expected_year=None,
             frozen_manifest_hashes=None, native_support=None, root=None):
    if frozen_manifest_hashes is None:
        frozen_manifest_hashes = _frozen_manifest_hashes(root)
    if native_support is None:
        native_support = set(range(0, 100000))
    return compare_fold_seed_pair(
        control_dir, multiscale_dir, split=split, expected_d=expected_d,
        expected_seed=expected_seed, frozen_manifest_hashes=frozen_manifest_hashes,
        native_support=native_support, label=label, expected_year=expected_year,
    )


class ComputeStageBMatchedSupportMetricsTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_unequal_support_is_refused_not_scored_on_intersection(self):
        """Bug 1, reproduced directly: control has an extra eligible x the
        multiscale arm doesn't. The intersection (1500, 1600) would still
        produce a 'valid-looking' NSE if silently scored -- this must be
        refused instead."""
        control_dir = _write_job_dir(
            self.tmp_path, "control",
            "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9), 1700: (7.0, 6.8)},
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale",
            "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)},
        )
        with self.assertRaisesRegex(MatchedSupportError, "NOT identical"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path)

    def test_sidecar_hash_mismatch_is_rejected(self):
        """Bug 2, reproduced directly: the sidecar's declared
        exported_config_sha256 no longer matches the actual file (as if the
        exported config changed, or the sidecar is stale/wrong) -- must be
        rejected, not silently trusted."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)},
        )
        # Mutate the exported config AFTER the sidecar was written, without
        # updating the sidecar's own recorded hash -- exactly the scenario
        # the hash is supposed to catch.
        (control_dir / "experiment_config.json").write_text(
            json.dumps({"experiment_id": "control", "random_seed": 999})
        )
        with self.assertRaisesRegex(MatchedSupportError, "does not match the actual current file"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path)

    def test_other_approach_sealed_timestamp_is_rejected_before_approach_filter(self):
        """Bug 3, reproduced directly: a row for a DIFFERENT approach
        (not the one being scored) carries a sealed/reserved-period
        timestamp. The old code checked `row["approach"] != approach` and
        skipped BEFORE validating x, so this row was silently ignored.
        Must be rejected regardless of which approach it belongs to."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
            extra_rows=[{
                "approach": "some_other_approach", "index": 99, "split": "test",
                "x": 77016, "observed": 1.0, "predicted": 1.0, "residual": 0.0,
            }],
        )
        with self.assertRaisesRegex(MatchedSupportError, "reserved-period seal"):
            read_split_rows_fail_closed(
                control_dir / "predictions.csv", "lstm_pinn", split="test",
                interval_start_x=_BOUNDS["validation_end_x"], interval_end_x=_BOUNDS["test_end_x"],
            )

    def test_matching_support_still_scores_correctly(self):
        """Control: confirms the fix didn't break the ordinary, genuinely-
        matched case -- the one that matters for the real Stage B data."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9), 1700: (7.0, 7.1)},
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale",
            {1500: (5.0, 5.2), 1600: (6.0, 6.1), 1700: (7.0, 6.9)},
        )
        result = _compare(control_dir, multiscale_dir, root=self.tmp_path)
        self.assertEqual(result["n_matched"], 3)
        self.assertTrue(result["support_matches_exactly"])
        self.assertIn("nse", result["control"])
        self.assertIn("nse", result["multiscale"])
        self.assertIn("rmse_m3_s", result["control"])
        self.assertAlmostEqual(
            result["control"]["rmse_m3_s"], result["control"]["rmse_mm_h"] * 19.0893 / 3.6, places=9,
        )

    def test_fabricated_sidecar_declaration_is_rejected_despite_matching_hash(self):
        """checkpoint66, reproduced directly per Codex's 2026-11-16 20:19
        review: a sidecar's declared_from_staged_input bounds are fabricated
        (100/1400/1800) while the staged file it points at -- which the
        sidecar's own hash correctly matches -- actually contains
        0/1000/2000. The old _verify_sidecar only hash-checked the staged
        file as an opaque blob and never parsed it, so this passed. Must now
        be rejected because the sidecar's claim disagrees with the staged
        file's real parsed content."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
            declared_bounds={"train_end_x": 100, "validation_end_x": 1400, "test_end_x": 1800},
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)},
        )
        with self.assertRaisesRegex(MatchedSupportError, "does not match .* staged input file's actual parsed content"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path)

    def test_staged_input_not_in_frozen_manifest_is_rejected(self):
        """checkpoint66: a staged input file's hash is only checked against
        the sidecar's own self-reported claim, never against an independent
        anchor -- two mutually-consistent-but-wrong values would otherwise
        pass. Here the frozen-manifest file supplied has no entry at all for
        the staged input, so verification must refuse rather than skip."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)},
        )
        with self.assertRaisesRegex(MatchedSupportError, "not present in the frozen manifest"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path, frozen_manifest_hashes={})

    def test_wrong_expected_d_is_rejected(self):
        """checkpoint66: D/seed/matched-control identity is now freshly
        re-derived from both files rather than trusted from the sidecar's
        own copy. Calling with the wrong expected_d (as if a Stage B job
        were accidentally scored against the wrong fold's resolved D) must
        be rejected."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)}, d=30,
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)}, d=30,
        )
        with self.assertRaisesRegex(MatchedSupportError, "identity mismatch"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path, expected_d=60)

    def test_non_native_matched_x_is_excluded_from_scoring(self):
        """2026-11-16 review: equal eligible-x sets are not by themselves
        evidence of NATIVE observation support. Restrict native_support to
        exclude x=1700 and confirm it's dropped from scoring (n_matched=2,
        not 3) even though it was eligible and matched."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9), 1700: (7.0, 7.1)},
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale",
            {1500: (5.0, 5.2), 1600: (6.0, 6.1), 1700: (7.0, 6.9)},
        )
        result = _compare(
            control_dir, multiscale_dir, root=self.tmp_path, native_support={1500, 1600},
        )
        self.assertEqual(result["n_eligible_matched"], 3)
        self.assertEqual(result["n_matched"], 2)
        self.assertEqual(result["n_native_matched"], 2)

    def test_unknown_split_is_rejected(self):
        """checkpoint67: an unrecognized split string used to fall through
        an if/else and be silently treated as 'validation'. Must be
        rejected explicitly instead."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)},
        )
        with self.assertRaisesRegex(MatchedSupportError, "is not one of"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path, split="bogus")

    def test_negative_x_is_rejected(self):
        """checkpoint67: hour-index timestamps must be nonnegative; a
        negative x previously passed the finite/integer/sealed checks
        unchallenged."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "predictions.csv"
            with open(csv_path, "w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["approach", "index", "split", "x", "observed", "predicted", "residual"]
                )
                writer.writeheader()
                writer.writerow({
                    "approach": "lstm_pinn", "index": 0, "split": "test", "x": -5,
                    "observed": 1.0, "predicted": 1.0, "residual": 0.0,
                })
            with self.assertRaisesRegex(MatchedSupportError, "Negative x"):
                read_split_rows_fail_closed(
                    csv_path, "lstm_pinn", split="test", interval_start_x=-100, interval_end_x=100,
                )

    def test_native_support_non_integer_coordinate_is_rejected(self):
        """checkpoint67: _load_native_discharge_support used to truncate a
        non-integer coordinate with int() instead of rejecting it -- a
        corrupted 1500.7 would silently become 1500. Must refuse instead."""
        import compute_stageb_matched_support_metrics as base
        with tempfile.TemporaryDirectory() as tmp:
            native_path = Path(tmp) / "native.json"
            native_path.write_text(json.dumps({"strict_observed_hour_x": [1500, 1600.7]}))
            orig = base.NATIVE_DISCHARGE_SUPPORT
            base.NATIVE_DISCHARGE_SUPPORT = native_path
            try:
                with self.assertRaisesRegex(MatchedSupportError, "non-finite, non-integer, or negative"):
                    base._load_native_discharge_support()
            finally:
                base.NATIVE_DISCHARGE_SUPPORT = orig

    def test_wrong_expected_seed_is_rejected_despite_internal_agreement(self):
        """checkpoint67: the prior identity check only confirmed exported
        and staged configs agreed WITH EACH OTHER on random_seed, not that
        either matched the caller's actual expectation. Build a pair that
        is perfectly self-consistent (seed=999 in both exported and staged)
        but wrong relative to what the caller (expected_seed=42) actually
        wants -- exactly the 'swapped pair' scenario Codex described."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)}, seed=999,
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)}, seed=999,
        )
        with self.assertRaisesRegex(MatchedSupportError, "does not match the expected seed"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path, expected_seed=42)

    def test_whole_directory_fold_swap_is_rejected_via_expected_year(self):
        """checkpoint67: two folds can share the same D (2018 and 2020 both
        resolved D=30 in the real Stage B run), so expected_d alone cannot
        catch a whole-directory swap between same-D folds. Build a job
        whose files internally agree (experiment_id says '2020', matching
        between exported and staged) but the caller actually expected
        '2018' -- simulating a 2018/2020 directory mixup."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
            experiment_id="sixyear_corrected_forcing_2020_zero_multiscale_control_retrain_s42",
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)},
            experiment_id="sixyear_corrected_forcing_2020_zero_multiscale_s42",
        )
        with self.assertRaisesRegex(MatchedSupportError, "does not contain the expected fold year"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path, expected_year="2018")

    def test_catchment_area_mismatch_is_rejected(self):
        """checkpoint67: the CATCHMENT_AREA_KM2 comment claimed this gets
        checked against each job's own dataset_manifest.json, but no such
        check existed anywhere -- a false claim, now actually implemented
        and tested."""
        control_dir = _write_job_dir(
            self.tmp_path, "control", "lstm_pinn", {1500: (5.0, 5.1), 1600: (6.0, 5.9)},
            catchment_area_m2=23754600.0,  # the superseded, wrong GIStoOHQ DEM-delineation area
        )
        multiscale_dir = _write_job_dir(
            self.tmp_path, "multiscale", "lstm_multiscale", {1500: (5.0, 5.2), 1600: (6.0, 6.1)},
        )
        with self.assertRaisesRegex(MatchedSupportError, "does not match the CATCHMENT_AREA_KM2"):
            _compare(control_dir, multiscale_dir, root=self.tmp_path)


class CliTest(unittest.TestCase):
    def test_cli_rejects_unknown_split_via_argparse_choices(self):
        import compute_stageb_matched_support_metrics as base
        with self.assertRaises(SystemExit):
            base._parse_args(["bogus-split"])

    def test_cli_parses_out_and_allow_overwrite(self):
        import compute_stageb_matched_support_metrics as base
        args = base._parse_args(["validation", "--out", "/tmp/x.json", "--allow-overwrite"])
        self.assertEqual(args.split, "validation")
        self.assertEqual(args.out, Path("/tmp/x.json"))
        self.assertTrue(args.allow_overwrite)

    def test_cli_defaults_to_test_split_no_overwrite(self):
        import compute_stageb_matched_support_metrics as base
        args = base._parse_args([])
        self.assertEqual(args.split, "test")
        self.assertIsNone(args.out)
        self.assertFalse(args.allow_overwrite)


if __name__ == "__main__":
    unittest.main()
