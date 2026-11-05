#!/usr/bin/env python3
"""Persistent regression tests for run_multiscale_selection_batch.sh.

Replaces the ad hoc, re-typed-by-hand bash probes used during this
session's Gate B review with a deterministic, re-runnable suite. Every test
uses a fake, non-training HydroBatch stub (writes expected output files or
exits 1, never trains anything) against synthetic manifests in a disposable
temp directory. No real HydroBatch binary, no real job, no network, no
reserved-period data.

Run directly: python3 test_run_multiscale_selection_batch.py
Or via unittest discovery alongside the other tests in this directory.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_multiscale_selection_batch.sh"

FAKE_HYDROBATCH = """#!/usr/bin/env bash
batch_file="$1"
job_out="$2"
sleep "${FAKE_JOB_SLEEP:-0.1}"
if [[ "$batch_file" == *shouldfail* ]]; then
  echo "simulated failure" >&2
  exit 1
fi
echo '{"ok":true}' > "$job_out/experiment_config.json"
echo "approach,index,split,x,observed,predicted,residual" > "$job_out/predictions.csv"
echo "file,sha256" > "$job_out/artifact_manifest.csv"
exit 0
"""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunMultiscaleSelectionBatchTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.bin_dir = self.tmp / "fakebin"
        self.bin_dir.mkdir()
        self.hydrobatch = self.bin_dir / "HydroBatch"
        self.hydrobatch.write_text(FAKE_HYDROBATCH)
        self.hydrobatch.chmod(self.hydrobatch.stat().st_mode | stat.S_IEXEC)
        self.manifests = self.tmp / "manifests"
        self.manifests.mkdir()
        # Configs are co-located with their .batch files, matching the real
        # phase1_selection deployment (each .batch file's referenced config
        # lives in the same manifest directory, not a separate tree) -- this
        # is what makes MANIFESTS_DIR itself the correct ROOT for a
        # MANIFEST_HASHES_FILE covering both.
        self.configs = self.manifests

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_job(self, name, config_name="d.json"):
        config = self.configs / config_name
        if not config.exists():
            config.write_text('{"ok":true}')
        batch = self.manifests / f"{name}.batch"
        batch.write_text(f"lstm_multiscale {config}\n")
        return batch, config

    def _run(self, output_root, *args, env_overrides=None, workers="2"):
        env = dict(os.environ)
        env["HYDROBATCH_BIN"] = str(self.hydrobatch)
        if env_overrides:
            env.update(env_overrides)
        cmd = ["bash", str(SCRIPT), str(self.manifests), str(output_root), workers, *args]
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)

    def _hashes_file(self, root, entries):
        lines = [f"ROOT: {root}"]
        for h, p in entries:
            lines.append(f"{h}  {p}")
        path = self.tmp / "hashes.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    # --- fail-closed defaults --------------------------------------------

    def test_refuses_without_hash_and_without_bypass(self):
        self._write_job("a")
        result = self._run(self.tmp / "out1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no expected HydroBatch hash supplied", result.stderr)

    def test_refuses_without_manifest_hashes_file_even_with_hash_supplied(self):
        self._write_job("a")
        result = self._run(
            self.tmp / "out2",
            env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch)},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("no MANIFEST_HASHES_FILE supplied", result.stderr)

    def test_bypass_flag_allows_launch_with_neither(self):
        self._write_job("a")
        result = self._run(self.tmp / "out3", env_overrides={"UNVERIFIED_HYDROBATCH_TEST_RUN": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("UNVERIFIED_HYDROBATCH_TEST_RUN=1", result.stderr)

    def test_hash_mismatch_refused(self):
        self._write_job("a")
        result = self._run(
            self.tmp / "out4",
            env_overrides={"EXPECTED_HYDROBATCH_SHA256": "0" * 64,
                            "MANIFEST_HASHES_FILE": "/dev/null"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("hash mismatch", result.stderr)

    # --- positive-integer validation --------------------------------------

    def test_zero_workers_refused(self):
        self._write_job("a")
        result = self._run(self.tmp / "out5", env_overrides={"UNVERIFIED_HYDROBATCH_TEST_RUN": "1"}, workers="0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("max_workers must be a positive integer", result.stderr)

    def test_negative_timeout_refused(self):
        self._write_job("a")
        result = self._run(
            self.tmp / "out6",
            env_overrides={"UNVERIFIED_HYDROBATCH_TEST_RUN": "1", "PER_JOB_TIMEOUT_SECONDS": "-1"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("PER_JOB_TIMEOUT_SECONDS must be a positive integer", result.stderr)

    def test_nonnumeric_memory_limit_refused(self):
        self._write_job("a")
        result = self._run(
            self.tmp / "out7",
            env_overrides={"UNVERIFIED_HYDROBATCH_TEST_RUN": "1", "PER_JOB_MEMORY_LIMIT_KB": "abc"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("PER_JOB_MEMORY_LIMIT_KB must be a positive integer", result.stderr)

    # --- manifest inventory verification ----------------------------------

    def test_manifest_verification_succeeds_with_complete_matching_inventory(self):
        batch_a, config_a = self._write_job("job_a", "d14.json")
        batch_b, config_b = self._write_job("job_b", "d30.json")
        hashes = self._hashes_file(self.manifests, [
            (sha256_of(batch_a), "job_a.batch"),
            (sha256_of(batch_b), "job_b.batch"),
            (sha256_of(config_a), str(config_a)),
            (sha256_of(config_b), str(config_b)),
        ])
        result = self._run(self.tmp / "out8", env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch), "MANIFEST_HASHES_FILE": str(hashes)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Manifest verification OK", result.stdout + result.stderr)

    def test_extra_unlisted_batch_file_refused(self):
        """The exact gap directly reproduced against `sha256sum --check`:
        an extra .batch file not in the inventory must be refused, not
        silently dispatched."""
        batch_a, config_a = self._write_job("job_a", "d14.json")
        hashes = self._hashes_file(self.manifests, [
            (sha256_of(batch_a), "job_a.batch"),
            (sha256_of(config_a), str(config_a)),
        ])
        self._write_job("job_b_unlisted", "d14.json")  # not in hashes file
        result = self._run(self.tmp / "out9", env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch), "MANIFEST_HASHES_FILE": str(hashes)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unlisted .batch file", result.stderr)

    def test_missing_batch_entry_refused(self):
        """The hash file expects a job_b.batch that was never actually
        written -- the inverse of the extra-file case above: an entry
        promised by the inventory but absent on disk."""
        batch_a, config_a = self._write_job("job_a", "d14.json")
        hashes = self._hashes_file(self.manifests, [
            (sha256_of(batch_a), "job_a.batch"),
            (sha256_of(config_a), str(config_a)),
            ("0" * 64, "job_b.batch"),  # never written
        ])
        result = self._run(self.tmp / "out10", env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch), "MANIFEST_HASHES_FILE": str(hashes)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing", result.stderr.lower())

    def test_referenced_config_not_covered_refused(self):
        batch_a, config_a = self._write_job("job_a", "d14.json")
        hashes = self._hashes_file(self.manifests, [(sha256_of(batch_a), "job_a.batch")])
        result = self._run(self.tmp / "out11", env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch), "MANIFEST_HASHES_FILE": str(hashes)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("Referenced config(s) not covered", result.stderr)

    def test_tampered_config_refused(self):
        batch_a, config_a = self._write_job("job_a", "d14.json")
        hashes = self._hashes_file(self.manifests, [
            (sha256_of(batch_a), "job_a.batch"),
            (sha256_of(config_a), str(config_a)),
        ])
        config_a.write_text('{"tampered": true}')
        result = self._run(self.tmp / "out12", env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch), "MANIFEST_HASHES_FILE": str(hashes)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("Manifest hash mismatch", result.stderr)

    def test_duplicate_entry_refused(self):
        batch_a, config_a = self._write_job("job_a", "d14.json")
        hashes_path = self.tmp / "dup_hashes.txt"
        hashes_path.write_text(
            f"ROOT: {self.manifests}\n"
            f"{sha256_of(batch_a)}  job_a.batch\n"
            f"{sha256_of(batch_a)}  job_a.batch\n"
            f"{sha256_of(config_a)}  {config_a}\n"
        )
        result = self._run(self.tmp / "out13", env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch), "MANIFEST_HASHES_FILE": str(hashes_path)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("Duplicate entry", result.stderr)

    def test_path_escaping_root_refused(self):
        batch_a, config_a = self._write_job("job_a", "d14.json")
        outside = self.tmp / "outside_root.json"
        outside.write_text("{}")
        hashes_path = self.tmp / "escape_hashes.txt"
        hashes_path.write_text(
            f"ROOT: {self.manifests}\n"
            f"{sha256_of(batch_a)}  job_a.batch\n"
            f"{sha256_of(outside)}  {outside}\n"
        )
        result = self._run(self.tmp / "out14", env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch), "MANIFEST_HASHES_FILE": str(hashes_path)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("resolves outside the declared ROOT", result.stderr)

    def test_missing_root_line_refused(self):
        batch_a, config_a = self._write_job("job_a", "d14.json")
        hashes_path = self.tmp / "noroot_hashes.txt"
        hashes_path.write_text(f"{sha256_of(batch_a)}  job_a.batch\n")
        result = self._run(self.tmp / "out15", env_overrides={"EXPECTED_HYDROBATCH_SHA256": sha256_of(self.hydrobatch), "MANIFEST_HASHES_FILE": str(hashes_path)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("must start with a 'ROOT:", result.stderr)

    # --- failure-stop and race-condition behavior -------------------------

    def test_stops_dispatch_after_failure_including_slot_wait_race(self):
        """1 worker, 3 jobs, the 2nd fails: the 3rd must never be dispatched,
        even though its top-of-loop check could pass before job 2 actually
        fails (the race this session's own fix closed)."""
        self._write_job("job_1")
        self._write_job("job_2_shouldfail")
        self._write_job("job_3")
        result = self._run(
            self.tmp / "out16",
            env_overrides={"UNVERIFIED_HYDROBATCH_TEST_RUN": "1"},
            workers="1",
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.tmp / "out16" / "job_3").exists(),
                          "job_3 must not be dispatched after job_2 failed")
        self.assertTrue((self.tmp / "out16" / "job_1").exists())

    # --- duplicate-launch lock ---------------------------------------------

    def test_concurrent_invocations_are_mutually_exclusive(self):
        self._write_job("job_a")
        env = dict(os.environ)
        env["HYDROBATCH_BIN"] = str(self.hydrobatch)
        env["UNVERIFIED_HYDROBATCH_TEST_RUN"] = "1"
        env["FAKE_JOB_SLEEP"] = "1.5"
        with subprocess.Popen(
            ["bash", str(SCRIPT), str(self.manifests), str(self.tmp / "out_dup1"), "1"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ) as first:
            time.sleep(0.3)
            second = subprocess.run(
                ["bash", str(SCRIPT), str(self.manifests), str(self.tmp / "out_dup2"), "1"],
                env=env, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(second.returncode, 2)
            self.assertIn("already holds the lock", second.stderr)
            first.wait(timeout=10)
            self.assertEqual(first.returncode, 0)

    # --- live running markers ------------------------------------------------

    def test_running_marker_present_during_and_absent_after(self):
        self._write_job("job_a")
        self._write_job("job_b")
        env = dict(os.environ)
        env["HYDROBATCH_BIN"] = str(self.hydrobatch)
        env["UNVERIFIED_HYDROBATCH_TEST_RUN"] = "1"
        env["FAKE_JOB_SLEEP"] = "1.0"
        output_root = self.tmp / "out_live"
        with subprocess.Popen(
            ["bash", str(SCRIPT), str(self.manifests), str(output_root), "2"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ) as proc:
            time.sleep(0.4)
            running_dir = output_root / ".running"
            markers_while_running = list(running_dir.glob("*")) if running_dir.exists() else []
            proc.wait(timeout=10)
            self.assertEqual(proc.returncode, 0)
        markers_after = list(running_dir.glob("*"))
        self.assertTrue(len(markers_while_running) > 0, "expected at least one live marker mid-run")
        self.assertEqual(markers_after, [], "markers must be removed after completion")


if __name__ == "__main__":
    unittest.main()
