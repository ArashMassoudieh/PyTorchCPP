#!/usr/bin/env python3
"""Verify a completed training job's exported output beyond "the files exist."

Per the standing preflight instruction (CLAUDE_RESULTS_IMPROVEMENT_TASKS_20261105.md
Task 1): "Finish output verification before accepting a job: filenames existing alone
is insufficient; validate schema, finite values, timestamps/support, config identity
and loadable checkpoint with the intended inference path. Avoid treating header-only
stub output as scientifically verified."

This module is intentionally separate from select_multiscale_history_length.py: the
selector answers "which D wins," scored on validation rows only; this module answers
"is this job's own export actually a valid, complete scientific artifact," checked
once per job, independent of whether that job happens to be a D-selection candidate.

Scope, stated precisely so it is not overclaimed:
  - Schema, non-emptiness, finiteness, split/support coverage, and config/D identity
    are fully verified here, from the actual exported files.
  - The checkpoint file is verified to be a structurally valid LibTorch/zip archive
    (correct magic bytes, non-trivial size) -- this catches a header-only or truncated
    stub file. It does NOT load the checkpoint through the real C++ inference path
    (HydroInferenceRunner) and does NOT reproduce a prediction from it; that requires
    the compiled HydroPINN/HydroBatch binary, which this no-training, Python-only
    preflight diagnostic does not invoke. Full semantic load verification -- the kind
    this program's own `shared_scale_inference.cpp`/`inference_runner.cpp` tooling has
    done for other checkpoint families -- remains a separate, real-binary step, not
    something this script claims to replace.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

REQUIRED_PREDICTIONS_COLUMNS = {"approach", "index", "split", "x", "observed", "predicted", "residual"}

# LibTorch's torch.save / torch::save format is a standard zip archive.
_ZIP_MAGIC = b"PK\x03\x04"


class JobOutputVerificationError(ValueError):
    """Raised when a job's exported output fails scientific-output verification."""


def _checkpoint_filename(approach: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in approach) + ".pt"


def verify_job_output(
    job_dir: Path | str,
    *,
    approach: str,
    expected_d: int,
    required_splits: Iterable[str] = ("validation", "test"),
) -> dict:
    """Verify one completed job's export. Raises JobOutputVerificationError on any
    failure; returns a report dict of what was checked and found on success."""
    job_dir = Path(job_dir)
    config_path = job_dir / "experiment_config.json"
    predictions_path = job_dir / "predictions.csv"
    checkpoint_path = job_dir / "models" / _checkpoint_filename(approach)

    for label, path in (
        ("config", config_path), ("predictions", predictions_path), ("checkpoint", checkpoint_path),
    ):
        if not path.is_file():
            raise JobOutputVerificationError(f"{job_dir}: missing {label} file {path}")

    # --- Config identity ---
    config_data = json.loads(config_path.read_text())
    declared_d = config_data.get("multiscale_daily_history_days")
    if declared_d != expected_d:
        raise JobOutputVerificationError(
            f"{job_dir}: config declares multiscale_daily_history_days={declared_d!r}, "
            f"expected {expected_d} -- config identity does not match the job's registration"
        )

    # --- Checkpoint: structurally valid archive, not a stub ---
    checkpoint_size = checkpoint_path.stat().st_size
    if checkpoint_size < 1024:
        raise JobOutputVerificationError(
            f"{job_dir}: checkpoint {checkpoint_path} is only {checkpoint_size} bytes -- "
            "too small to be a real exported model; this is the header-only/stub-output "
            "failure mode this check exists to catch"
        )
    with open(checkpoint_path, "rb") as handle:
        magic = handle.read(4)
    if magic != _ZIP_MAGIC:
        raise JobOutputVerificationError(
            f"{job_dir}: checkpoint {checkpoint_path} does not start with a valid zip-archive "
            f"signature (got {magic!r}) -- not a structurally valid LibTorch checkpoint. "
            "(This checks the archive's structural validity only; it does not load the "
            "checkpoint through the real inference path -- see module docstring.)"
        )

    # --- predictions.csv: schema, non-emptiness, finiteness, split/support coverage ---
    with open(predictions_path, newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        missing_columns = REQUIRED_PREDICTIONS_COLUMNS - header
        if missing_columns:
            raise JobOutputVerificationError(
                f"{job_dir}: predictions.csv is missing required column(s) {sorted(missing_columns)} "
                f"-- has {sorted(header)}"
            )

        rows_by_split: dict[str, int] = {}
        nonfinite_count = 0
        total_rows_for_approach = 0
        for row in reader:
            if row["approach"] != approach:
                continue
            total_rows_for_approach += 1
            split = row["split"]
            rows_by_split[split] = rows_by_split.get(split, 0) + 1
            try:
                observed_value = float(row["observed"])
                predicted_value = float(row["predicted"])
            except ValueError:
                nonfinite_count += 1
                continue
            if not (math.isfinite(observed_value) and math.isfinite(predicted_value)):
                nonfinite_count += 1

    if total_rows_for_approach == 0:
        raise JobOutputVerificationError(
            f"{job_dir}: predictions.csv has a valid header but zero data rows for "
            f"approach {approach!r} -- this is exactly the header-only stub-output case "
            "this check exists to catch; a schema-valid empty file is not a scientifically "
            "verified job"
        )
    if nonfinite_count:
        raise JobOutputVerificationError(
            f"{job_dir}: {nonfinite_count} row(s) for approach {approach!r} have a "
            "non-finite or unparseable observed/predicted value"
        )
    missing_splits = set(required_splits) - set(rows_by_split)
    if missing_splits:
        raise JobOutputVerificationError(
            f"{job_dir}: predictions.csv has no rows for required split(s) "
            f"{sorted(missing_splits)} for approach {approach!r} -- present splits: "
            f"{sorted(rows_by_split)}. A job missing an entire required split did not "
            "export completely, even if the file itself is non-empty."
        )

    return {
        "job_dir": str(job_dir),
        "approach": approach,
        "expected_d": expected_d,
        "config_declared_d": declared_d,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_structurally_valid_archive": True,
        "checkpoint_semantically_loaded_via_real_inference_path": False,
        "rows_by_split": rows_by_split,
        "total_rows_for_approach": total_rows_for_approach,
        "nonfinite_rows": nonfinite_count,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir")
    parser.add_argument("--approach", default="lstm_multiscale")
    parser.add_argument("--expected-d", type=int, required=True)
    parser.add_argument(
        "--required-split", action="append", dest="required_splits",
        help="May be repeated; defaults to validation and test if omitted.",
    )
    args = parser.parse_args(argv)
    required_splits = args.required_splits or ["validation", "test"]

    report = verify_job_output(
        args.job_dir, approach=args.approach, expected_d=args.expected_d,
        required_splits=required_splits,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
