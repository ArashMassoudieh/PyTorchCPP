#!/usr/bin/env python3
"""Select the winning daily-history length D per fold on common validation support.

Problem this fixes (2026-11-04 handoff addition to Task 4's preflight,
improvement_loop_20261023): eligibility is monotonic in D -- a window
eligible under a LARGER D (more required complete prior days) is
automatically eligible under every smaller D, but never the reverse. A
comparison that lets each D candidate report its own validation NSE over
its own eligible window set therefore compares different populations: the
largest D's own eligible set systematically excludes every early-record
window smaller D values still score, including whichever of those happen to
be hard examples. A longer D can then look better purely because its own
eligibility mask discarded the hard cases, not because more history helps.

Fix: every candidate's validation NSE is restricted to the intersection of
all candidates' eligible windows for that fold. Because eligibility is
monotonic, that intersection is exactly the largest D's own eligible set --
this module verifies that from the actual exported data rather than
assuming it, and refuses to select if it does not hold.

This selects on VALIDATION rows only -- never test -- and only ever reads
each candidate's own exported predictions.csv; it does not retrain or
recompute any metric inside the training loop. The report records the
common-support timestamp set itself (not just its size), the tie-break
rule in plain text, and a sha256 of each candidate's own config and
checkpoint file, so the selection is tied to exact, inspectable artifacts.

Input: each candidate's frozen job directory (the directory
experiment_exporter.cpp writes: experiment_config.json, predictions.csv,
models/<approach>.pt, ...).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Tuple

TIE_BREAK_RULE = (
    "Ties in common-support validation NSE are broken toward the smaller D "
    "(the simpler model) -- never toward whichever candidate happens first "
    "in iteration order."
)


class HistoryLengthSelectionError(ValueError):
    """Raised when the inputs do not support a valid common-support selection."""


# Reserved-period seal for the Sligo epoch (x>=77016 == 2023-10-15 onward).
# This selection script only ever reads a fold's own validation-split rows,
# and every phase1_selection fold's test_end_x is already below this bound
# by construction -- so this should never fire in practice. It exists as a
# defensive check, not a primary control: refusing explicitly on a seal
# violation is safer than silently scoring on it if some future caller ever
# points this script at a differently-bounded job directory.
RESERVED_PERIOD_X_CUTOFF = 77016


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_filename(approach: str) -> str:
    # Mirrors experiment_exporter.cpp's safeFileStem exactly: every
    # character lowercased if alphanumeric, else '_'.
    return "".join(c.lower() if c.isalnum() else "_" for c in approach) + ".pt"


def read_validation_rows(
    predictions_csv: Path,
    approach: str,
    *,
    validation_start_x: float,
    validation_end_x: float,
) -> Dict[float, Tuple[float, float]]:
    """Return {x: (observed, predicted)} for one approach's VALIDATION split only.

    Validates every field actually used, not only observed/predicted: x
    itself must be finite, must represent a whole-hour index (not merely
    "close to" one -- a non-integer x indicates a corrupted or
    wrong-format export, never a legitimate one), and must fall strictly
    before the reserved-period seal. Also checks split-consistency across
    the *whole* file for this approach, not only within the validation
    rows read: the same x must never appear under two different split
    labels for the same approach, since that would mean the export itself
    is ambiguous about which split a given hour belongs to.

    A row's CSV ``split`` label alone is not trusted either: a row labeled
    "validation" must also have x inside [validation_start_x,
    validation_end_x) -- the fold's own declared interval, taken from its
    config -- or it is rejected. Without this, a mislabeled or
    wrong-fold export could still be silently scored as if it were
    legitimate validation data for this fold.
    """
    rows: Dict[float, Tuple[float, float]] = {}
    x_to_split: Dict[float, str] = {}
    with open(predictions_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["approach"] != approach:
                continue
            x = float(row["x"])
            if not math.isfinite(x):
                raise HistoryLengthSelectionError(
                    f"Non-finite x={x} for approach {approach!r} in {predictions_csv}"
                )
            if x != int(x):
                raise HistoryLengthSelectionError(
                    f"Non-integer x={x} for approach {approach!r} in {predictions_csv} -- "
                    "hour-index timestamps must be whole numbers; this indicates a "
                    "corrupted or wrong-format export"
                )
            if x >= RESERVED_PERIOD_X_CUTOFF:
                raise HistoryLengthSelectionError(
                    f"x={x} is at or past the reserved-period seal "
                    f"(cutoff {RESERVED_PERIOD_X_CUTOFF}) in {predictions_csv} -- refusing "
                    "to process a row from the reserved period, regardless of split"
                )
            split = row["split"]
            if x in x_to_split and x_to_split[x] != split:
                raise HistoryLengthSelectionError(
                    f"x={x} for approach {approach!r} in {predictions_csv} is labeled both "
                    f"{x_to_split[x]!r} and {split!r} -- ambiguous split assignment, refusing "
                    "to select on it"
                )
            x_to_split[x] = split
            if split != "validation":
                continue
            if not (validation_start_x <= x < validation_end_x):
                raise HistoryLengthSelectionError(
                    f"x={x} for approach {approach!r} in {predictions_csv} is labeled "
                    "'validation' but falls outside the fold's own declared validation "
                    f"interval [{validation_start_x}, {validation_end_x}) -- the split label "
                    "alone is not trusted; refusing to score a row whose timestamp doesn't "
                    "actually belong to this fold's validation window"
                )
            if x in rows:
                raise HistoryLengthSelectionError(
                    f"Duplicate validation x={x} for approach {approach!r} in {predictions_csv}"
                )
            observed_value = float(row["observed"])
            predicted_value = float(row["predicted"])
            if not math.isfinite(observed_value) or not math.isfinite(predicted_value):
                raise HistoryLengthSelectionError(
                    f"Non-finite value at x={x} for approach {approach!r} in {predictions_csv} "
                    f"(observed={observed_value}, predicted={predicted_value}) -- refusing to "
                    "select on data that would silently poison the NSE comparison"
                )
            rows[x] = (observed_value, predicted_value)
    if not rows:
        raise HistoryLengthSelectionError(
            f"No validation rows found for approach {approach!r} in {predictions_csv}"
        )
    return rows


def nse(observed, predicted) -> float:
    if len(observed) != len(predicted) or not observed:
        raise HistoryLengthSelectionError("NSE requires matching, non-empty observed/predicted series")
    mean_observed = sum(observed) / len(observed)
    denominator = sum((value - mean_observed) ** 2 for value in observed)
    if denominator <= 0.0:
        raise HistoryLengthSelectionError(
            "NSE denominator is zero or negative (no observed variance on this support)"
        )
    numerator = sum((o - p) ** 2 for o, p in zip(observed, predicted))
    return 1.0 - numerator / denominator


def select_history_length(
    candidate_job_dirs: Dict[int, str | Path],
    *,
    approach: str = "lstm_multiscale",
) -> dict:
    """Pick the winning D for one fold, scored on common VALIDATION support.

    ``candidate_job_dirs`` maps each D candidate (e.g. 14/30/60) to that
    job's exported experiment directory. Returns a report with each
    candidate's coverage, common-support NSE, config/checkpoint sha256, the
    common-support timestamp set itself, the tie-break rule, and the
    selected D. Never reads a test-split row.
    """
    if len(candidate_job_dirs) < 2:
        raise HistoryLengthSelectionError("select_history_length requires at least two D candidates")

    job_dirs = {d: Path(path) for d, path in candidate_job_dirs.items()}
    fold_bounds_by_d: Dict[int, tuple] = {}
    for d, job_dir in job_dirs.items():
        if not isinstance(d, int) or d <= 0:
            raise HistoryLengthSelectionError(
                f"D={d!r} is not a positive integer -- daily-history length must be a "
                "positive number of days"
            )
        config_path = job_dir / "experiment_config.json"
        checkpoint_path = job_dir / "models" / _checkpoint_filename(approach)
        if not config_path.is_file():
            raise HistoryLengthSelectionError(f"D={d}: missing {config_path}")
        if not checkpoint_path.is_file():
            raise HistoryLengthSelectionError(f"D={d}: missing {checkpoint_path}")
        # A config's own sha256 says a file was hashed, not that it is the
        # *right* file: verify the config actually declares the D it was
        # registered under, not merely that some config exists at that path.
        config_data = json.loads(config_path.read_text())
        declared_d = config_data.get("multiscale_daily_history_days")
        if declared_d != d:
            raise HistoryLengthSelectionError(
                f"D={d}: {config_path} declares multiscale_daily_history_days="
                f"{declared_d!r}, not {d} -- this job directory does not match "
                "the D it was registered under"
            )
        raw_bounds = (
            config_data.get("train_end_x"),
            config_data.get("validation_end_x"),
            config_data.get("test_end_x"),
        )
        # Bounds must be PRESENT, FINITE, and CHRONOLOGICAL, not merely equal to every
        # other candidate's bounds -- three candidates that all simply omit these fields
        # would otherwise "match" trivially as (None, None, None) and pass undetected.
        if any(b is None for b in raw_bounds):
            raise HistoryLengthSelectionError(
                f"D={d}: {config_path} is missing one or more of "
                "train_end_x/validation_end_x/test_end_x -- a missing bound is not a "
                "valid fold boundary, even if every candidate is equally missing it"
            )
        train_end_x, validation_end_x, test_end_x = raw_bounds
        if not all(isinstance(b, (int, float)) and math.isfinite(b) for b in raw_bounds):
            raise HistoryLengthSelectionError(
                f"D={d}: {config_path}'s fold bounds are not all finite numbers: {raw_bounds}"
            )
        if not (train_end_x < validation_end_x < test_end_x):
            raise HistoryLengthSelectionError(
                f"D={d}: {config_path}'s fold bounds are not chronological "
                f"(train_end_x={train_end_x}, validation_end_x={validation_end_x}, "
                f"test_end_x={test_end_x}) -- expected train_end_x < validation_end_x < test_end_x"
            )
        if test_end_x >= RESERVED_PERIOD_X_CUTOFF:
            raise HistoryLengthSelectionError(
                f"D={d}: {config_path}'s test_end_x={test_end_x} is at or past the "
                f"reserved-period seal (cutoff {RESERVED_PERIOD_X_CUTOFF}) -- refusing a "
                "fold whose own declared bounds reach into the reserved period"
            )
        fold_bounds_by_d[d] = raw_bounds
    distinct_fold_bounds = set(fold_bounds_by_d.values())
    if len(distinct_fold_bounds) > 1:
        raise HistoryLengthSelectionError(
            "Candidates do not share identical (train_end_x, validation_end_x, "
            f"test_end_x) fold boundaries -- they are not the same fold: "
            f"{fold_bounds_by_d}"
        )
    shared_train_end_x, shared_validation_end_x, _shared_test_end_x = next(iter(distinct_fold_bounds))

    per_d_rows = {
        d: read_validation_rows(
            job_dir / "predictions.csv", approach,
            validation_start_x=shared_train_end_x, validation_end_x=shared_validation_end_x,
        )
        for d, job_dir in job_dirs.items()
    }
    per_d_xsets = {d: set(rows.keys()) for d, rows in per_d_rows.items()}

    # All candidates' observed values on identical x must agree before scoring: a
    # cross-export consistency check distinct from any single file's own internal checks
    # above. Two candidates sharing a real-world hour x but disagreeing on its observed
    # discharge means at least one export's target series is corrupted or mismatched --
    # directly reproduced before this check existed: two such candidates both "selected"
    # cleanly with NSE=1.0 each, computed against two different "truths" for the same hour.
    observed_by_x: Dict[float, Dict[int, float]] = {}
    for d, rows in per_d_rows.items():
        for x, (observed_value, _predicted_value) in rows.items():
            observed_by_x.setdefault(x, {})[d] = observed_value
    for x, per_d_observed in observed_by_x.items():
        if len(per_d_observed) < 2:
            continue
        values = list(per_d_observed.values())
        if max(values) - min(values) > 1e-6:
            raise HistoryLengthSelectionError(
                f"Candidates disagree on the observed value at shared x={x}: "
                f"{per_d_observed} -- this indicates a corrupted or mismatched target "
                "series between exports; refusing to score on inconsistent ground truth"
            )

    ordered_largest_first = sorted(per_d_xsets, reverse=True)
    largest_d = ordered_largest_first[0]
    common_support = set(per_d_xsets[largest_d])

    # Verify monotonicity from the actual exported data -- never assumed.
    for d in ordered_largest_first[1:]:
        if not common_support <= per_d_xsets[d]:
            missing = sorted(common_support - per_d_xsets[d])[:5]
            raise HistoryLengthSelectionError(
                f"Eligibility is not monotonic in this data: D={largest_d}'s eligible "
                f"validation windows are not a subset of D={d}'s (e.g. missing "
                f"x={missing}). Refusing to select on an assumption the data does "
                "not support."
            )
    if not common_support:
        raise HistoryLengthSelectionError("The common validation support across all D candidates is empty")

    per_candidate = {}
    for d, rows in per_d_rows.items():
        job_dir = job_dirs[d]
        total = len(rows)
        observed = [rows[x][0] for x in common_support]
        predicted = [rows[x][1] for x in common_support]
        per_candidate[d] = {
            "job_dir": str(job_dir),
            "total_eligible_validation_windows": total,
            "common_support_windows": len(common_support),
            "coverage_fraction": len(common_support) / total if total else 0.0,
            "common_support_nse": nse(observed, predicted),
            "config_sha256": _sha256_file(job_dir / "experiment_config.json"),
            "checkpoint_sha256": _sha256_file(job_dir / "models" / _checkpoint_filename(approach)),
        }

    best_nse = max(entry["common_support_nse"] for entry in per_candidate.values())
    selected_d = min(
        d for d, entry in per_candidate.items() if entry["common_support_nse"] == best_nse
    )

    return {
        "common_support_size": len(common_support),
        "common_support_x_values": sorted(common_support),
        "tie_break_rule": TIE_BREAK_RULE,
        "per_candidate": per_candidate,
        "selected_d": selected_d,
        "selected_config_sha256": per_candidate[selected_d]["config_sha256"],
        "selected_checkpoint_sha256": per_candidate[selected_d]["checkpoint_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="D=JOB_DIR",
        help="One per D candidate, e.g. --candidate 14=path/to/multiscale_selection_2018_D14_s42",
    )
    parser.add_argument("--approach", default="lstm_multiscale")
    args = parser.parse_args(argv)

    candidates: Dict[int, str] = {}
    for item in args.candidate:
        d_str, _, path = item.partition("=")
        if not path:
            parser.error(f"--candidate must be D=JOB_DIR, got {item!r}")
        candidates[int(d_str)] = path

    report = select_history_length(candidates, approach=args.approach)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
