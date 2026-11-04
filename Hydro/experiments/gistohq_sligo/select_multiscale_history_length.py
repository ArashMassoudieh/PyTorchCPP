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

Input: each candidate's frozen job directory's own predictions.csv
(approach,index,split,x,observed,predicted,residual), already produced by
training -- no retraining, no new metric computed inside the training loop.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple


class HistoryLengthSelectionError(ValueError):
    """Raised when the inputs do not support a valid common-support selection."""


def read_validation_rows(predictions_csv: str | Path, approach: str) -> Dict[float, Tuple[float, float]]:
    """Return {x: (observed, predicted)} for one approach's validation split."""
    rows: Dict[float, Tuple[float, float]] = {}
    with open(predictions_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["approach"] != approach or row["split"] != "validation":
                continue
            x = float(row["x"])
            if x in rows:
                raise HistoryLengthSelectionError(
                    f"Duplicate validation x={x} for approach {approach!r} in {predictions_csv}"
                )
            rows[x] = (float(row["observed"]), float(row["predicted"]))
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
    candidate_predictions: Dict[int, str | Path],
    *,
    approach: str = "lstm_multiscale",
) -> dict:
    """Pick the winning D for one fold, scored on common validation support.

    ``candidate_predictions`` maps each D candidate (e.g. 14/30/60) to that
    job's ``predictions.csv`` path. Returns a report with each candidate's
    coverage and common-support NSE, and the selected D. Ties are broken
    toward the smaller D (the simpler model), never arbitrarily.
    """
    if len(candidate_predictions) < 2:
        raise HistoryLengthSelectionError("select_history_length requires at least two D candidates")

    per_d_rows = {
        d: read_validation_rows(path, approach) for d, path in candidate_predictions.items()
    }
    per_d_xsets = {d: set(rows.keys()) for d, rows in per_d_rows.items()}

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
        total = len(rows)
        observed = [rows[x][0] for x in common_support]
        predicted = [rows[x][1] for x in common_support]
        per_candidate[d] = {
            "total_eligible_validation_windows": total,
            "common_support_windows": len(common_support),
            "coverage_fraction": len(common_support) / total if total else 0.0,
            "common_support_nse": nse(observed, predicted),
        }

    best_nse = max(entry["common_support_nse"] for entry in per_candidate.values())
    selected_d = min(
        d for d, entry in per_candidate.items() if entry["common_support_nse"] == best_nse
    )

    return {
        "common_support_size": len(common_support),
        "per_candidate": per_candidate,
        "selected_d": selected_d,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="D=PATH",
        help="One per D candidate, e.g. --candidate 14=path/to/predictions.csv",
    )
    parser.add_argument("--approach", default="lstm_multiscale")
    args = parser.parse_args(argv)

    candidates: Dict[int, str] = {}
    for item in args.candidate:
        d_str, _, path = item.partition("=")
        if not path:
            parser.error(f"--candidate must be D=PATH, got {item!r}")
        candidates[int(d_str)] = path

    report = select_history_length(candidates, approach=args.approach)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
