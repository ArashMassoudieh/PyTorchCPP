#!/usr/bin/env python3
"""Task 4 completion: select between 'mean-of-seed-NSE' and 'ensemble-of-predictions-NSE'
using each fold's own VALIDATION predictions only, then apply the validation-selected rule
to that fold's TEST predictions once, reported as exploratory.

This is the one remaining step report 90 named as not yet done: reports 43/52 already compute
the ensemble (median-of-predictions) metric directly on TEST data, by standing convention, not
because a validation-only comparison chose it over the mean-of-seed alternative. This script
performs that comparison properly: the choice of which rule to trust is made from validation
data alone, before any test value for that fold is read for selection purposes.

No training. No weight search. No recipe mixing. No reserved-period access (asserted at runtime
before any value is read, not just by convention). Existing same-recipe seed groups only -- the
frozen 36-job six-year corrected-forcing batch (12 recipe-year groups x 3 seeds).

Definitions, stated precisely per this program's own established estimand-precision requirement:
  - "mean-of-seed-NSE": each seed's own NSE computed independently on its own rows, then the
    arithmetic mean of the three NSE values.
  - "ensemble-of-predictions-NSE": the per-x median of the three seeds' PREDICTED values, merged
    into one series, scored once against the (verified-identical-across-seeds) observed series.
  These are not interchangeable (NSE is nonlinear in the prediction series) -- this program's own
  report 52 already established this distinction; this script enforces it by construction rather
  than conflating the two anywhere in the computation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

RESERVED_PERIOD_X_CUTOFF = 77016


class EnsembleRuleSelectionError(ValueError):
    pass


def read_split_rows(predictions_csv: Path, approach: str, split: str) -> Dict[float, Tuple[float, float]]:
    """Return {x: (observed, predicted)} for one approach's rows in exactly one split.
    Asserts the reserved-period seal and finiteness before any value is trusted, matching this
    program's established pattern (select_multiscale_history_length.py)."""
    rows: Dict[float, Tuple[float, float]] = {}
    with open(predictions_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["approach"] != approach or row["split"] != split:
                continue
            x = float(row["x"])
            if not math.isfinite(x):
                raise EnsembleRuleSelectionError(f"Non-finite x in {predictions_csv}")
            if x >= RESERVED_PERIOD_X_CUTOFF:
                raise EnsembleRuleSelectionError(
                    f"x={x} is at or past the reserved-period seal in {predictions_csv} -- refusing"
                )
            observed = float(row["observed"])
            predicted = float(row["predicted"])
            if not (math.isfinite(observed) and math.isfinite(predicted)):
                raise EnsembleRuleSelectionError(f"Non-finite value at x={x} in {predictions_csv}")
            if x in rows:
                raise EnsembleRuleSelectionError(f"Duplicate x={x} for split {split!r} in {predictions_csv}")
            rows[x] = (observed, predicted)
    if not rows:
        raise EnsembleRuleSelectionError(f"No {split!r} rows found for approach {approach!r} in {predictions_csv}")
    return rows


def nse(observed: List[float], predicted: List[float]) -> float:
    if len(observed) != len(predicted) or not observed:
        raise EnsembleRuleSelectionError("NSE requires matching, non-empty series")
    mean_obs = sum(observed) / len(observed)
    denom = sum((o - mean_obs) ** 2 for o in observed)
    if denom <= 0.0:
        raise EnsembleRuleSelectionError("NSE denominator is zero or negative")
    numer = sum((o - p) ** 2 for o, p in zip(observed, predicted))
    return 1.0 - numer / denom


def _assert_cross_seed_observed_agreement(per_seed_rows: Dict[str, Dict[float, Tuple[float, float]]],
                                           label: str) -> None:
    """All seeds must agree on the observed value at every x they share -- the same corrupted/
    mismatched-target-series check already applied to D-selection, applied here to seed-ensemble
    construction for the same reason."""
    observed_by_x: Dict[float, Dict[str, float]] = {}
    for seed, rows in per_seed_rows.items():
        for x, (observed, _predicted) in rows.items():
            observed_by_x.setdefault(x, {})[seed] = observed
    for x, per_seed_observed in observed_by_x.items():
        values = list(per_seed_observed.values())
        if max(values) - min(values) > 1e-6:
            raise EnsembleRuleSelectionError(
                f"{label}: seeds disagree on observed value at shared x={x}: {per_seed_observed}"
            )


def _common_support(per_seed_rows: Dict[str, Dict[float, Tuple[float, float]]]) -> List[float]:
    sets = [set(rows.keys()) for rows in per_seed_rows.values()]
    common = set.intersection(*sets)
    if not common:
        raise EnsembleRuleSelectionError("Empty common support across seeds")
    return sorted(common)


def mean_of_seed_nse(per_seed_rows: Dict[str, Dict[float, Tuple[float, float]]], support: List[float]) -> float:
    per_seed = []
    for rows in per_seed_rows.values():
        observed = [rows[x][0] for x in support]
        predicted = [rows[x][1] for x in support]
        per_seed.append(nse(observed, predicted))
    return sum(per_seed) / len(per_seed)


def ensemble_of_predictions_nse(per_seed_rows: Dict[str, Dict[float, Tuple[float, float]]],
                                 support: List[float]) -> float:
    observed = [next(iter(per_seed_rows.values()))[x][0] for x in support]
    predicted = []
    for x in support:
        preds = sorted(rows[x][1] for rows in per_seed_rows.values())
        predicted.append(statistics.median(preds))
    return nse(observed, predicted)


def analyze_group(job_dirs: Dict[str, Path], *, approach: str) -> dict:
    """job_dirs: {seed_label: job_dir} for one recipe-year group's 3 same-recipe seeds."""
    val_rows = {seed: read_split_rows(jd / "predictions.csv", approach, "validation")
                for seed, jd in job_dirs.items()}
    test_rows = {seed: read_split_rows(jd / "predictions.csv", approach, "test")
                 for seed, jd in job_dirs.items()}
    _assert_cross_seed_observed_agreement(val_rows, "validation")
    _assert_cross_seed_observed_agreement(test_rows, "test")

    val_support = _common_support(val_rows)
    test_support = _common_support(test_rows)

    val_mean = mean_of_seed_nse(val_rows, val_support)
    val_ensemble = ensemble_of_predictions_nse(val_rows, val_support)
    selected_rule = "mean_of_seed" if val_mean >= val_ensemble else "ensemble_of_predictions"

    test_mean = mean_of_seed_nse(test_rows, test_support)
    test_ensemble = ensemble_of_predictions_nse(test_rows, test_support)
    selected_test_value = test_mean if selected_rule == "mean_of_seed" else test_ensemble

    return {
        "validation_support_size": len(val_support),
        "test_support_size": len(test_support),
        "validation_mean_of_seed_nse": round(val_mean, 6),
        "validation_ensemble_of_predictions_nse": round(val_ensemble, 6),
        "selected_rule_from_validation": selected_rule,
        "test_mean_of_seed_nse": round(test_mean, 6),
        "test_ensemble_of_predictions_nse": round(test_ensemble, 6),
        "test_value_under_selected_rule_EXPLORATORY": round(selected_test_value, 6),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new report; existing files refused")
    parser.add_argument("--approach", default="lstm_pinn")
    parser.add_argument("--years", default="2018,2019,2020,2021,2022,2023")
    parser.add_argument("--recipes", default="zero,urban_no_detention")
    parser.add_argument("--seeds", default="42,123,27182")
    parser.add_argument("--prefix", default="sixyear_corrected_forcing")
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    years = args.years.split(",")
    recipes = args.recipes.split(",")
    seeds = args.seeds.split(",")

    per_group = {}
    for year in years:
        for recipe in recipes:
            job_dirs = {
                seed: args.batch_dir / f"{args.prefix}_{year}_{recipe}_s{seed}"
                for seed in seeds
            }
            missing = [str(p) for p in job_dirs.values() if not p.is_dir()]
            if missing:
                raise EnsembleRuleSelectionError(f"Missing job directories: {missing}")
            key = f"{year}_{recipe}"
            per_group[key] = analyze_group(job_dirs, approach=args.approach)

    rule_counts = {}
    for entry in per_group.values():
        rule_counts[entry["selected_rule_from_validation"]] = (
            rule_counts.get(entry["selected_rule_from_validation"], 0) + 1
        )

    report = {
        "batch_dir": str(args.batch_dir.resolve()),
        "approach": args.approach,
        "n_groups": len(per_group),
        "rule_selection_counts": rule_counts,
        "per_group": per_group,
        "note": (
            "selected_rule_from_validation is chosen using only each group's own validation "
            "predictions, before any test value is read for selection purposes. "
            "test_value_under_selected_rule_EXPLORATORY applies that frozen choice once to test "
            "predictions and is reported as exploratory, not a confirmatory result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_groups": len(per_group), "rule_selection_counts": rule_counts}, indent=2))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
