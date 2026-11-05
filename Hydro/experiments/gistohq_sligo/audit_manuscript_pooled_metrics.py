#!/usr/bin/env python3
"""Independent, reproducible audit of manuscript_v2_proposed's pooled five-configuration
comparison (improvement_loop_20261023 report 83, hardened per report 84 after Codex's
checkpoint-7 review found report 83's version was ephemeral command history, not a durable
artifact).

What this checks, and in what order (per the standing instruction to assert seal/split
limits BEFORE observing targets):
  1. Reads only the x (timestamp) column from all 13 out_of_sample_<model>_<seed>.csv files
     first. Asserts every x is < RESERVED_PERIOD_X_CUTOFF (77016) before any observed/
     predicted value is ever read from any file -- the reserved-period check happens before
     any target value is inspected, not after.
  2. Checks for duplicate x within each file (would indicate a corrupted export).
  3. Checks that the x-sets across all 13 files are IDENTICAL SETS, not merely equal counts
     -- equal row counts alone do not prove identical support, since two files could have the
     same count of different hours.
  4. Only after 1-3 pass does it read observed/predicted and compute metrics.

What "pooled NSE" means here, stated precisely because it is easy to conflate with a
different quantity: for one (model, seed), NSE is computed once over that seed's own full
49,478-hour development-period common-support series (pooled across years/folds within that
seed, not across seeds). The manuscript's single headline number per model (e.g., FFN =
0.4344) is the ARITHMETIC MEAN of the three seeds' own, independently computed NSE values
-- e.g. mean(0.4279, 0.4436, 0.4316) = 0.4344 -- NOT the NSE of an ensemble prediction (the
per-hour median of the three seeds' predicted series, scored once), which this program's own
report 52 already established is a materially different, not-interchangeable quantity and
which this script does not compute.

RMSE aggregation is stated explicitly too, because "mean of per-seed RMSE" and "RMSE computed
on all three seeds' rows pooled together into one 148,434-row set" are different numbers.
This script reports BOTH, labeled, so the manuscript's reported value (mean-of-per-seed-RMSE)
is not confused with the alternative.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

RESERVED_PERIOD_X_CUTOFF = 77016

SOURCE_DIR = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/annual_analysis_v2"
)

MODELS = {
    "two_reservoir": ["s42"],
    "ffn": ["s42", "s123", "s27182"],
    "ffn_residual_hybrid": ["s42", "s123", "s27182"],
    "lstm_24h": ["s42", "s123", "s27182"],
    "lstm_residual_hybrid": ["s42", "s123", "s27182"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_x_only(path: Path) -> list[float]:
    """Pass 1: read only x. No observed/predicted value is touched here."""
    xs = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["x"]))
    return xs


def read_full(path: Path) -> list[tuple[float, float, float]]:
    """Pass 2: read x, observed, predicted. Only called after the seal/duplicate/
    set-identity checks on x have already passed for every file."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["x"]), float(row["observed"]), float(row["predicted"])))
    return rows


def main() -> int:
    file_list: list[tuple[str, str, Path]] = []
    for model, seeds in MODELS.items():
        for seed in seeds:
            path = SOURCE_DIR / f"out_of_sample_{model}_{seed}.csv"
            file_list.append((model, seed, path))

    # --- Pass 1: x-only. Seal, duplicate, and set-identity checks, before any target
    # value is read from any file. ---
    x_by_file: dict[str, list[float]] = {}
    for model, seed, path in file_list:
        key = f"{model}_{seed}"
        xs = read_x_only(path)
        reserved_violations = [x for x in xs if x >= RESERVED_PERIOD_X_CUTOFF]
        if reserved_violations:
            print(f"FAIL: {key} contains {len(reserved_violations)} x >= "
                  f"{RESERVED_PERIOD_X_CUTOFF} (reserved period) -- refusing to proceed.",
                  file=sys.stderr)
            return 2
        if len(xs) != len(set(xs)):
            dup_count = len(xs) - len(set(xs))
            print(f"FAIL: {key} has {dup_count} duplicate x value(s) -- refusing to proceed.",
                  file=sys.stderr)
            return 2
        x_by_file[key] = xs

    reference_key = f"{list(MODELS.keys())[1]}_s42"  # ffn_s42, an arbitrary fixed reference
    reference_set = set(x_by_file[reference_key])
    set_mismatches = {}
    for key, xs in x_by_file.items():
        if set(xs) != reference_set:
            extra = set(xs) - reference_set
            missing = reference_set - set(xs)
            set_mismatches[key] = {"extra_count": len(extra), "missing_count": len(missing)}
    if set_mismatches:
        print(f"FAIL: x-timestamp sets are NOT identical across all 13 files (reference="
              f"{reference_key}, {len(reference_set)} hours). Mismatches: "
              f"{json.dumps(set_mismatches, indent=2)}", file=sys.stderr)
        return 2

    print(f"PASS: all 13 files share an identical {len(reference_set)}-hour x-timestamp set, "
          f"no duplicates, no reserved-period (x>={RESERVED_PERIOD_X_CUTOFF}) values in any file.")

    # --- Pass 2: now safe to read observed/predicted and compute metrics. ---
    per_model: dict[str, dict] = {}
    for model, seeds in MODELS.items():
        per_seed_nse = []
        per_seed_rmse = []
        all_rows_for_pooled_rmse: list[tuple[float, float]] = []  # (observed, predicted) across seeds
        for seed in seeds:
            path = SOURCE_DIR / f"out_of_sample_{model}_{seed}.csv"
            rows = read_full(path)
            obs = [r[1] for r in rows]
            pred = [r[2] for r in rows]
            mean_obs = sum(obs) / len(obs)
            denom = sum((o - mean_obs) ** 2 for o in obs)
            numer = sum((o - p) ** 2 for o, p in zip(obs, pred))
            nse = 1.0 - numer / denom
            rmse = (sum((o - p) ** 2 for o, p in zip(obs, pred)) / len(obs)) ** 0.5
            per_seed_nse.append(nse)
            per_seed_rmse.append(rmse)
            all_rows_for_pooled_rmse.extend(zip(obs, pred))

        mean_of_seed_nse = sum(per_seed_nse) / len(per_seed_nse)
        mean_of_seed_rmse = sum(per_seed_rmse) / len(per_seed_rmse)
        # Alternative RMSE: all seeds' rows concatenated into one set, one RMSE computed once.
        # Reported for disclosure only -- the manuscript's number is mean_of_seed_rmse, not this.
        n_concat = len(all_rows_for_pooled_rmse)
        concat_rmse = (
            sum((o - p) ** 2 for o, p in all_rows_for_pooled_rmse) / n_concat
        ) ** 0.5

        per_model[model] = {
            "seeds": seeds,
            "per_seed_nse": {s: round(v, 6) for s, v in zip(seeds, per_seed_nse)},
            "per_seed_rmse": {s: round(v, 6) for s, v in zip(seeds, per_seed_rmse)},
            "estimand_mean_of_seed_nse": round(mean_of_seed_nse, 6),
            "estimand_mean_of_seed_rmse": round(mean_of_seed_rmse, 6),
            "estimand_NOT_reported_in_manuscript_concat_seeds_rmse_n": n_concat,
            "estimand_NOT_reported_in_manuscript_concat_seeds_rmse": round(concat_rmse, 6),
            "note": (
                "estimand_mean_of_seed_nse/rmse are the arithmetic mean of each seed's own, "
                "independently computed metric over its full pooled (across years/folds) "
                "49,478-hour common-support series -- this is what the manuscript reports. "
                "NOT an ensemble metric (median-prediction-then-score). The concat_seeds_rmse "
                "field is a different, NOT manuscript-reported quantity, included only to make "
                "the distinction between 'mean of per-seed RMSE' and 'RMSE on all seeds' rows "
                "pooled together' explicit and inspectable."
            ),
        }

    source_hashes = {
        f"{model}_{seed}": sha256_file(SOURCE_DIR / f"out_of_sample_{model}_{seed}.csv")
        for model, seeds in MODELS.items() for seed in seeds
    }

    report = {
        "common_support_hours": len(reference_set),
        "reserved_period_x_cutoff": RESERVED_PERIOD_X_CUTOFF,
        "timestamp_set_identity_verified": True,
        "duplicate_x_check": "none found in any of the 13 files",
        "per_model": per_model,
        "source_file_sha256": source_hashes,
        "source_dir": str(SOURCE_DIR),
    }

    out_path = Path(__file__).resolve().parent / "audit_manuscript_pooled_metrics_output.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    print(json.dumps({m: {"mean_nse": v["estimand_mean_of_seed_nse"],
                           "mean_rmse": v["estimand_mean_of_seed_rmse"]}
                       for m, v in per_model.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
