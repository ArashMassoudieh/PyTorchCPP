#!/usr/bin/env python3
"""Scores the corrected-forcing six-year batch on the EXACT same
per-year common-native-support x-sets the original analysis used
(six_year_physics_analysis/out_of_sample_*.csv), per the explicit
instruction to use identical native-hour support for comparison --
not a freshly recomputed intersection for the corrected package.

No retraining. Reads each job's existing predictions.csv and looks up
the already-fixed x-set by exact index, mirroring the post-hoc
rescoring approach already validated (reports 38-40) for this exact
class of comparison.
"""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023")
ORIGINAL_ANALYSIS = ROOT / "six_year_physics_analysis"
CORRECTED_RUNS = ROOT / "six_year_physics_runs_corrected_forcing"
OUT_DIR = ROOT / "six_year_physics_analysis_corrected_forcing"
YEARS = ["2018", "2019", "2020", "2021", "2022", "2023"]
ARMS = ["zero", "urban_no_detention"]
SEEDS = [42, 123, 27182]
SEAL_START_X = 77016


def nse(obs, pred):
    mean_obs = sum(obs) / len(obs)
    num = sum((o - p) ** 2 for o, p in zip(obs, pred))
    den = sum((o - mean_obs) ** 2 for o in obs)
    return num and (1 - num / den) if den > 0 else float("nan")


def kge(obs, pred):
    n = len(obs)
    mean_o, mean_p = sum(obs) / n, sum(pred) / n
    var_o = sum((o - mean_o) ** 2 for o in obs) / n
    var_p = sum((p - mean_p) ** 2 for p in pred) / n
    std_o, std_p = math.sqrt(var_o), math.sqrt(var_p)
    cov = sum((o - mean_o) * (p - mean_p) for o, p in zip(obs, pred)) / n
    if std_o == 0 or std_p == 0 or mean_o == 0:
        return float("nan")
    r = cov / (std_o * std_p)
    alpha, beta = std_p / std_o, mean_p / mean_o
    return 1 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pearson_r2(obs, pred):
    n = len(obs)
    mean_o, mean_p = sum(obs) / n, sum(pred) / n
    cov = sum((o - mean_o) * (p - mean_p) for o, p in zip(obs, pred))
    den = math.sqrt(sum((o - mean_o) ** 2 for o in obs) * sum((p - mean_p) ** 2 for p in pred))
    return (cov / den) ** 2 if den > 0 else float("nan")


def rmse(obs, pred):
    return math.sqrt(sum((o - p) ** 2 for o, p in zip(obs, pred)) / len(obs))


def mae(obs, pred):
    return sum(abs(o - p) for o, p in zip(obs, pred)) / len(obs)


def pbias(obs, pred):
    s = sum(obs)
    return 100.0 * sum(p - o for o, p in zip(obs, pred)) / s if s else float("nan")


def load_original_year_xset(model: str, seed: int, year_start_x: int, year_end_x: int) -> dict:
    """Returns {x: observed} for this year's slice of the fixed original common-support x-set."""
    path = ORIGINAL_ANALYSIS / f"out_of_sample_{model}_s{seed}.csv"
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            x = int(float(row["x"]))
            if year_start_x <= x <= year_end_x:
                out[x] = float(row["observed"])
    return out


def load_corrected_predictions(job_dir: Path) -> dict:
    out = {}
    with open(job_dir / "predictions.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["x"])] = (float(row["observed"]), float(row["predicted"]))
    return out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Bound each year's x range generously; load_original_year_xset filters exactly
    # to the fixed common-support x's actually present in the original out_of_sample file.
    # Exact, non-overlapping per-year test windows: (validation_end_x, test_end_x],
    # read directly from the frozen configs (not guessed) -- each year's window
    # starts immediately after the previous year's test_end_x, confirmed by the
    # configs themselves (e.g. 2018 validation_end_x=26268/test_end_x=35063,
    # 2019 validation_end_x=35063/test_end_x=43823 -- adjacent, not overlapping).
    year_bounds = {
        "2018": (26269, 35063), "2019": (35064, 43823), "2020": (43824, 52607),
        "2021": (52608, 61367), "2022": (61368, 70127), "2023": (70128, 77015),
    }

    pooled_rows = []
    annual_rows = []
    coverage = {}
    out_of_sample_files = {}

    for arm in ARMS:
        for seed in SEEDS:
            pooled_obs, pooled_pred = [], []
            oos_rows = []
            for year in YEARS:
                y_start, y_end = year_bounds[year]
                year_xset = load_original_year_xset(arm, seed, y_start, y_end)
                if not year_xset:
                    raise RuntimeError(f"no original common-support x's found for {arm}/{seed}/{year}")

                job_dir = CORRECTED_RUNS / f"sixyear_corrected_forcing_{year}_{arm}_s{seed}"
                corrected = load_corrected_predictions(job_dir)

                missing = [x for x in year_xset if x not in corrected]
                if missing:
                    raise RuntimeError(
                        f"{arm}/s{seed}/{year}: {len(missing)} of {len(year_xset)} original common-support "
                        f"x's are absent from the corrected-forcing export -- cannot score on identical support"
                    )
                if max(year_xset) >= SEAL_START_X:
                    raise RuntimeError(f"refusing: {year} common-support x reaches sealed period")

                obs_mismatch = [x for x in year_xset if corrected[x][0] != year_xset[x]]
                if obs_mismatch:
                    raise RuntimeError(
                        f"{arm}/s{seed}/{year}: observed discharge differs from the original run at "
                        f"{len(obs_mismatch)} of {len(year_xset)} x's -- target should be byte-identical"
                    )

                obs = [year_xset[x] for x in sorted(year_xset)]
                pred = [corrected[x][1] for x in sorted(year_xset)]
                n = len(obs)

                annual_rows.append({
                    "year": year, "model": arm, "seed": seed, "n": n,
                    "nse": nse(obs, pred), "kge": kge(obs, pred), "pearson_r2": pearson_r2(obs, pred),
                    "rmse": rmse(obs, pred), "mae": mae(obs, pred), "pbias": pbias(obs, pred),
                })
                coverage[year] = {"common_hours": n}

                for x in sorted(year_xset):
                    oos_rows.append({"x": x, "observed": corrected[x][0], "predicted": corrected[x][1]})
                pooled_obs.extend(obs)
                pooled_pred.extend(pred)

            pooled_rows.append({
                "model": arm, "seed": seed, "n": len(pooled_obs),
                "nse": nse(pooled_obs, pooled_pred), "kge": kge(pooled_obs, pooled_pred),
                "pearson_r2": pearson_r2(pooled_obs, pooled_pred),
                "rmse": rmse(pooled_obs, pooled_pred), "mae": mae(pooled_obs, pooled_pred),
                "pbias": pbias(pooled_obs, pooled_pred),
            })
            oos_path = OUT_DIR / f"out_of_sample_{arm}_s{seed}.csv"
            with oos_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["x", "observed", "predicted"])
                w.writeheader()
                w.writerows(oos_rows)
            out_of_sample_files[f"{arm}_s{seed}"] = str(oos_path)
            print(f"{arm} s{seed}: pooled n={len(pooled_obs)} nse={nse(pooled_obs, pooled_pred):.6f}")

    with (OUT_DIR / "pooled_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "seed", "n", "nse", "kge", "pearson_r2", "rmse", "mae", "pbias"])
        w.writeheader()
        w.writerows(pooled_rows)
    with (OUT_DIR / "annual_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["year", "model", "seed", "n", "nse", "kge", "pearson_r2", "rmse", "mae", "pbias"])
        w.writeheader()
        w.writerows(annual_rows)
    (OUT_DIR / "coverage_note.json").write_text(json.dumps({
        "method": "Scored on the IDENTICAL per-year common-support x-sets already used by the "
                  "original six_year_physics_analysis (not recomputed for the corrected package), "
                  "per explicit instruction. Integrity enforced: raises if any original x is absent "
                  "from the corrected export, or if observed discharge differs at any shared x.",
        "per_year_n": {y: coverage[y]["common_hours"] for y in YEARS},
    }, indent=2))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
