#!/usr/bin/env python3
"""Add explicit metric-definition diagnostics to a completed adaptive paper run.

KGE and Pearson R^2 are mathematically undefined when Pearson correlation is
undefined, most commonly because a model's held-out predictions are constant
(zero variance). This script inspects Stage-4 per-seed batch metrics and
predictions, then adds explicit *_defined_seed_count / *_status fields plus
Pearson-R^2 robustness statistics to the paper summaries.

HydroPINN defines R^2 for paper reporting as squared Pearson correlation, r^2.
NSE remains the distinct hydrologic 1-SSE/SST efficiency metric. This script
also repairs legacy batch rows where the old r2 column duplicated NSE by
recomputing R^2 from the exported Pearson correlation. It does not alter model
predictions or any other finite metric values.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: str | None) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_value(value: str | None) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.nan
    except (TypeError, ValueError):
        return math.nan


def prediction_std(path: Path) -> float:
    if not path.exists():
        return math.nan
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "test":
                continue
            try:
                value = float(row["predicted"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    if len(values) < 2:
        return math.nan
    return statistics.pstdev(values)


def diagnostic_status(defined: int, total: int, near_constant: int, metric: str) -> str:
    if total <= 0:
        return "not_evaluated"
    if defined == total:
        return "defined_all_seeds"
    if defined == 0 and near_constant == total:
        return f"undefined_all_seeds_constant_prediction_{metric}"
    if defined == 0:
        return f"undefined_all_seeds_{metric}"
    if near_constant > 0:
        return f"partially_defined_some_constant_predictions_{metric}"
    return f"partially_defined_{metric}"


def mean_std(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def process(root: Path) -> None:
    stage4 = root / "04_stage4_robustness"
    summary_path = root / "paper_robustness_summary.csv"
    method_path = root / "paper_method_summary.csv"
    if not summary_path.exists() or not method_path.exists():
        raise SystemExit(f"Missing paper summaries under {root}")

    batch_paths = sorted(stage4.glob("*/batch_summary.csv"))
    if not batch_paths:
        aggregate = stage4 / "batch_summary.csv"
        batch_paths = [aggregate] if aggregate.exists() else []
    batch_rows: list[dict[str, str]] = []
    for path in batch_paths:
        rows = read_rows(path)
        changed = False
        for row in rows:
            corr = finite_value(row.get("correlation"))
            corrected_r2 = corr * corr if math.isfinite(corr) else math.nan
            corrected_text = str(corrected_r2)
            if row.get("r2") != corrected_text:
                row["r2"] = corrected_text
                changed = True
        if changed:
            write_rows(path, rows)
        batch_rows.extend(rows)
    if not batch_rows:
        raise SystemExit(f"No Stage-4 batch summaries found under {stage4}")

    diagnostics: dict[str, dict[str, str]] = {}
    modes = sorted({r.get("mode", "") for r in batch_rows if r.get("mode")})
    for mode in modes:
        members = [r for r in batch_rows if r.get("mode") == mode and r.get("success", "").lower() == "true"]
        total = len(members)
        kge_defined = sum(finite(r.get("kge")) for r in members)
        corr_defined = sum(finite(r.get("correlation")) for r in members)
        r2_values = [finite_value(r.get("r2")) for r in members]
        r2_defined = sum(math.isfinite(v) for v in r2_values)
        r2_mean, r2_std = mean_std(r2_values)

        stds: list[float] = []
        near_constant = 0
        for r in members:
            exp = r.get("experiment_id", "")
            pred_path = stage4 / mode / exp / "predictions.csv"
            s = prediction_std(pred_path)
            stds.append(s)
            if math.isfinite(s) and s <= 1.0e-12:
                near_constant += 1
        finite_stds = [s for s in stds if math.isfinite(s)]
        diagnostics[mode] = {
            "correlation_defined_seed_count": str(corr_defined),
            "pearson_r2_defined_seed_count": str(r2_defined),
            "pearson_r2_mean": str(r2_mean),
            "pearson_r2_std": str(r2_std),
            "pearson_r2_status": diagnostic_status(r2_defined, total, near_constant, "pearson_r2"),
            "kge_defined_seed_count": str(kge_defined),
            "prediction_near_constant_seed_count": str(near_constant),
            "prediction_test_std_mean": str(statistics.fmean(finite_stds)) if finite_stds else "nan",
            "kge_status": diagnostic_status(kge_defined, total, near_constant, "kge"),
            "r2_definition": "squared_pearson_correlation",
            "nse_definition": "1_minus_sse_over_observed_sst",
        }

    for path in (summary_path, method_path):
        rows = read_rows(path)
        for row in rows:
            row.update(diagnostics.get(row.get("mode", ""), {}))
        write_rows(path, rows)

    diag_rows = [{"mode": mode, **values} for mode, values in diagnostics.items()]
    write_rows(root / "paper_metric_definition_diagnostics.csv", diag_rows)
    print(root / "paper_metric_definition_diagnostics.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adaptive_root", type=Path)
    args = parser.parse_args()
    process(args.adaptive_root.resolve())


if __name__ == "__main__":
    main()
