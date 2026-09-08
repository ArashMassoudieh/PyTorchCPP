#!/usr/bin/env python3
"""Summarize a HydroPINN full-tuning pipeline without selecting on test metrics.

Usage:
    python3 select_pipeline_candidates.py <full_pipeline_output_directory>

The selector uses validation_mse for candidate selection. Test metrics are copied
into the report for diagnostics only; they are never used to rank candidates.
This is intentional because repeated test-set inspection during tuning would
invalidate the final held-out evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

STAGES = (
    ("stage1", "01_stage1_supervised"),
    ("stage2", "02_stage2_physics"),
    ("stage3", "03_stage3_optimizer"),
    ("stage4", "04_stage4_robustness"),
)

NUMERIC = {
    "validation_mse",
    "test_mse",
    "rmse",
    "mae",
    "r2",
    "nse",
    "kge",
    "correlation",
    "pbias",
    "physics_loss",
    "physics_residual_rmse",
    "learning_rate",
    "physics_weight",
    "latent_recession_per_hour",
}


def f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(value: float) -> bool:
    return math.isfinite(value)


def load_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if r.get("success", "").lower() == "true"]


def validation_key(row: dict[str, str]):
    value = f(row.get("validation_mse", "nan"))
    return (0 if finite(value) else 1, value if finite(value) else math.inf, row.get("experiment_id", ""))


def config_signature(row: dict[str, str]) -> tuple[str, ...]:
    keys = (
        "mode",
        "hidden_layers",
        "activation",
        "lstm_sequence_length",
        "input_lags",
        "normalization",
        "physics_profile",
        "data_weight",
        "physics_weight",
        "latent_recession_per_hour",
        "learning_rate",
        "batch_size",
    )
    return tuple(row.get(k, "") for k in keys)


def stage_winners(stage: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_mode[row.get("mode", "unknown")].append(row)

    winners = []
    for mode, candidates in sorted(by_mode.items()):
        winner = min(candidates, key=validation_key)
        out = dict(winner)
        out["stage"] = stage
        out["selection_metric"] = "validation_mse"
        winners.append(out)
    return winners


def robust_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[config_signature(row)].append(row)

    out = []
    for signature, members in groups.items():
        template = dict(members[0])
        template["stage"] = "stage4_robustness"
        template["seed_count"] = str(len(members))
        for field in ("validation_mse", "test_mse", "rmse", "mae", "nse", "kge", "pbias", "physics_loss"):
            values = [f(r.get(field, "nan")) for r in members]
            values = [v for v in values if finite(v)]
            template[field + "_mean"] = str(statistics.fmean(values)) if values else "nan"
            template[field + "_std"] = str(statistics.pstdev(values)) if len(values) > 1 else ("0" if values else "nan")
        out.append(template)

    out.sort(key=lambda r: (
        r.get("mode", ""),
        f(r.get("validation_mse_mean", "nan")) if finite(f(r.get("validation_mse_mean", "nan"))) else math.inf,
    ))
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    preferred = [
        "stage", "selection_metric", "experiment_id", "mode", "hidden_layers", "activation",
        "lstm_sequence_length", "input_lags", "normalization", "physics_profile", "data_weight",
        "physics_weight", "latent_recession_per_hour", "learning_rate", "batch_size", "random_seed",
        "validation_mse", "test_mse", "rmse", "mae", "r2", "nse", "kge", "pbias",
        "physics_loss", "physics_residual_rmse",
    ]
    keys = []
    seen = set()
    for key in preferred:
        if any(key in r for r in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_root", type=Path)
    args = parser.parse_args()
    root = args.pipeline_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Pipeline directory does not exist: {root}")

    all_winners: list[dict[str, str]] = []
    stage_counts = {}
    stage4 = []
    for stage, folder in STAGES:
        rows = load_summary(root / folder / "batch_summary.csv")
        stage_counts[stage] = len(rows)
        if stage == "stage4":
            stage4 = robust_rows(rows)
        if rows:
            all_winners.extend(stage_winners(stage, rows))

    if not all_winners:
        raise SystemExit(f"No successful batch summaries found under {root}")

    winners_path = root / "pipeline_validation_winners.csv"
    robust_path = root / "pipeline_robustness_summary.csv"
    report_path = root / "pipeline_selection_report.txt"
    write_csv(winners_path, all_winners)
    if stage4:
        write_csv(robust_path, stage4)

    lines = [
        "HydroPINN full-pipeline selection report",
        "========================================",
        "",
        "Selection rule: minimum validation_mse within each method/stage.",
        "Test metrics are diagnostic only and are NOT used for candidate selection.",
        "",
        "Successful rows by stage: " + ", ".join(f"{k}={v}" for k, v in stage_counts.items()),
        "",
    ]

    by_stage: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_winners:
        by_stage[row["stage"]].append(row)
    for stage, _ in STAGES:
        rows = by_stage.get(stage, [])
        if not rows:
            continue
        lines.append(stage.upper())
        for row in sorted(rows, key=lambda r: r.get("mode", "")):
            lines.append(
                "  {mode}: validation_mse={val}; test_rmse={rmse}; hidden={hidden}; seq={seq}; "
                "w={w}; k={k}; lr={lr}; batch={batch}; seed={seed}".format(
                    mode=row.get("mode", ""),
                    val=row.get("validation_mse", ""),
                    rmse=row.get("rmse", ""),
                    hidden=row.get("hidden_layers", ""),
                    seq=row.get("lstm_sequence_length", ""),
                    w=row.get("physics_weight", ""),
                    k=row.get("latent_recession_per_hour", ""),
                    lr=row.get("learning_rate", ""),
                    batch=row.get("batch_size", ""),
                    seed=row.get("random_seed", ""),
                )
            )
        lines.append("")

    if stage4:
        lines.append("STAGE4 ROBUSTNESS (grouped by configuration)")
        for row in stage4:
            lines.append(
                f"  {row.get('mode','')}: seeds={row.get('seed_count','')}; "
                f"validation_mse mean±sd={row.get('validation_mse_mean','')} ± {row.get('validation_mse_std','')}; "
                f"RMSE mean±sd={row.get('rmse_mean','')} ± {row.get('rmse_std','')}"
            )
        lines.append("")

    lines += [
        "Important:",
        "  The current one-click pipeline uses fixed Stage-3/Stage-4 descendant settings.",
        "  Therefore this report identifies what each stage actually preferred and makes",
        "  any mismatch between earlier winners and later fixed settings explicit.",
        "  A final publication run should use a fresh untouched holdout after configuration",
        "  choices are frozen.",
        "",
        f"CSV winners: {winners_path}",
        f"Robustness summary: {robust_path if stage4 else 'not available'}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
