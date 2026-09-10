#!/usr/bin/env python3
"""Assess whether physics-informed hybrids genuinely improve their parents.

This is a post-hoc reporting diagnostic only. It reads frozen Stage-4 robustness
summaries after tuning is complete and never participates in model selection.
A hybrid is labelled a genuine hydrologic improvement only when it improves
mean RMSE, mean NSE, and absolute mean PBIAS relative to its supervised parent
and has no near-constant prediction seeds.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

PAIRS = (("ffn", "ffn_pinn"), ("lstm", "lstm_pinn"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError):
        return math.nan


def diagnostics(root: Path) -> dict[str, dict[str, str]]:
    path = root / "paper_metric_definition_diagnostics.csv"
    if not path.exists():
        return {}
    return {r.get("mode", ""): r for r in read_rows(path)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paper_run_root", type=Path)
    a = p.parse_args()
    root = a.paper_run_root.resolve()
    real = root / "02_sligo_hydro"
    summary_path = real / "paper_method_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing real-data paper summary: {summary_path}")

    rows = {r.get("mode", ""): r for r in read_rows(summary_path)}
    diag = diagnostics(real)
    out_rows: list[dict[str, object]] = []

    for parent_mode, hybrid_mode in PAIRS:
        if parent_mode not in rows or hybrid_mode not in rows:
            continue
        parent = rows[parent_mode]
        hybrid = rows[hybrid_mode]
        p_rmse, h_rmse = f(parent, "rmse_mean"), f(hybrid, "rmse_mean")
        p_nse, h_nse = f(parent, "nse_mean"), f(hybrid, "nse_mean")
        p_pbias, h_pbias = f(parent, "pbias_mean"), f(hybrid, "pbias_mean")
        p_kge, h_kge = f(parent, "kge_mean"), f(hybrid, "kge_mean")

        rmse_better = math.isfinite(p_rmse) and math.isfinite(h_rmse) and h_rmse < p_rmse
        nse_better = math.isfinite(p_nse) and math.isfinite(h_nse) and h_nse > p_nse
        pbias_better = math.isfinite(p_pbias) and math.isfinite(h_pbias) and abs(h_pbias) < abs(p_pbias)
        kge_better = math.isfinite(p_kge) and math.isfinite(h_kge) and h_kge > p_kge

        d = diag.get(hybrid_mode, {})
        try:
            near_constant = int(float(d.get("prediction_near_constant_seed_count", "0") or 0))
        except ValueError:
            near_constant = 0
        genuine = rmse_better and nse_better and pbias_better and near_constant == 0

        out_rows.append({
            "parent": parent_mode,
            "hybrid": hybrid_mode,
            "parent_rmse": p_rmse,
            "hybrid_rmse": h_rmse,
            "rmse_better": rmse_better,
            "parent_nse": p_nse,
            "hybrid_nse": h_nse,
            "nse_better": nse_better,
            "parent_pbias": p_pbias,
            "hybrid_pbias": h_pbias,
            "abs_pbias_better": pbias_better,
            "parent_kge": p_kge,
            "hybrid_kge": h_kge,
            "kge_better_when_defined": kge_better,
            "hybrid_near_constant_seed_count": near_constant,
            "genuine_hydrologic_improvement": genuine,
        })

    csv_path = root / "paper_hybrid_gain_assessment.csv"
    md_path = root / "paper_hybrid_gain_assessment.md"
    if not out_rows:
        raise SystemExit("No supervised/hybrid pairs found in paper_method_summary.csv")

    with csv_path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        "# Hybrid gain assessment",
        "",
        "Post-hoc Stage-4 comparison only; these test results are not used for tuning.",
        "",
        "| Parent | Hybrid | RMSE better | NSE better | |PBIAS| better | KGE better* | Near-constant seeds | Genuine hydrologic improvement |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out_rows:
        lines.append(
            f"| {str(r['parent']).upper()} | {str(r['hybrid']).upper().replace('_', '+')} | "
            f"{'yes' if r['rmse_better'] else 'no'} | {'yes' if r['nse_better'] else 'no'} | "
            f"{'yes' if r['abs_pbias_better'] else 'no'} | {'yes' if r['kge_better_when_defined'] else 'no'} | "
            f"{r['hybrid_near_constant_seed_count']} | "
            f"{'YES' if r['genuine_hydrologic_improvement'] else 'NO'} |"
        )
    lines += [
        "",
        "*KGE comparison is reported only when finite for both methods.",
        "",
        "A genuine hydrologic improvement requires lower mean RMSE, higher mean NSE, lower absolute mean PBIAS, and zero near-constant prediction seeds.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[hybrid-assessment] wrote {csv_path}")
    print(f"[hybrid-assessment] wrote {md_path}")
    for r in out_rows:
        print(f"[hybrid-assessment] {r['hybrid']} genuine_improvement={r['genuine_hydrologic_improvement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
