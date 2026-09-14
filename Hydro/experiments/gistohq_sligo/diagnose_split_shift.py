#!/usr/bin/env python3
"""Diagnose temporal distribution shift for a completed real-data paper run.

Uses one representative Stage-4 predictions.csv because all five methods share
one common contiguous observation domain. Writes observed-flow split statistics
and simple event/flow-duration diagnostics without affecting model selection.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    f = pos - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: math.nan for k in ("mean","std","min","q50","q90","q95","q99","max","cv")}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean, "std": std, "min": min(values),
        "q50": quantile(values, .50), "q90": quantile(values, .90),
        "q95": quantile(values, .95), "q99": quantile(values, .99),
        "max": max(values), "cv": std / abs(mean) if abs(mean) > 1e-15 else math.nan,
    }


def locate_predictions(real_root: Path) -> Path:
    preferred = sorted((real_root / "04_stage4_robustness" / "lstm_pinn").glob("*/predictions.csv"))
    if preferred:
        return preferred[0]
    candidates = sorted(real_root.glob("04_stage4_robustness/*/*/predictions.csv"))
    if not candidates:
        raise FileNotFoundError("No Stage-4 predictions.csv found")
    return candidates[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("real_root", type=Path)
    a = p.parse_args()
    real = a.real_root.resolve()
    pred_path = locate_predictions(real)

    groups: dict[str, list[float]] = {"train": [], "validation": [], "test": []}
    with pred_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "")
            if split not in groups:
                continue
            try:
                obs = float(row["observed"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(obs):
                groups[split].append(obs)

    all_values = [v for vals in groups.values() for v in vals]
    if not all_values:
        raise RuntimeError("No finite observations found in representative predictions")
    global_q90 = quantile(all_values, .90)
    global_q95 = quantile(all_values, .95)

    rows = []
    for split in ("train", "validation", "test"):
        vals = groups[split]
        s = stats(vals)
        rows.append({
            "split": split,
            "count": len(vals),
            **{k: f"{v:.12g}" for k, v in s.items()},
            "fraction_above_global_q90": f"{(sum(v >= global_q90 for v in vals) / len(vals)) if vals else math.nan:.12g}",
            "fraction_above_global_q95": f"{(sum(v >= global_q95 for v in vals) / len(vals)) if vals else math.nan:.12g}",
            "global_q90": f"{global_q90:.12g}",
            "global_q95": f"{global_q95:.12g}",
        })

    out_csv = real / "paper_split_shift_diagnostics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    train = stats(groups["train"]); val = stats(groups["validation"]); test = stats(groups["test"])
    def ratio(a: float, b: float) -> float:
        return a / b if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-15 else math.nan

    lines = [
        "# Real-data split-shift diagnostics",
        "",
        f"Representative observations: `{pred_path}`",
        "",
        "All five methods share the same contiguous observation domain; this file is diagnostic only and is not used for tuning.",
        "",
        f"- Test/train mean-flow ratio: {ratio(test['mean'], train['mean']):.4g}",
        f"- Test/train standard-deviation ratio: {ratio(test['std'], train['std']):.4g}",
        f"- Test/train 95th-percentile ratio: {ratio(test['q95'], train['q95']):.4g}",
        f"- Validation/train 95th-percentile ratio: {ratio(val['q95'], train['q95']):.4g}",
        f"- Global observed Q90: {global_q90:.6g}",
        f"- Global observed Q95: {global_q95:.6g}",
        "",
        "See `paper_split_shift_diagnostics.csv` for complete split statistics.",
    ]
    (real / "paper_split_shift_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[split-diagnostics] wrote {out_csv}")
    print(f"[split-diagnostics] wrote {real / 'paper_split_shift_diagnostics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
