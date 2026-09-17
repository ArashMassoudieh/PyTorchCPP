#!/usr/bin/env python3
"""Preflight rolling-origin folds on a completed real-data HydroPINN run.

This is deliberately diagnostic-only: it reads observed values from one
representative Stage-4 predictions.csv, constructs leakage-safe expanding
chronological folds, and reports hydrologic regime coverage before any rolling
CV model training is attempted. It never reads predicted values for fold
construction or selection.

Default folds use train endpoints at 50, 60, 70, and 80 percent of the common
record, followed by contiguous 10-percent validation and 10-percent test
blocks. Global Q90/Q95 thresholds are computed once from the full observed
record only for descriptive regime diagnostics; they are not tuning inputs.
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
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    f = pos - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def locate_predictions(real_root: Path) -> Path:
    preferred = sorted((real_root / "04_stage4_robustness" / "lstm_pinn").glob("*/predictions.csv"))
    if preferred:
        return preferred[0]
    candidates = sorted(real_root.glob("04_stage4_robustness/*/*/predictions.csv"))
    if not candidates:
        raise FileNotFoundError("No Stage-4 predictions.csv found")
    return candidates[0]


def load_observations(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row["index"])
                x = float(row["x"])
                obs = float(row["observed"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(obs):
                rows.append({"index": idx, "x": x, "observed": obs})
    # predictions.csv contains one row per sample for a single approach. Guard
    # against accidental duplicates and preserve chronological exported order.
    seen: set[int] = set()
    unique = []
    for r in rows:
        idx = int(r["index"])
        if idx not in seen:
            unique.append(r); seen.add(idx)
    if not unique:
        raise RuntimeError("No finite observations found")
    return unique


def basic(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan, math.nan
    return (statistics.fmean(values),
            statistics.pstdev(values) if len(values) > 1 else 0.0,
            min(values), max(values))


def event_count(values: list[float], threshold: float) -> int:
    """Count contiguous excursions at/above threshold."""
    count = 0
    active = False
    for value in values:
        above = value >= threshold
        if above and not active:
            count += 1
        active = above
    return count


def fmt(v: float) -> str:
    return f"{v:.12g}" if math.isfinite(v) else "nan"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("real_root", type=Path, help="Completed real-data paper-run directory")
    p.add_argument("--train-end-fractions", default="0.50,0.60,0.70,0.80")
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.10)
    a = p.parse_args()

    real = a.real_root.resolve()
    pred = locate_predictions(real)
    data = load_observations(pred)
    n = len(data)
    obs_all = [float(r["observed"]) for r in data]
    q90, q95 = quantile(obs_all, .90), quantile(obs_all, .95)
    train_ends = [float(v.strip()) for v in a.train_end_fractions.split(",") if v.strip()]
    if not train_ends or any(v <= 0 or v >= 1 for v in train_ends):
        raise SystemExit("train-end fractions must lie strictly between 0 and 1")
    if a.validation_fraction <= 0 or a.test_fraction <= 0:
        raise SystemExit("validation/test fractions must be positive")

    rows: list[dict[str, str | int]] = []
    fold_summary: list[dict[str, float | int | str]] = []
    block = max(1, int(round(n * a.validation_fraction)))
    test_block = max(1, int(round(n * a.test_fraction)))

    for fold_no, frac in enumerate(train_ends, 1):
        train_end = int(round(n * frac))
        val_end = train_end + block
        test_end = val_end + test_block
        if test_end > n:
            raise SystemExit(f"fold {fold_no} exceeds record: train_end_fraction={frac}")
        bounds = (("train", 0, train_end), ("validation", train_end, val_end), ("test", val_end, test_end))
        split_stats = {}
        for split, start, end in bounds:
            part = data[start:end]
            vals = [float(r["observed"]) for r in part]
            mean, std, vmin, vmax = basic(vals)
            above90 = sum(v >= q90 for v in vals)
            above95 = sum(v >= q95 for v in vals)
            split_stats[split] = (mean, std)
            rows.append({
                "fold": fold_no, "train_end_fraction": fmt(frac), "split": split,
                "start_position": start, "end_position_exclusive": end, "count": len(vals),
                "start_index": int(part[0]["index"]), "end_index": int(part[-1]["index"]),
                "start_x": fmt(float(part[0]["x"])), "end_x": fmt(float(part[-1]["x"])),
                "mean_flow": fmt(mean), "std_flow": fmt(std), "min_flow": fmt(vmin), "max_flow": fmt(vmax),
                "global_q90": fmt(q90), "global_q95": fmt(q95),
                "count_above_global_q90": above90, "fraction_above_global_q90": fmt(above90 / len(vals)),
                "count_above_global_q95": above95, "fraction_above_global_q95": fmt(above95 / len(vals)),
                "q90_event_count": event_count(vals, q90), "q95_event_count": event_count(vals, q95),
            })
        tr_mean, tr_std = split_stats["train"]
        te_mean, te_std = split_stats["test"]
        fold_summary.append({
            "fold": fold_no, "train_end_fraction": frac,
            "test_train_mean_ratio": te_mean / tr_mean if abs(tr_mean) > 1e-15 else math.nan,
            "test_train_std_ratio": te_std / tr_std if abs(tr_std) > 1e-15 else math.nan,
        })

    out_csv = real / "paper_rolling_fold_diagnostics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    lines = [
        "# Rolling-origin hydrologic fold preflight", "",
        f"Representative observations: `{pred}`", "",
        f"Common record length: **{n}** samples. Global Q90 = **{q90:.6g}**; global Q95 = **{q95:.6g}**.", "",
        "These are diagnostic-only expanding chronological folds. No predictions or test metrics are used to construct or select folds.", "",
        "| Fold | Train end | Test/train mean | Test/train std | Test Q90 events | Test Q95 events |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for fs in fold_summary:
        fold = int(fs["fold"])
        test_row = next(r for r in rows if int(r["fold"]) == fold and r["split"] == "test")
        lines.append(
            f"| {fold} | {float(fs['train_end_fraction']):.0%} | "
            f"{float(fs['test_train_mean_ratio']):.3g} | {float(fs['test_train_std_ratio']):.3g} | "
            f"{test_row['q90_event_count']} | {test_row['q95_event_count']} |"
        )
    lines += ["", "See `paper_rolling_fold_diagnostics.csv` for all train/validation/test statistics.", "",
              "`x` is reported exactly as exported by HydroBatch; this diagnostic does not assume it is a calendar timestamp."]
    out_md = real / "paper_rolling_fold_diagnostics.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Optional publication-style diagnostic plot. CSV/Markdown remain the
    # authoritative outputs if matplotlib is unavailable.
    try:
        import matplotlib.pyplot as plt
        xs = list(range(n))
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(xs, obs_all, linewidth=0.8, label="Observed flow")
        ax.axhline(q90, linestyle="--", linewidth=0.8, label="Global Q90")
        ax.axhline(q95, linestyle=":", linewidth=0.8, label="Global Q95")
        for fold_no, frac in enumerate(train_ends, 1):
            train_end = int(round(n * frac)); val_end = train_end + block; test_end = val_end + test_block
            ax.axvspan(val_end, test_end, alpha=0.08)
            ax.text((val_end + test_end) / 2, ax.get_ylim()[1], f"T{fold_no}", ha="center", va="top", fontsize=8)
        ax.set_xlabel("Chronological sample position")
        ax.set_ylabel("Observed flow")
        ax.set_title("Rolling-origin test windows and observed-flow regime")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out_png = real / "paper_rolling_fold_hydrology.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[rolling-preflight] wrote {out_png}")
    except ImportError:
        print("[rolling-preflight] matplotlib unavailable; skipped PNG")

    print(f"[rolling-preflight] wrote {out_csv}")
    print(f"[rolling-preflight] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
