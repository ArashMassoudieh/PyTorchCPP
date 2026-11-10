#!/usr/bin/env python3
"""Grouped (per-channel), EXACT Shapley-value feature attribution for a frozen,
already-trained checkpoint -- the SHAP analysis named as not-yet-implemented in
report 92 / CLAUDE_DIAGNOSTIC_FIGURES_TASKS_20261105.md.

Methodology, stated precisely:
  - "Players" are the 6 stable-order input channels (P, T, RH, wind, solar, PET), each
    spanning the full 24-hour input window -- a GROUPED/temporal-block attribution, not
    independent-lag shuffling of 144 individual scalars, per the task spec's own
    preference (coarser groups are more interpretable and avoid implying hour-by-hour
    causal claims this single-reference-baseline design cannot support).
  - A single frozen BASELINE window is the per-hour, per-channel MEAN over a bounded,
    seed-frozen sample of real development (train-split) windows -- an interventional-
    SHAP / expected-gradients-style reference, not the full background distribution
    marginalized per-coalition (that would need background_size x 64 forward passes per
    explained instance; a single mean reference keeps the total forward-pass count
    bounded while still being a real, data-derived baseline rather than zeros).
  - With only 6 players, Shapley values are computed EXACTLY (64 coalitions per
    explained window, enumerated in full) -- not approximated by Monte Carlo sampling.
    Exact additivity (sum of the 6 channel attributions equals
    model(real window) - model(baseline window)) is verified numerically for every
    explained instance, not assumed.
  - Explanation instances are drawn only from development, pre-reserved-period test-
    split hours (x < 77016), capped at --n-explain (default 128) with a frozen seed.
  - No training. No gradient computation (the CLI bridge is forward-only). No causal
    claim: this attributes THIS model's output change relative to ONE baseline choice,
    for the windows explained here -- it does not establish which weather variable
    "matters" to the physical system.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

RESERVED_PERIOD_X_CUTOFF = 77016
N_CHANNELS = 6
CHANNEL_NAMES = ["precip", "temp", "rh", "wind", "solar", "pet"]


class ShapAttributionError(ValueError):
    pass


def read_dumped_series(path: Path) -> Dict[int, dict]:
    """Returns {x (int): {segment_id, features: [6 floats], target, target_valid}}."""
    rows: Dict[int, dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            x = int(round(float(row["x"])))
            if x >= RESERVED_PERIOD_X_CUTOFF:
                raise ShapAttributionError(f"x={x} at/past reserved-period seal in {path}")
            if x in rows:
                raise ShapAttributionError(f"Duplicate x={x} in {path}")
            features = [float(row[f"feature{i}"]) for i in range(N_CHANNELS)]
            if not all(math.isfinite(v) for v in features):
                raise ShapAttributionError(f"Non-finite feature at x={x} in {path}")
            rows[x] = dict(
                segment_id=int(row["segment_id"]),
                features=features,
                target=float(row["target"]),
                target_valid=row["target_valid"] == "1",
            )
    if not rows:
        raise ShapAttributionError(f"No rows read from {path}")
    return rows


def build_window(series: Dict[int, dict], end_x: int, sequence_length: int = 24) -> List[List[float]]:
    """24 consecutive same-segment rows ending at end_x (inclusive), oldest first."""
    xs = list(range(end_x - sequence_length + 1, end_x + 1))
    missing = [x for x in xs if x not in series]
    if missing:
        raise ShapAttributionError(f"Window ending at x={end_x} is missing x={missing}")
    segment_ids = {series[x]["segment_id"] for x in xs}
    if len(segment_ids) != 1:
        raise ShapAttributionError(f"Window ending at x={end_x} crosses a segment boundary")
    return [series[x]["features"] for x in xs]


def read_test_split_targets(predictions_csv: Path, approach: str) -> List[Tuple[int, float, float]]:
    """Returns [(x, observed, predicted)] for split=='test' rows, x < reserved seal."""
    rows = []
    with predictions_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["approach"] != approach or row["split"] != "test":
                continue
            x = int(round(float(row["x"])))
            if x >= RESERVED_PERIOD_X_CUTOFF:
                continue
            rows.append((x, float(row["observed"]), float(row["predicted"])))
    if not rows:
        raise ShapAttributionError(f"No test-split rows found in {predictions_csv}")
    return rows


def _write_predict_input(path: Path, rows: Sequence[Tuple[str, List[List[float]]]]) -> None:
    with path.open("w", newline="") as f:
        header = ["row_id"] + [f"h{h}f{c}" for h in range(24) for c in range(N_CHANNELS)]
        writer = csv.writer(f)
        writer.writerow(header)
        for row_id, window in rows:
            flat = [v for hour in window for v in hour]
            if len(flat) != 24 * N_CHANNELS:
                raise ShapAttributionError(f"Window for {row_id} is not 24x{N_CHANNELS}")
            writer.writerow([row_id] + flat)


def run_cli_predict(cli: Path, experiment_dir: Path, approach: str,
                     rows: Sequence[Tuple[str, List[List[float]]]]) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.csv"
        output_path = Path(tmp) / "output.csv"
        _write_predict_input(input_path, rows)
        result = subprocess.run(
            [str(cli), "predict", "--experiment-dir", str(experiment_dir), "--approach", approach,
             "--input", str(input_path), "--output", str(output_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ShapAttributionError(f"CLI predict failed: {result.stderr}")
        predictions: Dict[str, float] = {}
        with output_path.open(newline="") as f:
            for row in csv.DictReader(f):
                predictions[row["row_id"]] = float(row["prediction"])
    return predictions


def apply_mask(real_window: List[List[float]], baseline_window: List[List[float]],
               mask: int) -> List[List[float]]:
    """mask: bit i set => channel i uses the REAL value at every hour; else baseline."""
    hybrid = []
    for hour_idx in range(24):
        hour = []
        for c in range(N_CHANNELS):
            use_real = (mask >> c) & 1
            hour.append(real_window[hour_idx][c] if use_real else baseline_window[hour_idx][c])
        hybrid.append(hour)
    return hybrid


def exact_shapley_values(v: Dict[int, float]) -> List[float]:
    """v: {mask (0..63): model output for that coalition}. Returns 6 channel attributions.
    Standard exact Shapley formula: phi_i = sum_{S subset F\\{i}} w(|S|) * (v(S+i) - v(S)),
    w(|S|) = |S|!(n-|S|-1)!/n!."""
    n = N_CHANNELS
    factorial = math.factorial
    phi = [0.0] * n
    others = list(range(n))
    for i in range(n):
        rest = [c for c in others if c != i]
        for s_size in range(0, n):
            weight = factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
            for subset in combinations(rest, s_size):
                mask_without = 0
                for c in subset:
                    mask_without |= (1 << c)
                mask_with = mask_without | (1 << i)
                phi[i] += weight * (v[mask_with] - v[mask_without])
    return phi


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli", type=Path, required=True)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--approach", default="lstm")
    p.add_argument("--dumped-series", type=Path, required=True,
                   help="output of `<cli> dump` for this checkpoint")
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--n-background", type=int, default=128)
    p.add_argument("--n-explain", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    series = read_dumped_series(args.dumped_series)
    test_rows = read_test_split_targets(args.predictions_csv, args.approach)
    rng = random.Random(args.seed)

    # Background: frozen random sample of real windows ending somewhere in the whole
    # dumped (development-only) series, far enough from its start to have a full window.
    candidate_xs = sorted(x for x in series if x - 23 >= min(series))
    candidate_xs = [x for x in candidate_xs if x - 23 in series]
    background_xs = rng.sample(candidate_xs, min(args.n_background, len(candidate_xs)))
    background_windows = [build_window(series, x) for x in background_xs]
    baseline_window = [
        [statistics.mean(w[h][c] for w in background_windows) for c in range(N_CHANNELS)]
        for h in range(24)
    ]

    # Explanation set: frozen random sample of real test-split hours.
    explain_xs = rng.sample([x for x, _, _ in test_rows], min(args.n_explain, len(test_rows)))
    explain_windows = {x: build_window(series, x) for x in explain_xs}

    predict_rows: List[Tuple[str, List[List[float]]]] = []
    for x, window in explain_windows.items():
        for mask in range(64):
            predict_rows.append((f"{x}:{mask}", apply_mask(window, baseline_window, mask)))

    predictions = run_cli_predict(args.cli, args.experiment_dir, args.approach, predict_rows)

    per_instance = {}
    additivity_residuals = []
    channel_phi: Dict[str, List[float]] = {name: [] for name in CHANNEL_NAMES}
    for x in explain_xs:
        v = {mask: predictions[f"{x}:{mask}"] for mask in range(64)}
        phi = exact_shapley_values(v)
        full_minus_baseline = v[63] - v[0]
        residual = full_minus_baseline - sum(phi)
        additivity_residuals.append(residual)
        per_instance[x] = dict(
            phi=dict(zip(CHANNEL_NAMES, phi)),
            model_output_full=v[63],
            model_output_baseline=v[0],
            additivity_residual=residual,
        )
        for name, value in zip(CHANNEL_NAMES, phi):
            channel_phi[name].append(value)

    report = dict(
        n_background=len(background_windows),
        n_explained=len(explain_xs),
        seed=args.seed,
        max_abs_additivity_residual=max(abs(r) for r in additivity_residuals),
        mean_abs_additivity_residual=sum(abs(r) for r in additivity_residuals) / len(additivity_residuals),
        channel_mean_abs_phi={name: sum(abs(v) for v in vals) / len(vals)
                              for name, vals in channel_phi.items()},
        channel_mean_phi={name: sum(vals) / len(vals) for name, vals in channel_phi.items()},
        per_instance=per_instance,
        note=(
            "Grouped (6-channel), EXACT Shapley attribution against a single frozen "
            "mean-background-window reference. No training. No causal claim -- this "
            "describes how THIS checkpoint's output responds to each weather channel "
            "relative to the chosen baseline, for the explained windows only."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        dict(n_explained=len(explain_xs),
             max_abs_additivity_residual=report["max_abs_additivity_residual"],
             channel_mean_abs_phi=report["channel_mean_abs_phi"]),
        indent=2,
    ))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
