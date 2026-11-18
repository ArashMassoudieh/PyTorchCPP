#!/usr/bin/env python3
"""Rescores both arms (corrected-forcing vs. fresh original-forcing
controls) to the FROZEN, non-overlapping bounded test windows
(test_end_x: A=50251, B=58784, C=66910), restricted to each fold's
common native-observation support set, per report 39's established
protocol.

No retraining. Reads each job's already-exported predictions.csv
(validation-only checkpoint selection verified from source in report
39; test rows are a pure no-grad forward pass that never fed back
into training, so slicing a bounded window out of the full export is
a valid post-hoc repair, not a new measurement).

No package modification. No reserved-period access (every bounded
window's upper end is <= 66910, well below the x=77016 seal).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ORIGINAL_PACKAGE = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_corrected_devperiod/hydropinn"
)
CORRECTED_PACKAGE = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_interval_corrected_devperiod/hydropinn"
)
NATIVE_DISCHARGE_SUPPORT = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/native_discharge_support_v2.json"
)
LOOP_DIR = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023"
)
CORRECTED_RUNS = LOOP_DIR / "interval_corrected_screen_runs"
ORIGINAL_CONTROL_RUNS = LOOP_DIR / "interval_screen_original_forcing_controls_runs"

SEAL_START_X = 77016
FOLD_TEST_END_X = {"A": 50251, "B": 58784, "C": 66910}
SEEDS = [42, 123, 27182]

import datetime as _dt
EPOCH = _dt.datetime(2015, 1, 1, tzinfo=_dt.timezone.utc)


def ts_to_x(ts: str) -> int:
    dt = _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    return int((dt - EPOCH).total_seconds() // 3600)


def load_mrms_native_x(package_hydropinn_dir: Path) -> set[int]:
    path = package_hydropinn_dir / "observations" / "temporal_4.csv"
    xs = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variable"] != "PRECIP_MRMS_BASIN_AVG":
                continue
            xs.add(ts_to_x(row["timestamp_utc"]))
    return xs


def load_predictions(path: Path) -> dict[int, tuple[float, float]]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["x"])] = (float(row["observed"]), float(row["predicted"]))
    return out


def nse(obs: list[float], pred: list[float]) -> float:
    mean_obs = sum(obs) / len(obs)
    num = sum((o - p) ** 2 for o, p in zip(obs, pred))
    den = sum((o - mean_obs) ** 2 for o in obs)
    return 1 - num / den if den > 0 else float("nan")


def kge(obs: list[float], pred: list[float]) -> float:
    n = len(obs)
    mean_o = sum(obs) / n
    mean_p = sum(pred) / n
    var_o = sum((o - mean_o) ** 2 for o in obs) / n
    var_p = sum((p - mean_p) ** 2 for p in pred) / n
    std_o, std_p = math.sqrt(var_o), math.sqrt(var_p)
    cov = sum((o - mean_o) * (p - mean_p) for o, p in zip(obs, pred)) / n
    if std_o == 0 or std_p == 0 or mean_o == 0:
        return float("nan")
    r = cov / (std_o * std_p)
    alpha = std_p / std_o
    beta = mean_p / mean_o
    return 1 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def mae(obs: list[float], pred: list[float]) -> float:
    return sum(abs(o - p) for o, p in zip(obs, pred)) / len(obs)


def volume_bias_percent(obs: list[float], pred: list[float]) -> float:
    sum_o = sum(obs)
    if sum_o == 0:
        return float("nan")
    return 100.0 * (sum(pred) - sum_o) / sum_o


def main() -> None:
    discharge_native = set(json.loads(NATIVE_DISCHARGE_SUPPORT.read_text())["strict_observed_hour_x"])
    mrms_native_original = load_mrms_native_x(ORIGINAL_PACKAGE)
    mrms_native_corrected = load_mrms_native_x(CORRECTED_PACKAGE)

    results = {"generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "method": "post-hoc rescoring from existing predictions.csv exports; no retraining; "
                         "window=[test_start_x, test_end_x_frozen_bound] intersected with common "
                         "native-observation support (discharge AND MRMS native in both packages).",
               "folds": {}}

    for fold, test_end_x in FOLD_TEST_END_X.items():
        corrected_fold_dir = CORRECTED_RUNS / f"intervalScreen_lstm_plain_24h_fold{fold}_s42" / "predictions.csv"
        test_start_x = min(int(r["x"]) for r in csv.DictReader(open(corrected_fold_dir)) if r["split"] == "test")
        if test_end_x >= SEAL_START_X:
            raise RuntimeError(f"refusing: fold {fold} bound {test_end_x} would reach sealed period")

        window = set(range(test_start_x, test_end_x + 1))
        common_support = window & discharge_native & mrms_native_original & mrms_native_corrected

        fold_result = {
            "test_start_x": test_start_x,
            "test_end_x_frozen_bound": test_end_x,
            "n_common_support": len(common_support),
            "seeds": {},
        }

        for seed in SEEDS:
            corrected_path = CORRECTED_RUNS / f"intervalScreen_lstm_plain_24h_fold{fold}_s{seed}" / "predictions.csv"
            control_path = (ORIGINAL_CONTROL_RUNS /
                             f"intervalScreen_originalForcingControl_lstm_plain_24h_fold{fold}_s{seed}" /
                             "predictions.csv")

            corrected_preds = load_predictions(corrected_path)
            control_preds = load_predictions(control_path)

            common_x_corrected = sorted(x for x in common_support if x in corrected_preds)
            common_x_control = sorted(x for x in common_support if x in control_preds)
            if common_x_corrected != common_x_control:
                raise RuntimeError(
                    f"fold {fold} seed {seed}: export coverage of common-support x differs between arms "
                    f"({len(common_x_corrected)} vs {len(common_x_control)}) -- investigate before scoring"
                )
            xs = common_x_corrected

            obs_corrected = [corrected_preds[x][0] for x in xs]
            pred_corrected = [corrected_preds[x][1] for x in xs]
            obs_control = [control_preds[x][0] for x in xs]
            pred_control = [control_preds[x][1] for x in xs]

            if obs_corrected != obs_control:
                mismatches = sum(1 for a, b in zip(obs_corrected, obs_control) if a != b)
                raise RuntimeError(
                    f"fold {fold} seed {seed}: observed target differs between arms at "
                    f"{mismatches}/{len(xs)} common-support rows -- target should be byte-identical"
                )

            fold_result["seeds"][seed] = {
                "n_scored": len(xs),
                "original_forcing_control": {
                    "nse": nse(obs_control, pred_control),
                    "kge": kge(obs_control, pred_control),
                    "mae": mae(obs_control, pred_control),
                    "volume_bias_percent": volume_bias_percent(obs_control, pred_control),
                },
                "corrected_forcing": {
                    "nse": nse(obs_corrected, pred_corrected),
                    "kge": kge(obs_corrected, pred_corrected),
                    "mae": mae(obs_corrected, pred_corrected),
                    "volume_bias_percent": volume_bias_percent(obs_corrected, pred_corrected),
                },
                "delta_nse_corrected_minus_control": nse(obs_corrected, pred_corrected) - nse(obs_control, pred_control),
            }

        results["folds"][fold] = fold_result
        print(f"Fold {fold}: window=[{test_start_x},{test_end_x}], common_support={len(common_support)}")
        for seed in SEEDS:
            s = fold_result["seeds"][seed]
            print(f"  seed {seed}: n={s['n_scored']} "
                  f"control_NSE={s['original_forcing_control']['nse']:.6f} "
                  f"corrected_NSE={s['corrected_forcing']['nse']:.6f} "
                  f"delta={s['delta_nse_corrected_minus_control']:+.6f}")

    out_path = LOOP_DIR / "component_diagnostics" / "bounded_window_rescoring_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
