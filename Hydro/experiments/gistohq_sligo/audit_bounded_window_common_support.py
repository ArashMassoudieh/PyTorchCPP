#!/usr/bin/env python3
"""Determines common native-observation support within the FROZEN,
non-overlapping per-fold test windows (test_end_x: A=50251, B=58784,
C=66910 -- the pre-existing bound this program established in
08_development_period_and_folds.md / phase3's fine-tune templates, NOT
a new invention), for both the original and interval-corrected MRMS
packages.

This does NOT retrain anything. It reads the already-exported
predictions.csv (for the chronological train/validation/test_start
boundaries, independently verified identical between packages) and
each package's own temporal_4.csv (for native MRMS support), and
reports, per fold, within the BOUNDED window only:
  - the discharge-native-observed-hour set (reused from the project's
    own established native_discharge_support_v2.json, not redefined)
  - which x-values have native MRMS support in the original package
  - which x-values have native MRMS support in the corrected package
  - where those two MRMS-support sets disagree, if anywhere, inside
    the bounded window specifically (distinct from the single
    end-of-series disagreement at x=77015 already found in report 38,
    which is outside every bounded window)

No training. No package modification. No reserved-period access (every
bounded window's upper end is well before x=77016/2023-10-15).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

EPOCH = datetime(2015, 1, 1, tzinfo=timezone.utc)  # x=0; verified x=77015 <-> 2023-10-14T23:00:00Z exactly

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
CORRECTED_RUNS = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "improvement_loop_20261023/interval_corrected_screen_runs"
)
SEAL_START_X = 77016  # x=1 .. x=77015 permitted; this script never reads x >= 77016

# Previously frozen, non-overlapping per-fold bounded test windows.
# test_start values verified directly from predictions.csv below (not
# assumed); test_end_x values are the user-specified frozen bounds.
FOLD_TEST_END_X = {"A": 50251, "B": 58784, "C": 66910}


def ts_to_x(ts: str) -> int:
    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
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


def load_test_start_x() -> dict[str, int]:
    starts = {}
    for fold in ("A", "B", "C"):
        path = CORRECTED_RUNS / f"intervalScreen_lstm_plain_24h_fold{fold}_s42" / "predictions.csv"
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["split"] == "test":
                    x = int(row["x"])
                    starts[fold] = min(starts.get(fold, x), x)
    return starts


def main() -> None:
    discharge_native = set(json.loads(NATIVE_DISCHARGE_SUPPORT.read_text())["strict_observed_hour_x"])
    mrms_native_original = load_mrms_native_x(ORIGINAL_PACKAGE)
    mrms_native_corrected = load_mrms_native_x(CORRECTED_PACKAGE)
    test_start_x = load_test_start_x()

    report = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "folds": {}}

    for fold, test_end_x in FOLD_TEST_END_X.items():
        start = test_start_x[fold]
        end = test_end_x
        if end >= SEAL_START_X:
            raise RuntimeError(f"refusing: fold {fold} test_end_x={end} would reach the sealed period")
        if end <= start:
            raise RuntimeError(f"fold {fold}: test_end_x {end} does not leave a non-empty bounded window after test_start {start}")
        window = set(range(start, end + 1))

        window_discharge_native = window & discharge_native
        window_mrms_original = window & mrms_native_original
        window_mrms_corrected = window & mrms_native_corrected
        mrms_support_mismatch = window_mrms_original.symmetric_difference(window_mrms_corrected)
        common_support = window_discharge_native & window_mrms_original & window_mrms_corrected

        report["folds"][fold] = {
            "test_start_x": start,
            "test_end_x_frozen_bound": end,
            "bounded_window_size": len(window),
            "n_discharge_native_in_window": len(window_discharge_native),
            "n_mrms_native_original_in_window": len(window_mrms_original),
            "n_mrms_native_corrected_in_window": len(window_mrms_corrected),
            "n_mrms_support_mismatch_in_window": len(mrms_support_mismatch),
            "mrms_support_mismatch_x_values": sorted(mrms_support_mismatch),
            "n_common_native_support": len(common_support),
            "common_support_x_min_max": [min(common_support), max(common_support)] if common_support else None,
        }
        print(f"Fold {fold}: window x=[{start},{end}] (n={len(window)}), "
              f"discharge-native={len(window_discharge_native)}, "
              f"mrms-native(orig)={len(window_mrms_original)}, mrms-native(corr)={len(window_mrms_corrected)}, "
              f"mismatch={len(mrms_support_mismatch)}, common-support={len(common_support)}")

    out_path = Path(
        "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/"
        "component_diagnostics/bounded_window_common_support.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
