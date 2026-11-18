#!/usr/bin/env python3
"""Extends native_discharge_support_v2.json's exact method (5-minute
forward-support convention, see audit_native_discharge_support.py) to
the full discharge series, including the reserved period. Identical
logic throughout -- ordering/duplicate checks, the per-5-minute-sample
forward-support accumulation, the >=3600s-covered strict threshold --
with only the reserved-period refusal removed, since reading that
period is the explicitly authorized action this script supports.

Does not change how any development-period hour is classified: the
development-period portion of strict_observed_hour_x produced here is
compared against native_discharge_support_v2.json's own values as a
regression check before this script's output is used for anything.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CSV_PATH = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_interval_corrected_full/hydropinn/observations/temporal_3.csv"
)
EXISTING_V2 = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "native_discharge_support_v2.json"
)
OUTPUT = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "improvement_loop_20261023/component_diagnostics/native_discharge_support_full.json"
)


def main() -> None:
    epoch = datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp()
    times = []
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            t = int(datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).timestamp() - epoch)
            if times and t <= times[-1]:
                raise ValueError("Unordered or duplicate native timestamps")
            times.append(t)

    covered = defaultdict(int)
    for i, t in enumerate(times):
        end = min(t + 300, times[i + 1] if i + 1 < len(times) else t + 300)
        while t < end:
            h = t // 3600
            stop = min(end, (h + 1) * 3600)
            covered[h] += stop - t
            t = stop

    strict = sorted(h for h, s in covered.items() if s >= 3600)

    # Regression check: every development-period (x<77016) hour must match
    # the existing frozen native_discharge_support_v2.json exactly.
    existing = set(json.loads(EXISTING_V2.read_text())["strict_observed_hour_x"])
    strict_dev = set(h for h in strict if h < 77016)
    if strict_dev != existing:
        missing = existing - strict_dev
        extra = strict_dev - existing
        raise RuntimeError(
            f"Development-period native support changed vs the frozen v2 file: "
            f"{len(missing)} hours missing, {len(extra)} hours added. Refusing to proceed -- "
            f"this must reproduce the frozen development-period result exactly."
        )
    print(f"Development-period regression check passed: {len(strict_dev)} hours match "
          f"native_discharge_support_v2.json exactly.")

    reserved = sorted(h for h in strict if h >= 77016)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "sample_support_seconds": 300,
        "strict_observed_hour_x": strict,
        "strict_observed_hour_x_reserved_period_only": reserved,
        "note": "Native-only coverage under the harmonizer five-minute forward-support convention; "
                "excludes interpolated target hours. Development-period (x<77016) portion verified "
                "identical to native_discharge_support_v2.json. Reserved-period (x>=77016) portion "
                "is new, built by the same method, for the one-shot holdout evaluation.",
    }, indent=2))
    print(f"Total native-supported hours: {len(strict)}; reserved-period native-supported hours: {len(reserved)}")
    print(f"Reserved-period x range in data: [{min(h for h in strict if h>=77016) if reserved else None}, "
          f"{max(times)//3600}]")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
