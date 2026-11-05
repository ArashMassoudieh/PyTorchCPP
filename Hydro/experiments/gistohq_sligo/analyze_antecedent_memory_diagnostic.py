#!/usr/bin/env python3
"""Antecedent-memory diagnostic (Task 2, CLAUDE_RESULTS_IMPROVEMENT_TASKS_20261105.md).

Development-forcing-only, no training, no outcome-optimized lag search. Quantifies:
  1. Antecedent rainfall totals ending at each frozen event's start, at hourly
     resolution (24h/72h/168h, trailing from the event start) and at daily
     resolution (14/30/60 complete prior calendar days, same alignment rule
     already established for the multiscale daily-history branch: only
     fully-completed days strictly before the window's start).
  2. What daily aggregation structurally discards versus the hourly series,
     measured directly from the forcing data (burstiness: the fraction of a
     day's total rainfall concentrated in its single wettest hour), not
     inferred from model error.
  3. The existing, already-frozen 174-event failure atlas
     (results/framework_decade_20261022/failure_atlas_v1/event_summary.csv --
     reused verbatim, not regenerated, per the standing "freeze before
     inspecting additional events" rule and report 76's precedent of reusing
     this exact table), stratified by the new antecedent-wetness and
     event-duration bins computed here -- across ALL 174 events, not just
     the worst five.

Source: entirely the development-period-sealed MRMS package
sligocreekdemo_mrms_corrected_devperiod (study_end 2023-10-14T23:00:00Z, strictly
before the x=77016 / 2023-10-15T00:00:00Z seal). No reserved-period timestamp or
value is read by this script.
"""
from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

MRMS_PATH = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_corrected_devperiod/hydropinn/observations/temporal_4.csv"
)
EVENT_TABLE_PATH = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "failure_atlas_v1/event_summary.csv"
)
OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "Hydro" / "experiments" / "gistohq_sligo"
# (kept local to this experiments dir; output written next to this script)
OUT_JSON = Path(__file__).resolve().parent / "antecedent_memory_diagnostic_output.json"
OUT_CSV = Path(__file__).resolve().parent / "antecedent_memory_diagnostic_events.csv"

RESERVED_PERIOD_X_CUTOFF_TIMESTAMP = datetime(2023, 10, 15, 0, 0, 0, tzinfo=timezone.utc)


def load_mrms_hourly(path: Path) -> dict[datetime, float]:
    series: dict[datetime, float] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            ts = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
            if ts >= RESERVED_PERIOD_X_CUTOFF_TIMESTAMP:
                raise RuntimeError(
                    f"Reserved-period timestamp {ts} encountered in {path} -- this script "
                    "must never read a value at or past the seal, even from a package "
                    "documented as development-only; refusing to proceed."
                )
            series[ts] = float(row["value"])
    return series


def hourly_window_sum(series: dict[datetime, float], end_exclusive: datetime, hours: int) -> tuple[float, float]:
    """Trailing sum over [end_exclusive - hours, end_exclusive). Returns (sum_mm,
    coverage_fraction) -- coverage < 1.0 means some hours in the window have no
    recorded value (a real, short forcing gap), which is reported explicitly
    rather than silently treated as zero rainfall."""
    start = end_exclusive - timedelta(hours=hours)
    total = 0.0
    present = 0
    t = start
    while t < end_exclusive:
        if t in series:
            total += series[t]
            present += 1
        t += timedelta(hours=1)
    return total, present / hours


def daily_window_sum(series: dict[datetime, float], end_exclusive: datetime, days: int) -> tuple[float, float, int]:
    """Sum over the `days` most recent FULLY COMPLETE UTC calendar days strictly
    before the day containing end_exclusive -- same alignment rule as the
    multiscale daily-history branch (report 66/68): a day is only used if every
    one of its 24 hours has a recorded value. Returns (sum_mm, coverage_fraction
    = complete_days_found / days_requested, complete_days_found)."""
    day_start = datetime(end_exclusive.year, end_exclusive.month, end_exclusive.day, tzinfo=timezone.utc)
    total = 0.0
    complete_days_found = 0
    cursor = day_start
    while complete_days_found < days:
        cursor -= timedelta(days=1)
        hours = [cursor + timedelta(hours=h) for h in range(24)]
        if all(h in series for h in hours):
            total += sum(series[h] for h in hours)
            complete_days_found += 1
        else:
            # A day with any missing hour is skipped entirely (not partially
            # summed), matching the "fully-completed days only" rule -- but we
            # keep walking backward rather than aborting, so one short gap
            # doesn't silently truncate the whole window to zero days found.
            # Walking indefinitely is bounded by the series' own start.
            if cursor < min(series):
                break
    return total, complete_days_found / days, complete_days_found


def main() -> int:
    series = load_mrms_hourly(MRMS_PATH)
    series_start = min(series)
    series_end = max(series)

    # --- Diagnostic 1: what does daily aggregation structurally discard? ---
    # Burstiness = (wettest single hour) / (day total), for every fully-complete
    # UTC day with nonzero rainfall. A value near 1.0 means the day's rain was
    # concentrated in essentially one hour -- information a daily total alone
    # cannot distinguish from the same total spread evenly across 24 hours.
    burstiness_values = []
    day_cursor = datetime(series_start.year, series_start.month, series_start.day, tzinfo=timezone.utc)
    while day_cursor <= series_end:
        hours = [day_cursor + timedelta(hours=h) for h in range(24)]
        if all(h in series for h in hours):
            values = [series[h] for h in hours]
            day_total = sum(values)
            if day_total > 0.5:  # ignore near-zero days where burstiness is a noisy ratio
                burstiness_values.append(max(values) / day_total)
        day_cursor += timedelta(days=1)

    burstiness_summary = {
        "n_wet_complete_days": len(burstiness_values),
        "mean": statistics.mean(burstiness_values),
        "median": statistics.median(burstiness_values),
        "p90": statistics.quantiles(burstiness_values, n=10)[8],
        "fraction_above_0.5": sum(1 for v in burstiness_values if v > 0.5) / len(burstiness_values),
    }

    # --- Diagnostic 2: per-event antecedent totals, hourly and daily resolution ---
    events = []
    with open(EVENT_TABLE_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            start_utc = datetime.fromisoformat(row["start_utc"].replace("Z", "+00:00"))
            end_utc = datetime.fromisoformat(row["end_utc"].replace("Z", "+00:00"))
            a24, c24 = hourly_window_sum(series, start_utc, 24)
            a72, c72 = hourly_window_sum(series, start_utc, 72)
            a168, c168 = hourly_window_sum(series, start_utc, 168)
            d14, dc14, _ = daily_window_sum(series, start_utc, 14)
            d30, dc30, _ = daily_window_sum(series, start_utc, 30)
            d60, dc60, _ = daily_window_sum(series, start_utc, 60)
            events.append({
                "event": row["event"],
                "start_utc": row["start_utc"],
                "end_utc": row["end_utc"],
                "duration_hours": (end_utc - start_utc).total_seconds() / 3600.0,
                "observed_peak": float(row["observed_peak"]),
                "lstm_hybrid_sse_share_percent": float(row["lstm_hybrid_sse_share_percent"]),
                "lstm_hybrid_peak_error_percent": float(row["lstm_hybrid_peak_error_percent"]),
                "lstm_hybrid_peak_timing_hours": float(row["lstm_hybrid_peak_timing_hours"]),
                "antecedent_24h_mm": round(a24, 4), "antecedent_24h_coverage": round(c24, 4),
                "antecedent_72h_mm": round(a72, 4), "antecedent_72h_coverage": round(c72, 4),
                "antecedent_72h_mm_existing_column": float(row["rain_antecedent_72h_mm"]),
                "antecedent_168h_mm": round(a168, 4), "antecedent_168h_coverage": round(c168, 4),
                "antecedent_14d_mm": round(d14, 4), "antecedent_14d_coverage": round(dc14, 4),
                "antecedent_30d_mm": round(d30, 4), "antecedent_30d_coverage": round(dc30, 4),
                "antecedent_60d_mm": round(d60, 4), "antecedent_60d_coverage": round(dc60, 4),
            })

    # Integrity check: this script's own 72h computation should match the
    # existing frozen table's own rain_antecedent_72h_mm column closely --
    # confirms the window-alignment convention used here matches the one
    # already established, not a silently different definition.
    diffs_72h = [abs(e["antecedent_72h_mm"] - e["antecedent_72h_mm_existing_column"]) for e in events]
    max_72h_diff = max(diffs_72h)

    # --- Diagnostic 3: stratify existing event-level model error by antecedent
    # wetness (168h proxy -- a week of catchment memory) and by event duration,
    # across all 174 events, not just the worst five. ---
    def tercile_bins(values, labels=("dry/short", "mid", "wet/long")):
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        t1 = sorted_vals[n // 3]
        t2 = sorted_vals[2 * n // 3]
        def bin_of(v):
            if v <= t1:
                return labels[0]
            if v <= t2:
                return labels[1]
            return labels[2]
        return bin_of, (t1, t2)

    wetness_bin_fn, wetness_cutoffs = tercile_bins(
        [e["antecedent_168h_mm"] for e in events], labels=("dry", "moderate", "wet"))
    duration_bin_fn, duration_cutoffs = tercile_bins(
        [e["duration_hours"] for e in events], labels=("short", "medium", "long"))

    for e in events:
        e["antecedent_wetness_bin"] = wetness_bin_fn(e["antecedent_168h_mm"])
        e["event_duration_bin"] = duration_bin_fn(e["duration_hours"])

    def summarize_bin(rows, key):
        vals = [r[key] for r in rows]
        return {
            "n": len(vals),
            "mean_peak_error_percent": round(statistics.mean(r["lstm_hybrid_peak_error_percent"] for r in rows), 3),
            "median_peak_error_percent": round(statistics.median(r["lstm_hybrid_peak_error_percent"] for r in rows), 3),
            "mean_sse_share_percent": round(statistics.mean(r["lstm_hybrid_sse_share_percent"] for r in rows), 3),
        }

    wetness_strata = {}
    for label in ("dry", "moderate", "wet"):
        rows = [e for e in events if e["antecedent_wetness_bin"] == label]
        if rows:
            wetness_strata[label] = summarize_bin(rows, "antecedent_wetness_bin")

    duration_strata = {}
    for label in ("short", "medium", "long"):
        rows = [e for e in events if e["event_duration_bin"] == label]
        if rows:
            duration_strata[label] = summarize_bin(rows, "event_duration_bin")

    report = {
        "source_package": str(MRMS_PATH),
        "source_event_table": str(EVENT_TABLE_PATH),
        "development_series_span": [str(series_start), str(series_end)],
        "reserved_period_timestamps_read": 0,
        "burstiness_diagnostic": burstiness_summary,
        "event_count": len(events),
        "max_72h_antecedent_diff_vs_existing_column_mm": round(max_72h_diff, 6),
        "wetness_tercile_cutoffs_mm_168h": wetness_cutoffs,
        "duration_tercile_cutoffs_hours": duration_cutoffs,
        "wetness_strata": wetness_strata,
        "duration_strata": duration_strata,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)

    print(json.dumps(report, indent=2, sort_keys=True))
    print("wrote", OUT_JSON)
    print("wrote", OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
