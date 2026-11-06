#!/usr/bin/env python3
"""Audit rainfall-product overlap and MRMS-first/POWER-fallback coverage.

Development-only, rainfall-only numeric audit; no training or discharge values.
This evaluates the stated selection policy, not C++ feature construction. Use
--package for a development HydroPINN package and --output for a NEW report.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

EPOCH = datetime(2015, 1, 1, tzinfo=timezone.utc)
SEAL = datetime(2023, 10, 15, tzinfo=timezone.utc)


def timestamp(raw):
    t = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('timestamp requires timezone')
    t = t.astimezone(timezone.utc)
    if t >= SEAL:
        raise ValueError('reserved-period timestamp refused')
    if t < EPOCH or t.minute or t.second or t.microsecond:
        raise ValueError('timestamp must be a development whole hour')
    return t


def load_rain(path, variable, start, end):
    out = {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = timestamp(row['timestamp_utc'])  # before numeric access, for every variable
            if not start <= t <= end:
                raise ValueError('row outside declared package window')
            if row['variable'] != variable:
                continue
            if t in out:
                raise ValueError('duplicate rainfall timestamp')
            if row['native_unit'] not in {'mm/hour', 'mm/h', 'mm/hr'}:
                raise ValueError('hourly rainfall units required')
            v = float(row['value'])
            if not math.isfinite(v) or v < 0:
                raise ValueError('rainfall must be finite and nonnegative')
            out[t] = v
    return out


def fraction(n, d):
    return round(n / d, 6) if d else None


def audit(package):
    mf = package / 'manifest.json'
    manifest = json.loads(mf.read_text())
    start, end = timestamp(manifest['study_start']), timestamp(manifest['study_end'])
    if end < start:
        raise ValueError('study_end precedes study_start')
    area = float(manifest['catchment_area_m2'])
    if not math.isfinite(area) or area <= 0:
        raise ValueError('area must be finite and positive')
    mp, np = package / 'observations/temporal_4.csv', package / 'observations/temporal_1.csv'
    mrms = load_rain(mp, 'PRECIP_MRMS_BASIN_AVG', start, end)
    nasa = load_rain(np, 'PRECTOTCORR', start, end)
    total = int((end-start).total_seconds() // 3600) + 1
    overlap = sorted(mrms.keys() & nasa.keys())
    diff = [abs(mrms[t]-nasa[t]) for t in overlap]
    fallback = len(nasa.keys()-mrms.keys())
    missing = total-len(mrms.keys() | nasa.keys())
    if len(mrms)+fallback+missing != total:
        raise ValueError('coverage accounting failed')
    return dict(
        source_package=str(package.resolve()),
        policy_evaluated='MRMS when present (including zero); otherwise POWER; otherwise missing',
        policy_scope='numeric coverage audit; consumer implementation/model features need separate review',
        catchment_area_m2_used=area, catchment_area_km2_used=area/1e6,
        total_hours_in_study_window=total,
        mrms_present_hours=len(mrms), mrms_coverage_fraction=fraction(len(mrms), total),
        nasa_power_present_hours=len(nasa), nasa_power_raw_coverage_fraction=fraction(len(nasa), total),
        hours_with_both_present=len(overlap), hours_using_nasa_power_fallback=fallback,
        fraction_of_hours_using_nasa_power_fallback=fraction(fallback, total),
        hours_with_both_missing=missing, fraction_of_hours_with_both_missing_gap=fraction(missing, total),
        mrms_zero_value_fraction_when_present=fraction(sum(v == 0 for v in mrms.values()), len(mrms)),
        nasa_power_zero_value_fraction_when_present=fraction(sum(v == 0 for v in nasa.values()), len(nasa)),
        product_disagreement_mm_per_hour=dict(
            n_hours_both_present=len(diff),
            mean_abs_diff=round(statistics.mean(diff), 6) if diff else None,
            median_abs_diff=round(statistics.median(diff), 6) if diff else None,
            p90_abs_diff=round(statistics.quantiles(diff, n=10)[8], 6) if len(diff) >= 10 else None,
            p90_method='statistics.quantiles exclusive deciles; at least ten overlaps',
            fraction_with_zero_abs_diff=fraction(sum(v == 0 for v in diff), len(diff))),
        input_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in [mf, mp, np]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new report; existing files refused')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    report = audit(args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        f.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(f"Audited {report['total_hours_in_study_window']} development hours; wrote {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
