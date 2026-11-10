#!/usr/bin/env python3
"""Independent re-derivation of the per-year (annual) NSE breakdown reported in
annual_metrics.csv and quoted in the manuscript (e.g. "2019: LSTM 0.734 vs. FFN 0.305").

This is a second, independent implementation -- it does NOT import or call
analyze_annual_evaluation.py or audit_development_predictions.metrics. It re-reads the
same raw per-job predictions.csv files directly, re-derives per-year common test support
from the same manifest/native-support inputs with its own set-intersection logic, and
recomputes NSE with its own formula. Disagreement with the existing annual_metrics.csv
at more than a rounding tolerance indicates a real bug in one of the two
implementations; agreement is the independent confirmation this program's own reports
83/84 named as not yet done.

No training. No reserved-period access (asserted at runtime). Reads only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

RESERVED_PERIOD_X_CUTOFF = 77016


class ReverifyError(ValueError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def nse(observed: List[float], predicted: List[float]) -> float:
    n = len(observed)
    if n == 0 or n != len(predicted):
        raise ReverifyError("NSE requires matching, non-empty series")
    mean_obs = sum(observed) / n
    denom = sum((o - mean_obs) ** 2 for o in observed)
    if denom <= 0.0:
        raise ReverifyError("NSE denominator is zero or negative")
    numer = sum((o - p) ** 2 for o, p in zip(observed, predicted))
    return 1.0 - numer / denom


def read_job_test_rows(predictions_csv: Path, test_start_x: float) -> Dict[float, Tuple[float, float]]:
    """Independently re-reads one job's own predictions.csv: test-split rows at or
    after that job's own frozen test_start_x. Rejects reserved-period x, non-finite
    values, and duplicate x, matching this program's established defensive pattern."""
    rows: Dict[float, Tuple[float, float]] = {}
    with predictions_csv.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] != 'test':
                continue
            x = float(row['x'])
            if x < test_start_x:
                continue
            if x >= RESERVED_PERIOD_X_CUTOFF:
                raise ReverifyError(f"x={x} at/past reserved-period seal in {predictions_csv}")
            o = float(row['observed'])
            p = float(row['predicted'])
            if not (o == o and p == p):  # NaN check without importing math
                raise ReverifyError(f"Non-finite value at x={x} in {predictions_csv}")
            if x in rows:
                raise ReverifyError(f"Duplicate test x={x} in {predictions_csv}")
            rows[x] = (o, p)
    if not rows:
        raise ReverifyError(f"No test rows at/after test_start_x in {predictions_csv}")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--native-support', type=Path, required=True)
    p.add_argument('--compare-against', type=Path, required=True,
                   help="existing annual_metrics.csv to cross-check against (read-only)")
    p.add_argument('--output', type=Path, required=True, help="new report; existing files refused")
    args = p.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    manifest = json.loads(args.manifest.read_text())
    jobs = manifest['jobs']
    support = json.loads(args.native_support.read_text())
    for check in support['boundary_interpolation_checks']:
        if check['interpolation_spans']:
            raise ReverifyError("Target interpolation crosses a partition boundary")
    native = set(support['strict_observed_hour_x'])

    # Independently verify each job's predictions.csv matches its own recorded
    # VERIFIED.json hash before trusting it -- same defense-in-depth as the original
    # script, reimplemented here rather than imported.
    job_rows: Dict[str, Dict[float, Tuple[float, float]]] = {}
    for j in jobs:
        job_root = args.runs / j['id']
        verified = json.loads((job_root / 'VERIFIED.json').read_text())
        predictions_csv = job_root / j['id'] / 'predictions.csv'
        actual_hash = sha256_of(predictions_csv)
        if actual_hash != verified['predictions_sha256']:
            raise ReverifyError(f"predictions.csv hash mismatch for {j['id']}: "
                                 f"recorded {verified['predictions_sha256']}, actual {actual_hash}")
        job_rows[j['id']] = read_job_test_rows(predictions_csv, j['test_start_x'])

    # Independent re-derivation of per-year common support and per-model/seed NSE.
    reverified: Dict[Tuple[str, str, int], dict] = {}
    for year in sorted({j['year'] for j in jobs}):
        group = [j for j in jobs if j['year'] == year]
        common = set.intersection(*(set(job_rows[j['id']]) for j in group))
        common &= native
        if not common:
            raise ReverifyError(f"No common native test support for year {year}")
        reference_job = group[0]['id']
        reference = job_rows[reference_job]
        for j in group:
            rows = job_rows[j['id']]
            for x in common:
                if abs(rows[x][0] - reference[x][0]) > 1e-6:
                    raise ReverifyError(
                        f"Observation disagreement at x={x} between {j['id']} and {reference_job}"
                    )
            observed = [rows[x][0] for x in sorted(common)]
            predicted = [rows[x][1] for x in sorted(common)]
            reverified[(year, j['model'], j['seed'])] = dict(
                n=len(common), nse=round(nse(observed, predicted), 6)
            )

    # Cross-check against the existing annual_metrics.csv (read-only comparison, not
    # a dependency of the computation above).
    mismatches = []
    checked = 0
    with args.compare_against.open(newline='') as f:
        for row in csv.DictReader(f):
            key = (row['year'], row['model'], int(row['seed']))
            if key not in reverified:
                mismatches.append(f"{key}: present in annual_metrics.csv but not reproduced here")
                continue
            existing_nse = float(row['nse'])
            new_nse = reverified[key]['nse']
            checked += 1
            if abs(existing_nse - new_nse) > 1e-4:
                mismatches.append(
                    f"{key}: annual_metrics.csv nse={existing_nse} vs. independently "
                    f"re-derived nse={new_nse}"
                )

    report = {
        "n_year_model_seed_combinations_reverified": len(reverified),
        "n_compared_against_annual_metrics_csv": checked,
        "mismatches": mismatches,
        "reverified_per_year_ffn_vs_lstm_24h": {
            year: {
                model + "_mean_nse": round(sum(nses) / len(nses), 6)
                for model in ("ffn", "lstm_24h")
                if (nses := [
                    reverified[(year, model, s)]['nse']
                    for s in (42, 123, 27182)
                    if (year, model, s) in reverified
                ])
            }
            for year in sorted({j['year'] for j in jobs})
        },
        "note": (
            "Independent second implementation: re-reads raw per-job predictions.csv files "
            "directly (not analyze_annual_evaluation.py's output), re-derives per-year common "
            "native test support with its own set-intersection logic, and recomputes NSE with "
            "its own formula. Agreement with annual_metrics.csv (checked to 1e-4 absolute NSE "
            "tolerance) is independent confirmation, not a restatement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(
        {"n_reverified": len(reverified), "n_mismatches": len(mismatches)}, indent=2
    ))
    print("wrote", args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
