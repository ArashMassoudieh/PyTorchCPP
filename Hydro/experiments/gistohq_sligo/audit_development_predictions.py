#!/usr/bin/env python3
"""Read-only audit of explicitly supplied development predictions.

Reject reserved-period rows, duplicate times, nonfinite values and temporally
interleaved partitions. Output is separate from immutable experiment artifacts.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SEAL_X = 77016


def metrics(rows):
    n = len(rows)
    if not n:
        raise ValueError('empty evaluation support')
    obs = [r[1] for r in rows]
    pred = [r[2] for r in rows]
    mo, mp = sum(obs)/n, sum(pred)/n
    sst = sum((v-mo)**2 for v in obs)
    spp = sum((v-mp)**2 for v in pred)
    sse = sum((p-o)**2 for o, p in zip(obs, pred))
    cov = sum((o-mo)*(p-mp) for o, p in zip(obs, pred))
    r = cov/math.sqrt(sst*spp) if sst*spp > 0 else None
    kge = (1-math.sqrt((r-1)**2 + (math.sqrt(spp/sst)-1)**2 + (mp/mo-1)**2)
           if r is not None and mo else None)
    return dict(n=n, nse=1-sse/sst if sst else None,
                kge=kge,
                pearson_r2=r*r if r is not None else None,
                rmse=math.sqrt(sse/n), mae=sum(abs(p-o) for o,p in zip(obs,pred))/n,
                pbias=100*sum(p-o for o,p in zip(obs,pred))/sum(obs) if sum(obs) else None)


def audit(path):
    groups = {s: [] for s in ('train', 'validation', 'test', 'beyond_test_end')}
    seen = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            x, o, p = (float(row[k]) for k in ('x','observed','predicted'))
            if not all(math.isfinite(v) for v in (x,o,p)):
                raise ValueError(f'{path}: nonfinite row')
            if x >= SEAL_X:
                raise ValueError(f'{path}: reserved boundary reached: {x}')
            if x in seen:
                raise ValueError(f'{path}: duplicate timestamp {x}')
            seen.add(x)
            groups[row['split']].append((x,o,p))
    last = -math.inf
    report = {}
    for split, rows in groups.items():
        if not rows and split == 'beyond_test_end':
            continue
        if not rows:
            raise ValueError(f'{path}: empty {split}')
        if rows != sorted(rows):
            raise ValueError(f'{path}: unordered {split}')
        if rows[0][0] <= last:
            raise ValueError(f'{path}: partition overlap')
        last = rows[-1][0]
        report[split] = dict(n=len(rows), first_x=rows[0][0], last_x=last,
                            missing_hours=int(last-rows[0][0]+1-len(rows)))
    report['test_metrics'] = metrics(groups['test'])
    report['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report, groups['test']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True,
                        help='Known development-only experiment root; never use full-package roots')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for root in args.root:
        paths = sorted(root.glob('*/predictions.csv'))
        if not paths:
            raise ValueError(f'No predictions under {root}')
        for path in paths:
            report, _ = audit(path)
            reports[str(path)] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, allow_nan=False)+'\n')
    print(f'Validated {len(reports)} development runs; report: {args.output}')


if __name__ == '__main__':
    main()
