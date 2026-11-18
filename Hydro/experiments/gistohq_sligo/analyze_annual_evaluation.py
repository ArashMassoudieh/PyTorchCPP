#!/usr/bin/env python3
"""Score completed annual runs on identical out-of-sample timestamp support.

Requires verified completion of the entire frozen manifest. No fitted or
validation predictions enter the continuous evaluation or aggregate metrics.
"""
import argparse
import csv
import json
from pathlib import Path
from audit_development_predictions import metrics
from run_isolated_batch import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-support',type=Path,required=True,
                   help='Native discharge support audit; do not score interpolated targets as observations')
    args=p.parse_args()
    jobs=json.loads(args.manifest.read_text())['jobs']
    support=json.loads(args.native_support.read_text())
    if any(c['interpolation_spans'] for c in support['boundary_interpolation_checks']):
        raise ValueError('Target interpolation crosses a partition boundary; resolve before scoring')
    native=set(support['strict_observed_hour_x'])
    missing=[j['id'] for j in jobs if not (args.runs/j['id']/'VERIFIED.json').exists()]
    if missing:raise RuntimeError(f'{len(missing)} incomplete runs; refusing final analysis')
    data={};coverage={};pooled={};annual=[]
    for j in jobs:
        root=args.runs/j['id'];path=root/j['id']/'predictions.csv'
        verified=json.loads((root/'VERIFIED.json').read_text())
        if sha(path)!=verified['predictions_sha256']:raise ValueError('Predictions changed since verification')
        if sha(j['config'])!=verified['config_sha256']:raise ValueError('Configuration changed since verification')
        rows={}
        with path.open() as f:
            for row in csv.DictReader(f):
                if row['split']=='test' and float(row['x']) >= j['test_start_x']:
                    x=float(row['x'])
                    if x in rows:raise ValueError('Duplicate test time')
                    rows[x]=(float(row['observed']),float(row['predicted']))
        data[j['id']]=rows
    for year in sorted({j['year'] for j in jobs}):
        group=[j for j in jobs if j['year']==year]
        common=set.intersection(*(set(data[j['id']]) for j in group))
        before_native=len(common)
        common &= native
        if not common:raise ValueError('No common test support')
        coverage[year]=dict(common_hours=len(common),excluded_non_native_hours=before_native-len(common),expected_calendar_hours=group[0]['bounds']['test_end_x']-group[0]['test_start_x']+1,individual_test_hours={j['id']:len(data[j['id']]) for j in group})
        reference=data[group[0]['id']]
        for j in group:
            rows=[]
            for x in sorted(common):
                o,pred=data[j['id']][x]
                if abs(o-reference[x][0])>1e-6:raise ValueError(f'Observation disagreement at {x}')
                rows.append((x,o,pred))
            annual.append(dict(year=year,model=j['model'],seed=j['seed'],**metrics(rows)))
            pooled.setdefault((j['model'],j['seed']),[]).extend(rows)
    aggregate=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for (model,seed),rows in pooled.items():
        xs=[r[0] for r in rows]
        if len(xs)!=len(set(xs)):raise ValueError('Overlapping annual tests')
        aggregate.append(dict(model=model,seed=seed,**metrics(rows)))
        with (args.output/f'out_of_sample_{model}_s{seed}.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['x','observed','predicted']);writer.writerows(rows)
    for name,rows in [('annual_metrics.csv',annual),('pooled_metrics.csv',aggregate)]:
        with (args.output/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    print('Wrote common-support annual and pooled metrics; pooled NSE recomputed from all out-of-sample rows.')


if __name__=='__main__':main()
