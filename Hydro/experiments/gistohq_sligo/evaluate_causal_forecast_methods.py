#!/usr/bin/env python3
"""Evaluate declared common native support; never infer it from model scores.

Inputs must already be presealed slices. This tool does not certify upstream
native provenance; managed-bundle/qualifier verification precedes its use.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev

SEAL = 77016


def timestamp(token):
    value = float(token)
    if not math.isfinite(value) or not value.is_integer() or not 0 <= value < SEAL:
        raise ValueError('Invalid/presealed hourly identity required.')
    return int(value)


def finite_nonnegative(token):
    value = float(token)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative discharge required.')
    return value


def csv_rows(path, fields):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError('Unexpected CSV schema: '+str(path))
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Empty/ragged CSV: '+str(path))
    return rows


def load_support(path):
    values = [timestamp(row['x']) for row in csv_rows(path, ['x'])]
    if values != sorted(set(values)):
        raise ValueError('Support must be unique and sorted.')
    return values


def training_threshold(path, train_end):
    rows = csv_rows(path, ['x', 'observed'])
    identities = [timestamp(row['x']) for row in rows]
    if identities != sorted(set(identities)) or any(x > train_end for x in identities):
        raise ValueError('Q90 must use unique training-only identities.')
    q = sorted(finite_nonnegative(row['observed']) for row in rows)
    position = (len(q)-1)*0.9
    lower = int(position)
    return q[lower]+(q[min(lower+1, len(q)-1)]-q[lower])*(position-lower)


def load_predictions(path, approach):
    rows = csv_rows(path, ['approach','index','split','x','observed','predicted','residual'])
    output = {}
    for row in rows:
        x = timestamp(row['x'])  # before numerical access to target/prediction
        if x in output or row['approach'] != approach or row['split'] != 'test':
            raise ValueError('Duplicate, mismatched approach or non-test prediction row.')
        observed = finite_nonnegative(row['observed'])
        predicted = finite_nonnegative(row['predicted'])
        residual = float(row['residual'])
        if not math.isfinite(residual) or not math.isclose(residual, predicted-observed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('Residual column disagrees with physical discharge.')
        output[x] = observed, predicted
    return output


def metrics(observed, predicted, q90):
    if not observed:
        return {'n':0}
    errors = [p-o for o,p in zip(observed,predicted)]
    sse = sum(e*e for e in errors)
    mu, mp = mean(observed), mean(predicted)
    so, sp = pstdev(observed), pstdev(predicted)
    denominator = sum((o-mu)**2 for o in observed)
    nse = 1-sse/denominator if denominator else None
    correlation = sum((o-mu)*(p-mp) for o,p in zip(observed,predicted))/(len(observed)*so*sp) if so and sp else None
    kge = 1-math.sqrt((correlation-1)**2+(sp/so-1)**2+(mp/mu-1)**2) if correlation is not None and mu else None
    bias = 100*sum(errors)/sum(observed) if sum(observed) else None
    high = [e for o,e in zip(observed,errors) if o >= q90]
    low = [e for o,e in zip(observed,errors) if o < q90]
    if not math.isclose(sum(e*e for e in high)+sum(e*e for e in low),sse,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Tail partition mismatch.')
    return {'n':len(observed),'nse':nse,'kge':kge,'rmse':math.sqrt(sse/len(errors)),
            'mae':mean(abs(e) for e in errors),'pbias_percent':bias,
            'absolute_pbias_percent':abs(bias) if bias is not None else None,
            'high_flow_n':len(high),'high_flow_rmse':math.sqrt(sum(e*e for e in high)/len(high)) if high else None,
            'remaining_n':len(low),'remaining_rmse':math.sqrt(sum(e*e for e in low)/len(low)) if low else None}


def evaluate(methods, support, q90, events):
    if not support or support != sorted(set(support)):
        raise ValueError('Nonempty sorted common support required.')
    reference = None
    output = {}
    for name, values in methods.items():
        if not set(support).issubset(values):
            raise ValueError('Method missing prospectively declared common support: '+name)
        observed = [values[x][0] for x in support]
        predicted = [values[x][1] for x in support]
        if reference is not None and any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(reference,observed)):
            raise ValueError('Observed values disagree across methods.')
        reference = observed
        item = {'pooled':metrics(observed,predicted,q90),'per_year':{},'events':{}}
        years = [(datetime(2015,1,1,tzinfo=timezone.utc)+timedelta(hours=x)).year for x in support]
        for year in sorted(set(years)):
            chosen = [i for i,y in enumerate(years) if y == year]
            item['per_year'][str(year)] = metrics([observed[i] for i in chosen],[predicted[i] for i in chosen],q90)
        for event in events:
            start,end = timestamp(event['start_x']),timestamp(event['end_x'])
            if start > end or not set(range(start,end+1)).issubset(support):
                raise ValueError('Event is reversed or incompletely covered; do not silently truncate.')
            times = list(range(start,end+1))
            obs = [values[x][0] for x in times];pred = [values[x][1] for x in times]
            item['events'][event['name']] = {'n':len(times),'observed_volume_mm':sum(obs),'predicted_volume_mm':sum(pred),
                'peak_magnitude_error_mm_h':max(pred)-max(obs),
                'peak_timing_error_hours':times[pred.index(max(pred))]-times[obs.index(max(obs))]}
        output[name] = item
    return output


def paired_seed_summary(report, comparisons):
    groups = {}
    identities = set()
    for comparison in comparisons:
        identity = comparison['name'], comparison['seed']
        if identity in identities:
            raise ValueError('Repeated paired comparison seed.')
        identities.add(identity)
        control = report[comparison['control']]['pooled']
        candidate = report[comparison['candidate']]['pooled']
        if control['nse'] is None or candidate['nse'] is None:
            raise ValueError('Undefined NSE in paired seed comparison.')
        groups.setdefault(comparison['name'], []).append(candidate['nse']-control['nse'])
    return {name:{'n_seeds':len(values),'mean_delta_nse':mean(values),
                  'population_std_delta_nse':pstdev(values) if len(values)>1 else None,
                  'positive_seeds':sum(value>0 for value in values)} for name,values in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',action='append',required=True,help='unique label=approach=/path/predictions.csv; labels may encode seed')
    parser.add_argument('--support',required=True)
    parser.add_argument('--training-observations',required=True)
    parser.add_argument('--train-end',required=True,type=timestamp)
    parser.add_argument('--events',help='prospectively frozen JSON list, never selected by test skill')
    parser.add_argument('--paired-comparisons',help='frozen JSON name/seed/control/candidate records; no duplicated deterministic fits')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        parser.error('Refusing existing output.')
    support = load_support(args.support)
    methods, approaches, files = {}, {}, [args.support,args.training_observations]
    for specification in args.method:
        name,approach,path = specification.split('=',2)
        if name in methods:
            raise ValueError('Duplicate method label.')
        methods[name] = load_predictions(path,approach);approaches[name]=approach;files.append(path)
    events = json.loads(Path(args.events).read_text()) if args.events else []
    if len({e['name'] for e in events}) != len(events):
        raise ValueError('Duplicate event name.')
    if args.events:
        files.append(args.events)
    threshold = training_threshold(args.training_observations,args.train_end)
    report = {'units':'raw mm/h; event volume mm; positive PBIAS means overprediction',
              'method_approaches':approaches,
              'comparison_note':'ffn_hydrofusion is a compatibility path with broader training support; use ffn_causal_forecast for matched architecture comparisons. No synthetic skill claim.',
              'q90_training_only':threshold,'support_sha256':hashlib.sha256(('\n'.join(map(str,support))+'\n').encode()).hexdigest(),
              'input_sha256':{str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files},
              'limitations':'Native provenance/managed bundle integrity must be verified upstream. No accuracy conclusion from synthetic checks.',
              'methods':evaluate(methods,support,threshold,events)}
    if args.paired_comparisons:
        pairs = json.loads(Path(args.paired_comparisons).read_text())
        report['paired_seed_deltas'] = paired_seed_summary(report['methods'], pairs)
        report['input_sha256'][args.paired_comparisons] = hashlib.sha256(Path(args.paired_comparisons).read_bytes()).hexdigest()
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
