#!/usr/bin/env python3
"""Describe existing bounded-network output ceilings, without changing weights."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from analyze_hydrofusion_all_methods import sha
import evaluate_causal_forecast_methods as e


def analyze(source,capacity,output):
    source,capacity,output=map(Path,(source,capacity,output))
    if output.exists():raise FileExistsError(output)
    observed={}
    for split in ('train','validation','test'):
        support=e.load_support(source/f'frozen_support/{split}_support.csv')
        rows=e.csv_rows(source/f'frozen_support/{split}_observations.csv',['x','observed'])
        if [e.timestamp(r['x']) for r in rows]!=support:raise ValueError('Observation support mismatch')
        observed[split]=[e.finite_nonnegative(r['observed']) for r in rows]
    models=[];identities=set()
    with capacity.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['bundle'] in identities:raise ValueError('Duplicate checkpoint')
            identities.add(r['bundle'])
            qref=float(r['q_ref']);upper=float(r['maximum_transformed_output']);lower=float(r['minimum_transformed_output'])
            if not all(math.isfinite(v) for v in (qref,upper,lower)) or qref<=0 or upper<lower:raise ValueError('Invalid checkpoint bound')
            if r['target_mode']!='direct_log_q':continue
            cap=max(0,qref*math.expm1(upper));splits={}
            for split,q in observed.items():
                mu=mean(q);unavoidable=sum(max(0,v-cap)**2 for v in q);variance=sum((v-mu)**2 for v in q)
                splits[split]=dict(n=len(q),observed_max_mm_h=max(q),count_above_ceiling=sum(v>cap+1e-5 for v in q),
                                   minimum_sse_from_ceiling_alone=unavoidable,nse_upper_bound=1-unavoidable/variance if variance else None)
            models.append(dict(bundle=r['bundle'],approach=r['approach'],parameters=int(r['parameters']),ceiling_mm_h=cap,splits=splits))
    result=dict(definition='Current weights and scalers imply this real-arithmetic upper bound because final hidden values are bounded by1. Comparison margin1e-5. It is not an immutable architecture-wide ceiling; refitting can change weights.',
                models=models,source_sha256=sha(capacity),script_sha256=sha(__file__),weights_changed=False)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('Saved checkpoint ceiling diagnosis:',output);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','capacity','output'):p.add_argument(k)
    a=p.parse_args();analyze(a.source,a.capacity,a.output)
