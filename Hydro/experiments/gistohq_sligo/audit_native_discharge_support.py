#!/usr/bin/env python3
"""Audit original discharge support and interpolation across proposed cutoffs."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    epoch=datetime(2015,1,1,tzinfo=timezone.utc).timestamp()
    times=[]
    with args.csv.open() as f:
        for row in csv.DictReader(f):
            t=int(datetime.fromisoformat(row['timestamp_utc'].replace('Z','+00:00')).timestamp()-epoch)
            if t>=77016*3600:raise ValueError('Reserved-period source row')
            if times and t<=times[-1]:raise ValueError('Unordered or duplicate native timestamps')
            times.append(t)
    covered=defaultdict(int)
    for i,t in enumerate(times):
        end=min(t+300,times[i+1] if i+1<len(times) else t+300)
        while t<end:
            h=t//3600;stop=min(end,(h+1)*3600);covered[h]+=stop-t;t=stop
    checks=[]
    for fold in json.loads(args.protocol.read_text())['folds']:
        for key in ['train_end_x','validation_end_x']:
            cutoff=(fold[key]+1)*3600
            spans=[(a,b) for a,b in zip(times,times[1:]) if a<cutoff<=b and 450<b-a<=10800]
            checks.append(dict(year=fold['name'],boundary=key,interpolation_spans=spans))
    strict=sorted(h for h,s in covered.items() if s>=3600)
    args.output.write_text(json.dumps(dict(sample_support_seconds=300,
        strict_observed_hour_x=strict,boundary_interpolation_checks=checks,
        note='Native-only coverage under the harmonizer five-minute forward-support convention; excludes interpolated target hours.'),indent=2)+'\n')
    print(f'{len(strict)} native-supported hours; {sum(bool(c["interpolation_spans"]) for c in checks)} crossing interpolation gaps')


if __name__=='__main__':main()
