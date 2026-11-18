#!/usr/bin/env python3
"""Inventory development CSV coverage without opening any full/holdout package."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

SEAL = datetime(2023,10,15,tzinfo=timezone.utc)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.package/'manifest.json').read_text())
    if datetime.fromisoformat(manifest['study_end'].replace('Z','+00:00')) >= SEAL:
        raise ValueError('Refusing a package that declares reserved-period coverage')
    data={}
    for asset in manifest['assets']:
        relative=Path(asset['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe asset path')
        if relative.suffix != '.csv':
            continue
        groups=defaultdict(lambda:dict(rows=0,nonfinite=0,units=set(),first=None,last=None,years=defaultdict(set)))
        with (args.package/relative).open() as f:
            for row in csv.DictReader(f):
                dt=datetime.fromisoformat(row['timestamp_utc'].replace('Z','+00:00'))
                if dt >= SEAL:
                    raise ValueError('Development package contains reserved-period timestamp')
                g=groups[row['variable']]
                g['rows']+=1
                g['nonfinite']+=int(not math.isfinite(float(row['value'])))
                g['units'].add(row['native_unit'])
                g['first']=min(g['first'],dt) if g['first'] else dt
                g['last']=max(g['last'],dt) if g['last'] else dt
                g['years'][dt.year].add(dt.replace(minute=0,second=0,microsecond=0))
        for name,g in groups.items():
            data[f'{relative}:{name}']=dict(rows=g['rows'],nonfinite=g['nonfinite'],units=sorted(g['units']),first=g['first'].isoformat(),last=g['last'].isoformat(),occupied_hours_by_year={y:len(v) for y,v in g['years'].items()})
    args.output.write_text(json.dumps(dict(package=str(args.package),manifest_qc_gate=manifest.get('qc_gate'),catchment_area_m2=manifest['catchment_area_m2'],variables=data,note='Occupied timestamp bins are not harmonized valid model hours; daily PET is not hourly coverage.'),indent=2)+'\n')
    print(f'Checked development timestamps and numeric finiteness for {len(data)} variable streams')


if __name__=='__main__':
    main()
