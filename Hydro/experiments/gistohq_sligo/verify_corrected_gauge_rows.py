#!/usr/bin/env python3
"""Independent read-only checks of C++-decoded native-v2 rows (no fitting)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from prepare_gauge_alignment_review import validate_identity_csv

EPOCH=1420070400


def verify(manifest_path,decoded_csv,output):
    manifest_path,decoded_csv,output=map(Path,(manifest_path,decoded_csv,output))
    if output.exists():raise FileExistsError(output)
    m=json.loads(manifest_path.read_text())
    first,end=m['first_interval_x'],m['end_interval_x_exclusive']
    if m['schema']!='HydroFusionPreparedNative-v2' or type(first) is not int or type(end) is not int or not 0<=first<end<77016:
        raise ValueError('Bounded native-v2 input required')
    path=manifest_path.parent/'intervals.csv'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=m['interval_csv_sha256']:
        raise ValueError('Native CSV hash mismatch')
    validate_identity_csv(path,first,end)
    raw={int(row['epoch_seconds']):row for row in csv.DictReader(path.open())}
    if not math.isfinite(m['catchment_area_m2']) or m['catchment_area_m2']<=0:raise ValueError('Invalid normalization area')
    for native in raw.values():
        if not all(math.isfinite(float(native[k])) for k in ['q_m3_s','p_mm_h','t_c','rh_pct','wind_m_s','solar_mj_m2_h','pet_mm_h','native_coverage']):
            raise ValueError('Nonfinite native value')
    daily={d['day_start_epoch_seconds']+86400:d for d in m['daily_pet']}
    if any(not math.isfinite(d['reference_et_mm_day']) or d['reference_et_mm_day']<0 for d in daily.values()):
        raise ValueError('Invalid reference ET')
    errors={name:0. for name in ['q_depth','lagged_rain','completed_reference_et','temperature','humidity','wind','solar']}
    rows=list(csv.DictReader(decoded_csv.open()))
    if len(rows)!=m['row_count']:raise ValueError('Decoded row count mismatch')
    for n,row in enumerate(rows):
        if not all(math.isfinite(float(row[k])) for k in ['x','q','p','t','rh','wind','solar','pet']):
            raise ValueError('Nonfinite decoded value')
        if any(row[k] not in {'0','1'} for k in ['forcing_valid','discharge_valid']):
            raise ValueError('Invalid decoded mask')
        x=int(row['x'])
        if x!=first+n+1:raise ValueError('Decoded endpoint identity mismatch')
        start=EPOCH+(x-1)*3600
        original=raw[start]
        expected_q_valid=(original['discharge_valid']=='1' and original['native']=='1' and original['interpolated']=='0' and float(original['native_coverage'])>=m.get('minimum_native_coverage',.75) and float(original['q_m3_s'])>=0)
        if (row['discharge_valid']=='1')!=expected_q_valid:raise ValueError('Native discharge mask mismatch')
        if row['discharge_valid']=='1':
            expected=float(original['q_m3_s'])*3600000/m['catchment_area_m2']
            errors['q_depth']=max(errors['q_depth'],abs(float(row['q'])-expected))
        prior=raw.get(start-m['hourly_forcing_delay_hours']*3600)
        if prior is not None:
            for output_field,native_field,error in [('p','p_mm_h','lagged_rain'),('t','t_c','temperature'),
                    ('rh','rh_pct','humidity'),('wind','wind_m_s','wind'),('solar','solar_mj_m2_h','solar')]:
                errors[error]=max(errors[error],abs(float(row[output_field])-float(prior[native_field])))
        offset=m['daily_pet_utc_start_hour']*3600
        expected_end=(start+3600-m['daily_pet_delay_hours']*3600-offset)//86400*86400+offset
        day=daily.get(expected_end)
        expected_pet=day['reference_et_mm_day']/24 if day is not None and day['valid'] else 0.
        errors['completed_reference_et']=max(errors['completed_reference_et'],abs(float(row['pet'])-expected_pet))
        expected_valid=prior is not None and prior['forcing_valid']=='1' and day is not None and day['valid']
        if (row['forcing_valid']=='1')!=expected_valid:raise ValueError('Causal forcing mask mismatch')
    if not all(math.isfinite(v) and v<1e-12 for v in errors.values()):raise ValueError('Decoded units/time mapping mismatch: '+str(errors))
    report={'rows_checked':len(rows),'max_absolute_errors':errors,'training_performed':False,
            'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'decoded_rows_sha256':hashlib.sha256(decoded_csv.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','decoded_csv','new_report']:parser.add_argument(key,type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.manifest,args.decoded_csv,args.new_report),indent=2))
