#!/usr/bin/env python3
"""Prepare pinned interval inputs; C++ causal adapter performs endpoint/lag mapping.

Retrospective idealized hindcast only, not proof of operational product latency.
No fitting or hyperparameter selection occurs here.
"""
import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from audit_hydrofusion_producer_package import audit, read_json, sha, timestamp, EPOCH

CSV_HEADER = 'epoch_seconds,q_m3_s,p_mm_h,t_c,rh_pct,wind_m_s,solar_mj_m2_h,pet_mm_h,forcing_valid,discharge_valid,native,interpolated,native_coverage,source_id'.split(',')
MET = [('PRECTOTCORR', {'mm/hour', 'mm/hr', 'mm/h'}), ('T2M', {'C', 'degC'}),
       ('RH2M', {'%', 'percent'}), ('WS2M', {'m/s'}),
       ('ALLSKY_SFC_SW_DWN', {'MJ/hr', 'MJ/m2/h', 'MJ/m^2/h'})]


def prepare(source, policy, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    result = audit(source, policy)
    variables = read_json((source / 'variables.json').read_bytes())
    declared = {v['name']: v['units'] for v in variables['variables']}
    if len(declared) != len(variables['variables']):
        raise ValueError('duplicate declared variable')
    for name, inventory in result['variable_inventory'].items():
        if declared.get(name) != inventory['units']:
            raise ValueError('variables.json mismatch: ' + name)
    for name, units in MET + [('EVPTRNS', {'MJ/m^2/day', 'MJ/m2/day'})]:
        if not set(result['variable_inventory'][name]['units']) <= units:
            raise ValueError('unsupported forcing units: ' + name)
    if result['source_qc_status'] not in ('pass', 'warning') or set(result['source_failed_qc_rule_ids'] or []) - {'temporal.study_period_coverage'}:
        raise ValueError('source QC has unaccepted failures')
    hourly = {}
    daily = []
    for path in sorted((source / 'observations').glob('*.csv')):
        # Read only pinned manifest assets, never incidental files.
        manifest = read_json((source / 'manifest.json').read_bytes())
        if path.relative_to(source).as_posix() not in {a['path'] for a in manifest['assets']}:
            raise ValueError('unlisted observation asset')
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                variable = row['variable']
                if variable == '00060':
                    continue
                t = timestamp(row['timestamp_utc'])
                if variable == 'EVPTRNS':
                    if t % 86400:
                        raise ValueError('PET must identify a UTC day start')
                    daily.append({'day_start_epoch_seconds': t, 'energy_mj_m2_day': float(row['value']) if row['value'] else 0,
                                  'valid': bool(row['value'])})
                elif variable in {m[0] for m in MET}:
                    if t % 3600:
                        raise ValueError('forcing must identify a whole-hour interval')
                    hourly.setdefault(t, {})[variable] = float(row['value']) if row['value'] else None
                else:
                    raise ValueError('unsupported variable: ' + variable)
    stage = Path(tempfile.mkdtemp(prefix='.prepared-native-', dir=destination.parent))
    try:
        csv_path = stage / 'intervals.csv'
        with csv_path.open('x', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(CSV_HEADER)
            for hour in result['native_discharge_hours']:
                t = EPOCH + hour['interval_start_x'] * 3600
                met = [hourly.get(t, {}).get(name) for name, _ in MET]
                valid = all(value is not None for value in met)
                qvalid = hour['eligible_discharge']
                writer.writerow([t, hour['q_m3_s'] if qvalid else 0] + [v if v is not None else 0 for v in met] +
                                [0, int(valid), int(qvalid), int(qvalid), 0, hour['native_coverage'], result['source_package_id']])
        (stage / 'source_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        manifest = {'schema': 'HydroFusionPreparedNative-v1', 'time_label': 'interval_end',
                    'study_type': 'idealized_retrospective_causal_hindcast',
                    'interval_csv': 'intervals.csv', 'interval_csv_sha256': sha(csv_path.read_bytes()),
                    'row_count': len(result['native_discharge_hours']),
                    'first_interval_x': result['native_discharge_hours'][0]['interval_start_x'],
                    'end_interval_x_exclusive': result['native_discharge_hours'][-1]['interval_end_x_exclusive'],
                    'catchment_area_m2': result['catchment_area_m2'], 'source_id': result['source_package_id'],
                    'minimum_native_coverage': policy['minimum_native_hour_coverage'],
                    'discharge_delay_hours': 0, 'hourly_forcing_delay_hours': 24, 'daily_pet_delay_hours': 24,
                    'availability_evidence': 'Frozen exploratory hindcast assumptions: zero Q release delay, 24h lagged hourly forcings, daily PET complete-day plus 24h lag. No claim these equal provider operational publication latencies.',
                    'daily_pet': daily, 'producer_audit_sha256': sha((stage/'source_audit.json').read_bytes()),
                    'producer_variables_sha256': sha((source/'variables.json').read_bytes()),
                    'producer_manifest_sha256': result['manifest_sha256'], 'policy_sha256': result['policy_sha256'],
                    'limitations': ['Original mixed-period assets not fully hashed; exact extracted prefixes pinned instead.',
                                    'Legacy POWER MJ/hr interpreted as surface energy density per hour per existing source contract.',
                                    'No actual provider delivery-time records; idealized retrospective hindcast only.']}
        (stage / 'native_input_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('policy', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    prepare(args.source, read_json(args.policy.read_bytes()), args.destination)
