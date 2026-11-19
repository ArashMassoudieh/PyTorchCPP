#!/usr/bin/env python3
"""Read-only flow-volume/reference/target-space diagnosis; no correction fitting."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean,pstdev
from analyze_hydrofusion_all_methods import sha,quantile
import evaluate_causal_forecast_methods as e


def native_split(x):
    x=e.timestamp(x)
    if x<=17543:return 'train'
    if x<=26303:return 'validation'
    if x<=35063:return 'test'
    raise ValueError('Outside fixed diagnostic study')


def summarize_neural(rows,q90):
    if not rows:return dict(n=0)
    observed=[r['observed'] for r in rows];predicted=[r['predicted'] for r in rows]
    target=[r['transformed_target'] for r in rows];raw=[r['transformed_prediction'] for r in rows]
    mse=mean((a-b)**2 for a,b in zip(target,raw));variance=pstdev(target)**2
    return dict(n=len(rows),physical=e.metrics(observed,predicted,q90),
                clipped_fraction=mean(r['clipped'] for r in rows),
                target_space=dict(mean_target=mean(target),mean_prediction=mean(raw),mean_error=mean(raw)-mean(target),
                                  target_std=pstdev(target),prediction_std=pstdev(raw),mse=mse,
                                  variance_explained=1-mse/variance if variance else None))


def analyze(source,extension,root,output):
    source,extension,root,output=map(Path,(source,extension,root,output))
    if output.exists():raise FileExistsError(output)
    support={split:e.load_support(source/f'frozen_support/{split}_support.csv') for split in ('train','validation','test')}
    observed={split:{e.timestamp(r['x']):e.finite_nonnegative(r['observed']) for r in e.csv_rows(source/f'frozen_support/{split}_observations.csv',['x','observed'])} for split in support}
    train=list(observed['train'].values());q90=quantile(train,.9)
    physical=list(csv.DictReader((root/'physical_inputs.csv').open()))
    by_target={}
    for r in physical:
        origin=e.timestamp(r['origin_x']);target=e.timestamp(r['target_x'])
        if target!=origin+1 or target in by_target:raise ValueError('Invalid physical sample identity')
        by_target[target]=r
    native_manifest=json.loads((source/'prepared_input/native_input_manifest.json').read_text())
    if sha(source/'prepared_input/intervals.csv')!=native_manifest['interval_csv_sha256']:raise ValueError('Native source changed')
    raw={}
    with (source/'prepared_input/intervals.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            t=int(row['epoch_seconds'])
            if t<1420070400 or t>=1420070400+35064*3600 or t in raw:raise ValueError('Native diagnostic timestamp domain')
            raw[t]=row
    pet={int(day['day_start_epoch_seconds'])+86400:day for day in native_manifest['daily_pet']}
    max_conversion=0.;qchecks=0;max_rain=0.;max_pet=0.;max_ledger=0.
    for target,row in by_target.items():
        x=target-1;original=raw[1420070400+(x-1)*3600]
        if row['q_valid']=='1':
            expected=float(original['q_m3_s'])*3600000/native_manifest['catchment_area_m2']
            max_conversion=max(max_conversion,abs(expected-float(row['origin_q'])));qchecks+=1
        forcing=raw[1420070400+(x-1-native_manifest['hourly_forcing_delay_hours'])*3600]
        max_rain=max(max_rain,abs(float(forcing['p_mm_h'])-float(row['p_mm_h'])))
        day_end=(1420070400+x*3600-native_manifest['daily_pet_delay_hours']*3600)//86400*86400
        max_pet=max(max_pet,abs(float(pet[day_end]['energy_mj_m2_day'])/2.45/24-float(row['pet_mm_h'])))
        ledger=float(row['storage_before'])+float(row['effective_mm_h'])-float(row['interval_mean_q'])-float(row['storage_after'])
        if not math.isfinite(ledger):raise ValueError('Nonfinite physical ledger')
        max_ledger=max(max_ledger,abs(ledger))
    if max_conversion>1e-12 or max_rain>1e-12 or max_pet>1e-12 or max_ledger>1e-9:raise ValueError('Physical input/ledger reproduction failed')
    checks=dict(discharge_rows=qchecks,forcing_rows=len(by_target),max_abs_unit_conversion_error=max_conversion,
                max_abs_rain_mapping_error=max_rain,max_abs_pet_conversion_error=max_pet,max_abs_hourly_ledger_error=max_ledger)
    phys={}
    for split,times in support.items():
        chosen=[by_target[x] for x in times]
        if any(r['eligible']!='1' for r in chosen):raise ValueError('Physical diagnostic missing eligible support')
        obs=[observed[split][x] for x in times]
        sums={key:sum(float(r[key]) for r in chosen) for key in ('p_mm_h','pet_mm_h','effective_mm_h','endpoint_q','interval_mean_q')}
        sums['observed_mm']=sum(obs)
        sums['rain_removed_by_instantaneous_pet_mm']=sums['p_mm_h']-sums['effective_mm_h']
        sums['pet_not_applied_to_stored_water_mm']=sums['pet_mm_h']-sums['rain_removed_by_instantaneous_pet_mm']
        models={key:e.metrics(obs,[float(r[key]) for r in chosen],q90) for key in ('endpoint_q','interval_mean_q')}
        if split!='train':
            bundle=extension/'fits/pinn_causal_forecast_log_residual_s42'
            path=bundle/'validation_predictions.csv' if split=='validation' else bundle/'bundles/pinn_causal_forecast_log_residual_s42/predictions.csv'
            old=e.load_predictions(path,'pinn_causal_forecast',split)
            if set(old)!=set(times) or any(abs(float(by_target[x]['endpoint_q'])-old[x][1])>1e-10 or observed[split][x]!=old[x][0] for x in times):raise ValueError('Physical production export parity failed')
        phys[split]=dict(n=len(times),matched_hour_sums_mm=sums,effective_input_to_observed_ratio=sums['effective_mm_h']/sum(obs),fixed_reference_metrics=models)
    manifest=json.loads((root/'neural_export_verification.json').read_text());neural={}
    for j in manifest['jobs']:
        path=Path(j['csv'])
        if sha(path)!=j['csv_sha256']:raise ValueError('Diagnostic CSV changed')
        groups={s:[] for s in support}
        with path.open(newline='') as f:
            reader=csv.DictReader(f)
            for r in reader:
                target=e.timestamp(r['target_x']);origin=e.timestamp(r['origin_x']);split=native_split(target)
                if r['split']!=split or target!=origin+1 or r['approach']!=j['family']:raise ValueError('Neural diagnostic identity mismatch')
                if target not in observed[split]:raise ValueError('Unexpected support')
                values={k:float(r[k]) for k in ('observed','reference','q_ref','transformed_target','transformed_prediction','predicted','clipped')}
                if not all(math.isfinite(v) for v in values.values()) or min(values[k] for k in ('observed','reference','predicted'))<0 or values['q_ref']<=0 or values['clipped'] not in (0,1):raise ValueError('Invalid neural diagnostic numeric domain')
                if values['observed']!=observed[split][target]:raise ValueError('Observed disagreement')
                expected=math.log1p(values['observed']/values['q_ref'])-(math.log1p(values['reference']/values['q_ref']) if j['mode']=='log_residual' else 0)
                if not math.isclose(expected,values['transformed_target'],abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Transformed target mismatch')
                z=values['transformed_prediction']+(math.log1p(values['reference']/values['q_ref']) if j['mode']=='log_residual' else 0)
                q=max(0,values['q_ref']*math.expm1(z))
                if not math.isclose(q,values['predicted'],rel_tol=1e-5,abs_tol=1e-5):raise ValueError('Reconstruction mismatch')
                if values['clipped']!=int(z<0):raise ValueError('Clipping flag mismatch')
                groups[split].append(dict(values,target_x=target))
        report={}
        for split,rows in groups.items():
            if [r['target_x'] for r in rows]!=support[split]:raise ValueError('Duplicate/unordered/missing diagnostic support')
            report[split]=summarize_neural(rows,q90)
            report[split]['high_flow']=summarize_neural([r for r in rows if r['observed']>=q90],q90)
        neural[j['name']]=dict(family=j['family'],mode=j['mode'],seed=j['seed'],splits=report)
    result=dict(schema='HydroFusionErrorDiagnosis-v1',fitting_performed=False,original_outputs_modified=False,
                q90_training_only=q90,physical_input_checks=checks,physical=phys,neural=neural,
                input_sha256={str(p):sha(p) for p in [root/'physical_inputs.csv',root/'neural_export_verification.json',Path(__file__)]},
                limits=['Matched-hour sums are not a complete-year catchment water balance.',
                       'Interval-mean rescoring is a fixed-equation diagnostic, not new calibration or an accepted new model.',
                       'Descriptive target-space patterns do not identify a unique causal loss-function mechanism.',
                       'No test-selected coefficients, loss tuning, additional fitting or paper promotion.'])
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('Saved verified diagnosis:',output);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','extension','root','output'):p.add_argument(key)
    a=p.parse_args();analyze(a.source,a.extension,a.root,a.output)
