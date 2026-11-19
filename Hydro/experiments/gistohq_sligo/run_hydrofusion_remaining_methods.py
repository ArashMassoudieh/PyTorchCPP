#!/usr/bin/env python3
"""Bounded all-method extension: twelve neural fits and one fixed physical run.

Uses the completed FFN stage's exact data/support. Exclusive claim, no retries,
no test-based recipe selection, and no changes to original campaign artifacts.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from run_hydrofusion_six_fit_screen import digest, write_new, bounded_fit, directory_bytes

HERE=Path(__file__).resolve().parent
LIMITS=dict(epochs=50,wall_seconds_per_fit=28800,rss_bytes=12*1024**3,threads=2,gpu=False,artifact_bytes_per_fit=250_000_000)
FAMILIES=('lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast','pinn_causal_forecast')


def specifications():
    return [(family,mode,seed) for family in FAMILIES for mode in
            (('direct_log_q','log_residual') if family=='lstm_hydrofusion' else ('log_residual',))
            for seed in ((42,) if family=='pinn_causal_forecast' else (42,43,44))]


def expected_config(source,root,family,mode,seed):
    c=json.loads((HERE/'hydrofusion_five_method_checks'/f'{family}_{mode}.json').read_text())
    native=root/'prepared_input/native_input_manifest.json'
    c.update(experiment_id=f'{family}_{mode}_s{seed}',epochs=50,batch_size=64,learning_rate=.001,
             hidden_layers='32',random_seed=seed,train_end_x=17543,validation_end_x=26303,test_end_x=35063,
             hydrofusion_synthetic_verification=False,hydrofusion_target_mode=mode,
             hydrofusion_native_manifest_path=str(native),hydrofusion_native_manifest_sha256=digest(native))
    return c


def prepare(source,root,binary,verifier):
    source=Path(source).resolve(strict=True);root=Path(root).absolute()
    binary=Path(binary).resolve(strict=True);verifier=Path(verifier).resolve(strict=True)
    old=json.loads((source/'screen_plan.json').read_text());done=json.loads((source/'screen_completion.json').read_text())
    if done['plan_sha256']!=digest(source/'screen_plan.json') or done['completed']!=[j['name'] for j in old['jobs']]:
        raise ValueError('Original six fits not complete')
    for path,h in old['input_sha256'].items():
        if digest(path)!=h:raise ValueError('Original pin changed: '+path)
    root.mkdir(exist_ok=False)
    for folder in ('prepared_input','frozen_support'):
        shutil.copytree(source/folder,root/folder)
    shutil.copy2(source/'native_policy.json',root/'native_policy.json')
    (root/'configs').mkdir()
    dispatch=HERE.parents[2]/'build-hydrobatch'/('remaining_methods_'+digest(source/'screen_plan.json')[:12])
    dispatch.mkdir(exist_ok=False,parents=True)
    jobs=[]
    for family,mode,seed in specifications():
        c=expected_config(source,root,family,mode,seed);name=c['experiment_id']
        config=dispatch/(name+'.json');batch=dispatch/(name+'.batch')
        write_new(config,c);batch.write_text(f'{family} {config.name}\n')
        shutil.copy2(config,root/'configs'/config.name);shutil.copy2(batch,root/'configs'/batch.name)
        jobs.append(dict(name=name,family=family,mode=mode,seed=seed,config=str(config),batch=str(batch),neural=family!='pinn_causal_forecast'))
    pins=[binary,verifier,Path(__file__),HERE/'run_hydrofusion_six_fit_screen.py',HERE/'hydrofusion_native_family_reproduction.cpp',HERE/'evaluate_causal_forecast_methods.py',source/'screen_plan.json',source/'screen_completion.json']
    pins += [p for folder in ('prepared_input','frozen_support','configs') for p in (root/folder).rglob('*') if p.is_file()]
    pins += [root/'native_policy.json']+[Path(j[k]) for j in jobs for k in ('config','batch')]
    pins += [HERE/'hydrofusion_five_method_checks'/f'{family}_{mode}.json' for family,mode,_ in specifications()]
    plan=dict(schema='HydroFusionRemainingMethods-v1',approval='User: Go ahead for all; bounded extension of saved five-method proposal',
              source_campaign=str(source),source_plan_sha256=digest(source/'screen_plan.json'),
              binary=str(binary),verifier=str(verifier),jobs=jobs,limits=LIMITS,
              neural_fits=12,deterministic_physical_runs=1,hidden_width=32,
              parameter_counts={'ffn':705,'lstm':6945,'physical':0},
              advancement=old['advancement'],input_sha256={str(p):digest(p) for p in pins},
              limitations=old['limitations']+['All extension settings fixed before extension scores. No parameter sweep or retries.',
              'Physical recipe is fixed and uncalibrated; one run, not three independent seeds.',
              'Width matching is not parameter matching: FFN705 versus LSTM6945 trainable parameters.',
              'No event windows prospectively frozen; no event-specific inference.'])
    write_new(root/'screen_plan.json',plan)
    validate(plan,root)


def validate(plan,root):
    if plan['schema']!='HydroFusionRemainingMethods-v1' or plan['limits']!=LIMITS:
        raise ValueError('Budget/schema changed')
    jobs=plan['jobs']
    if len(jobs)!=13 or [(j['family'],j['mode'],j['seed']) for j in jobs]!=specifications():
        raise ValueError('Exactly twelve neural fits and one physical run required')
    if len({j['name'] for j in jobs})!=13:raise ValueError('Duplicate job')
    for path,h in plan['input_sha256'].items():
        if digest(path)!=h:raise ValueError('Changed pin: '+path)
    for j in jobs:
        expected=expected_config(Path(plan['source_campaign']),root,j['family'],j['mode'],j['seed'])
        if json.loads(Path(j['config']).read_text())!=expected or Path(j['batch']).read_text()!=f"{j['family']} {Path(j['config']).name}\n":
            raise ValueError('Prospective job recipe changed')
        if j['neural']!=(j['family']!='pinn_causal_forecast'):raise ValueError('Job type changed')


def launch(root):
    root=Path(root).resolve(strict=True);plan=json.loads((root/'screen_plan.json').read_text());validate(plan,root)
    write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),no_resume=True))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='')
    (root/'fits').mkdir(exist_ok=False);completed=[]
    try:
        for j in plan['jobs']:
            validate(plan,root);fit=root/'fits'/j['name'];fit.mkdir()
            print('START '+j['name'],flush=True)
            record=bounded_fit([plan['binary'],j['batch'],str(fit/'bundles')],fit,env,plan['limits'])
            bundle=fit/'bundles'/j['name']
            if j['neural']:
                with (bundle/'training_history.csv').open(newline='') as f:history=list(csv.DictReader(f))
                if len(history)!=50 or [int(r['epoch']) for r in history]!=list(range(1,51)):raise ValueError('Expected50 epochs')
                loss=[float(r['validation_loss']) for r in history]
                if not all(math.isfinite(v) for v in loss+[float(r['training_loss']) for r in history]):raise ValueError('Nonfinite loss')
                selected=[i for i,r in enumerate(history) if r['selected_checkpoint']=='1']
                if selected!=[min(range(50),key=lambda i:loss[i])]:raise ValueError('Checkpoint selection changed')
                record['selected_epoch']=selected[0]+1
            else:record['selected_epoch']=None
            with (fit/'fresh_reload.log').open('x') as log:
                subprocess.run([plan['verifier'],str(bundle),str(fit/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
            record['retained_artifact_bytes']=directory_bytes(fit)
            if record['retained_artifact_bytes']>LIMITS['artifact_bytes_per_fit']:raise ValueError('Artifact ceiling')
            record['file_sha256']={str(p.relative_to(fit)):digest(p) for p in fit.rglob('*') if p.is_file()}
            write_new(fit/'completion.json',record);completed.append(j['name']);print('VERIFIED '+j['name'],flush=True)
        write_new(root/'screen_completion.json',dict(completed=completed,plan_sha256=digest(root/'screen_plan.json')))
    except BaseException as e:
        write_new(root/'screen_failure.json',dict(error=str(e),completed=completed,no_automatic_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    a=p.add_mutually_exclusive_group(required=True);a.add_argument('--prepare',nargs=3,metavar=('FFN_CAMPAIGN','BINARY','VERIFIER'));a.add_argument('--launch-approved-extension',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare(args.prepare[0],args.root,args.prepare[1],args.prepare[2])
    else:launch(args.root)
