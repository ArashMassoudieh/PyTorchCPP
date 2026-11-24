#!/usr/bin/env python3
"""NEW finite validation-only preissue-rain pilot: eight neural fits + two fixed physics runs."""
import argparse
import csv
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from run_hydrofusion_six_fit_screen import bounded_fit, digest, write_new, directory_bytes
from evaluate_causal_forecast_methods import metrics, load_predictions, timestamp, finite_nonnegative
HERE=Path(__file__).resolve().parent
FAMILIES=('ffn_causal_forecast','lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast','pinn_causal_forecast')
RECIPES=('zero_channel_v1','rate_hold_v1')
LIMITS=dict(epochs=50,wall_seconds_per_fit=28800,rss_bytes=12*1024**3,threads=2,gpu=False,artifact_bytes_per_fit=250000000)
def context(f):return 'legacy20' if f=='pinn_causal_forecast' else 'anchored25' if 'physical' in f else 'zero25'
def support(root):
    out={s:{} for s in ('train','validation')}
    with (root/'support.csv').open() as f:
        for r in csv.DictReader(f):
            s=r['split'];x=timestamp(r['target_x'])
            if s not in out or x in out[s] or not (0<x<=17543 if s=='train' else 17543<x<=26303):raise ValueError('Invalid development support')
            out[s][x]=finite_nonnegative(r['observed'])
    if list(map(len,out.values()))!=[13072,7112]:raise ValueError('Matched rainfall support mismatch')
    return out

def verify(plan):
    if plan['limits']!=LIMITS or len(plan['jobs'])!=10 or {(j['family'],j['recipe']) for j in plan['jobs']}!={(f,r) for f in FAMILIES for r in RECIPES}:raise ValueError('Finite job inventory changed')
    for p,h in plan['pins'].items():
        if digest(p)!=h:raise ValueError('Pinned input changed: '+p)
    for j in plan['jobs']:
        c=json.loads(Path(j['config']).read_text());f=j['family']
        expected=dict(epochs=50,hidden_layers='32',batch_size=64,learning_rate=.001,random_seed=42,
                      train_end_x=17543,validation_end_x=26303,test_end_x=35063,lstm_sequence_length=24,
                      hydrofusion_synthetic_verification=False,hydrofusion_development_only=True,
                      hydrofusion_context_recipe=context(f),hydrofusion_history_hours='0,1,3,6,12,23' if f.startswith('ffn') else '0',
                      hydrofusion_preissue_rain_recipe=j['recipe'],hydrofusion_target_mode='log_residual',
                      hydrofusion_objective='log_mse_v1' if f=='pinn_causal_forecast' else 'log_flow_mse_v2',
                      physics_profile='causal_valid_time_dry_v3',hydrofusion_ffn_activation='tanh',
                      storage_coeff=1.,lambda_decay=.1,runoff_coeff=.7,forcing_gain=1.,use_hydro_package=False,use_csv_data=False)
        if f=='pinn_causal_forecast':expected['hydrofusion_history_hours']='0'
        if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Prospective recipe changed')
        for key,path in [('hydrofusion_native_manifest',plan['manifest']),('hydrofusion_preissue_rain',plan['weather'])]:
            if c[key+'_path']!=path or c[key+'_sha256']!=digest(path):raise ValueError('Bound input/config mismatch')
        if c['hydrofusion_preissue_rain_provenance_sha256']!=digest(Path(plan['weather']).parent/'preissue_rain_provenance.json'):raise ValueError('Weather provenance mismatch')

def prepare(root,manifest,weather,support_csv,binary,verifier,preflight):
    root=Path(root).absolute();root.mkdir(parents=True,exist_ok=False)
    p=json.loads(Path(preflight).read_text())
    if p.get('status')!='VERIFIED' or p['binary_sha256']!=digest(binary) or p['native_verifier_sha256']!=digest(verifier) or not p.get('native_feature_only_pass'):raise ValueError('Verified native and synthetic preflight required')
    shutil.copytree(Path(manifest).parent,root/'prepared_input');shutil.copytree(Path(weather).parent,root/'weather')
    m=root/'prepared_input/native_input_manifest.json';w=root/'weather/preissue_rain.csv';meta=json.loads(m.read_text())
    if meta.get('execution_scope')!='train_validation_only' or meta['target_cutoff_x']!=26303 or meta['end_interval_x_exclusive']!=26303:raise ValueError('Uncropped source rejected')
    for name in ('executables','configs','provenance'):(root/name).mkdir()
    for src,name in [(binary,'HydroBatch'),(verifier,'native_reload')]:shutil.copy2(src,root/'executables'/name)
    shutil.copy2(support_csv,root/'support.csv');support(root)
    for src,name in [(preflight,'preflight.json'),(str(binary)+'.build/source_manifest.json','batch_build.json'),(str(verifier)+'.link_manifest.json','reload_build.json')]:shutil.copy2(src,root/'provenance'/name)
    for src in (Path(__file__),HERE/'run_hydrofusion_six_fit_screen.py',HERE/'evaluate_causal_forecast_methods.py'):shutil.copy2(src,root/'provenance'/src.name)
    dispatch=HERE.parents[2]/'build-hydrobatch'/('preissue_screen_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
    jobs=[]
    for family in FAMILIES:
        for recipe in RECIPES:
            name=family+'_'+recipe+'_s42';c=json.loads((HERE/'hydrofusion_five_method_checks'/(family+'_log_residual.json')).read_text())
            c.update(experiment_id=name,epochs=50,hidden_layers='32',batch_size=64,learning_rate=.001,random_seed=42,
                     train_end_x=17543,validation_end_x=26303,test_end_x=35063,lstm_sequence_length=24,
                     hydrofusion_synthetic_verification=False,hydrofusion_development_only=True,
                     hydrofusion_context_recipe=context(family),hydrofusion_history_hours='0,1,3,6,12,23' if family.startswith('ffn') else '0',
                     hydrofusion_preissue_rain_recipe=recipe,hydrofusion_target_mode='log_residual',
                     hydrofusion_objective='log_mse_v1' if family=='pinn_causal_forecast' else 'log_flow_mse_v2',
                     physics_profile='causal_valid_time_dry_v3',hydrofusion_ffn_activation='tanh',
                     storage_coeff=1.,lambda_decay=.1,runoff_coeff=.7,forcing_gain=1.,
                     hydrofusion_native_manifest_path=str(m),hydrofusion_native_manifest_sha256=digest(m),
                     hydrofusion_preissue_rain_path=str(w),hydrofusion_preissue_rain_sha256=digest(w),
                     hydrofusion_preissue_rain_provenance_sha256=digest(w.parent/'preissue_rain_provenance.json'))
            cp=dispatch/(name+'.json');write_new(cp,c);shutil.copy2(cp,root/'configs'/cp.name)
            bp=dispatch/(name+'.batch');bp.write_text(f'{family} {cp.name}\n');shutil.copy2(bp,root/'configs'/bp.name)
            jobs.append(dict(name=name,family=family,recipe=recipe,seed=42,config=str(cp),batch=str(bp),neural=family!='pinn_causal_forecast'))
    paths=[p for folder in ('prepared_input','weather','executables','configs','provenance') for p in (root/folder).rglob('*') if p.is_file()]
    paths+=[root/'support.csv',Path(__file__),HERE/'run_hydrofusion_six_fit_screen.py',HERE/'evaluate_causal_forecast_methods.py']+[Path(j[k]) for j in jobs for k in ('config','batch')]
    plan=dict(schema='HydroFusionPreissueValidationPilot-v1',approval='Standing human autonomous research authorization; latest Go ahead24Nov',
              jobs=jobs,limits=LIMITS,manifest=str(m),weather=str(w),binary=str(root/'executables/HydroBatch'),verifier=str(root/'executables/native_reload'),
              source_commit=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD'],text=True).strip(),
              development_only=True,test_scores=0,neural_fits=8,fixed_physical_runs=2,no_automatic_expansion=True,
              goals=dict(nse=.8,r_squared=.8,acceptable=.7),gates=dict(nse_gain=.02,r_squared_gain=.01,mae_relative_increase=.02,high_flow_rmse_relative_increase=.02,absolute_pbias_increase_points=2),
              limitations=['Reused validation; not blind generalization or a paper promotion.','Archive HTTP timestamps are release proxies, not verified operational delivery.','Available600s-old rain rate held over next hour; not future observed rainfall.','All24historical rain samples required symmetrically; missing is never zero filled.','One seed exploratory; no automatic confirmation or reruns.','Hybrid reference anchored; plain physical-context channels zero; same26-channel neural width.'],pins={str(p):digest(p) for p in paths})
    write_new(root/'screen_plan.json',plan);verify(plan);print('FROZEN8neural fits+2fixed physics runs; NOT LAUNCHED')

def analyze(root,plan):
    common=support(root);v=sorted(common['train'].values());p=(len(v)-1)*.9;i=int(p);q90=v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
    scores={}
    for j in plan['jobs']:
        rows=load_predictions(root/'fits'/j['name']/'validation_predictions.csv',j['family'],'validation')
        if set(rows)!=set(common['validation']) or any(rows[x][0]!=common['validation'][x] for x in rows):raise ValueError('Unmatched validation identities/observations')
        obs=[rows[x][0] for x in sorted(rows)];pred=[rows[x][1] for x in sorted(rows)];s=metrics(obs,pred,q90)
        s['r_squared']=statistics.correlation(obs,pred)**2 if statistics.pstdev(pred)>0 else None;scores[j['name']]=s
    decisions={};g=plan['gates']
    for f in FAMILIES:
        a,b=(scores[f+'_'+r+'_s42'] for r in RECIPES)
        checks=dict(nse=b['nse']-a['nse']>=g['nse_gain'],r_squared=b['r_squared'] is not None and a['r_squared'] is not None and b['r_squared']-a['r_squared']>=g['r_squared_gain'],mae=b['mae']<=a['mae']*(1+g['mae_relative_increase']),high_flow=b['high_flow_rmse']<=a['high_flow_rmse']*(1+g['high_flow_rmse_relative_increase']),bias=abs(b['pbias_percent'])<=abs(a['pbias_percent'])+g['absolute_pbias_increase_points'])
        decisions[f]=dict(checks=checks,passed=all(checks.values()),goal_met=b['r_squared'] is not None and b['nse']>=.8 and b['r_squared']>=.8)
    write_new(root/'results_summary.json',dict(validation=scores,decisions=decisions,test_scores=0,training_q90=q90));return decisions

def launch(root):
    root=Path(root).resolve(strict=True);plan=json.loads((root/'screen_plan.json').read_text());verify(plan);common=support(root)
    write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_retry=True));(root/'fits').mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='');completed=[]
    try:
        for j in plan['jobs']:
            verify(plan);fit=root/'fits'/j['name'];fit.mkdir();print('START '+j['name'],flush=True)
            rec=bounded_fit([plan['binary'],j['batch'],str(fit/'bundles')],fit,env,LIMITS);bundle=fit/'bundles'/j['name']
            if j['neural']:
                with (bundle/'training_history.csv').open() as f:h=list(csv.DictReader(f))
                losses=[float(r['validation_loss']) for r in h]
                if len(h)!=50 or [int(r['epoch']) for r in h]!=list(range(1,51)) or not all(math.isfinite(x) for x in losses+[float(r['training_loss']) for r in h]):raise ValueError('Invalid50epoch history')
                if [i for i,r in enumerate(h) if r['selected_checkpoint']=='1']!=[min(range(50),key=lambda i:losses[i])]:raise ValueError('Selected epoch mismatch')
            with (fit/'fresh_reload.log').open('x') as log:subprocess.run([plan['verifier'],str(bundle),str(fit/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
            for p in (bundle/'predictions.csv',fit/'validation_predictions.csv'):
                rows=load_predictions(p,j['family'],'validation')
                with p.open() as f:n=sum(1 for _ in csv.DictReader(f))
                if n!=len(rows) or set(rows)!=set(common['validation']) or any(rows[x][0]!=common['validation'][x] for x in rows):raise ValueError('Extra/test/unmatched prediction rows')
            if directory_bytes(fit)>LIMITS['artifact_bytes_per_fit']:raise ValueError('Artifact cap')
            rec['sha256']={str(p.relative_to(fit)):digest(p) for p in fit.rglob('*') if p.is_file()};write_new(fit/'completion.json',rec);completed.append(j['name']);print('VERIFIED '+j['name'],flush=True)
        decisions=analyze(root,plan);write_new(root/'screen_completion.json',dict(completed=completed,plan_sha256=digest(root/'screen_plan.json'),decisions=decisions,no_automatic_expansion=True))
    except BaseException as e:write_new(root/'screen_failure.json',dict(completed=completed,error=str(e),no_retry=True));raise
def argument_parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',nargs=6,type=Path);p.add_argument('--launch',action='store_true');return p
if __name__=='__main__':
    p=argument_parser();a=p.parse_args()
    if bool(a.prepare)==a.launch:p.error('Choose prepare OR launch')
    if a.prepare:prepare(a.root,*a.prepare)
    else:launch(a.root)
