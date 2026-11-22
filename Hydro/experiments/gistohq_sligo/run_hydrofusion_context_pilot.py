#!/usr/bin/env python3
"""Six NEW validation-only context/history fits. No retry, test scoring or expansion."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from run_hydrofusion_six_fit_screen import bounded_fit, digest, write_new, directory_bytes
from evaluate_causal_forecast_methods import metrics, load_predictions, timestamp, finite_nonnegative

HERE = Path(__file__).resolve().parent
LIMITS = dict(epochs=50, wall_seconds_per_fit=28800, rss_bytes=12*1024**3,
              threads=2, gpu=False, artifact_bytes_per_fit=250000000)
FAMILIES = ('ffn_physical_forecast', 'lstm_physical_forecast')
RECIPES = ('zero25', 'absolute25', 'anchored25')

def verify(plan):
    if plan['limits'] != LIMITS or {(j['family'], j['recipe'], j['seed']) for j in plan['jobs']} != {(f,r,42) for f in FAMILIES for r in RECIPES} or len(plan['jobs']) != 6:
        raise ValueError('Frozen six-fit inventory/resource mismatch')
    for path, expected in plan['pins'].items():
        if digest(path) != expected:
            raise ValueError('Pinned input changed: '+path)
    for j in plan['jobs']:
        c=json.loads(Path(j['config']).read_text())
        expected=dict(epochs=50,hidden_layers='32',batch_size=64,learning_rate=.001,random_seed=42,
                      train_end_x=17543,validation_end_x=26303,test_end_x=35063,
                      hydrofusion_development_only=True,hydrofusion_synthetic_verification=False,
                      hydrofusion_context_recipe=j['recipe'],hydrofusion_history_hours='0,1,3,6,12,23' if j['family'].startswith('ffn') else '0',
                      lstm_sequence_length=24,hydrofusion_objective='log_flow_mse_v2',
                      hydrofusion_target_mode='log_residual',physics_profile='causal_valid_time_dry_v3',
                      use_hydro_package=False,use_csv_data=False)
        if any(c.get(k)!=v for k,v in expected.items()):
            raise ValueError('Config does not match frozen pilot recipe')
        if c['hydrofusion_native_manifest_path'] != plan['manifest'] or c['hydrofusion_native_manifest_sha256'] != digest(plan['manifest']):
            raise ValueError('Development input/config mismatch')

def support(root):
    with (root/'support.csv').open() as f:
        records=list(csv.DictReader(f))
    out={s:{} for s in ('train','validation')}
    for r in records:
        s=r['split']
        if s not in out: raise ValueError('Test rows forbidden in search support')
        x=timestamp(r['target_x'])
        if (s=='train' and x>17543) or (s=='validation' and not 17543<x<=26303) or x in out[s]:
            raise ValueError('Ambiguous development support identity')
        out[s][x]=finite_nonnegative(r['observed'])
    if [len(out[s]) for s in ('train','validation')] != [14369,8476]:
        raise ValueError('Matched development support changed')
    return out

def prepare(root, manifest, frozen_support, binary, verifier, preflight):
    root=Path(root).absolute();root.mkdir(parents=True,exist_ok=False)
    accept=json.loads(Path(preflight).read_text())
    if accept.get('status')!='VERIFIED' or accept.get('binary_sha256')!=digest(binary) or accept.get('native_verifier_sha256')!=digest(verifier): raise ValueError('Completed independent preflight required')
    shutil.copytree(Path(manifest).parent,root/'prepared_input')
    m=root/'prepared_input/native_input_manifest.json';meta=json.loads(m.read_text())
    if meta['execution_scope']!='train_validation_only' or meta['target_cutoff_x']!=26303 or meta['end_interval_x_exclusive']!=26303:
        raise ValueError('Physically cropped train/validation package required')
    (root/'executables').mkdir();(root/'configs').mkdir();(root/'provenance').mkdir()
    for src,name in [(binary,'HydroBatch'),(verifier,'native_reload')]:shutil.copy2(src,root/'executables'/name)
    shutil.copy2(preflight,root/'provenance/preflight.json')
    for source,name in [(Path(str(binary)+'.build/source_manifest.json'),'build_source_manifest.json'),(Path(str(verifier)+'.link_manifest.json'),'native_verifier_link_manifest.json')]:
        shutil.copy2(source,root/'provenance'/name)
    with Path(frozen_support).open() as f,(root/'support.csv').open('x') as out:
        reader=csv.DictReader(f);writer=csv.DictWriter(out,fieldnames=['split','target_x','observed','persistence']);writer.writeheader()
        for r in reader:
            # Inspect identity/split before any Q conversion; later values are not parsed.
            x=timestamp(r['target_x'])
            if x<=26303 and r['split'] in ('train','validation'):writer.writerow(r)
    support(root)
    dispatch=HERE.parents[2]/'build-hydrobatch'/('context_pilot_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
    jobs=[]
    for family in FAMILIES:
        for recipe in RECIPES:
            name=family+'_'+recipe+'_s42'
            c=json.loads((HERE/'hydrofusion_five_method_checks'/(family+'_log_residual.json')).read_text())
            c.update(experiment_id=name,epochs=50,hidden_layers='32',batch_size=64,learning_rate=.001,random_seed=42,
                     train_end_x=17543,validation_end_x=26303,test_end_x=35063,
                     hydrofusion_synthetic_verification=False,hydrofusion_development_only=True,
                     hydrofusion_context_recipe=recipe,hydrofusion_history_hours='0,1,3,6,12,23' if family.startswith('ffn') else '0',
                     lstm_sequence_length=24,hydrofusion_native_manifest_path=str(m),hydrofusion_native_manifest_sha256=digest(m),
                     hydrofusion_objective='log_flow_mse_v2',hydrofusion_target_mode='log_residual',
                     physics_profile='causal_valid_time_dry_v3',storage_coeff=1.,lambda_decay=.1,runoff_coeff=.7,forcing_gain=1.)
            cp=dispatch/(name+'.json');write_new(cp,c);shutil.copy2(cp,root/'configs'/cp.name)
            bp=dispatch/(name+'.batch');bp.write_text(f'{family} {cp.name}\n');shutil.copy2(bp,root/'configs'/bp.name)
            jobs.append(dict(name=name,family=family,recipe=recipe,seed=42,config=str(cp),batch=str(bp)))
    for script in [Path(__file__),HERE/'run_hydrofusion_six_fit_screen.py',HERE/'evaluate_causal_forecast_methods.py']:
        shutil.copy2(script,root/'provenance'/script.name)
    paths=[p for folder in ['prepared_input','executables','configs','provenance'] for p in (root/folder).rglob('*') if p.is_file()]
    paths += [root/'support.csv',Path(__file__),HERE/'run_hydrofusion_six_fit_screen.py',HERE/'evaluate_causal_forecast_methods.py']+[Path(j[k]) for j in jobs for k in ('config','batch')]
    plan=dict(schema='HydroFusionContextHistoryDevelopmentPilot-v1',approval='Human: Do this work and supervise yourself till finding the solution',
              jobs=jobs,limits=LIMITS,manifest=str(m),binary=str(root/'executables/HydroBatch'),verifier=str(root/'executables/native_reload'),
              development_only=True,source_commit=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD'],text=True).strip(),goal=dict(nse=.8,r_squared=.8),
              gates=dict(nse_gain=.02,r_squared_gain=.01,maximum_mae_relative_increase=.02,maximum_high_flow_rmse_relative_increase=.02,maximum_absolute_pbias_increase_percentage_points=2),
              no_automatic_expansion=True,pins={str(p):digest(p) for p in paths},
              limitations=['Reused validation fold, not blind generalization.','All arms have identical 25-channel width and FFN lag inventory; zero channels are an information ablation, not equal effective capacity.','Current Q assumed available; radar release assumed1h; no operational certification.','No manuscript promotion from a single-seed screen.'])
    write_new(root/'screen_plan.json',plan);verify(plan)
    print('FROZEN six NEW train/validation-only fits; not launched')

def analyze(root, plan):
    common=support(root);train=sorted(common['train'].values());pos=(len(train)-1)*.9;lo=int(pos);q90=train[lo]+(train[min(lo+1,len(train)-1)]-train[lo])*(pos-lo)
    results={}
    for j in plan['jobs']:
        vals=load_predictions(root/'fits'/j['name']/'validation_predictions.csv',j['family'],'validation')
        if set(vals)!=set(common['validation']) or any(vals[x][0]!=common['validation'][x] for x in vals):raise ValueError('Unmatched validation support')
        observed=[vals[x][0] for x in sorted(vals)];predicted=[vals[x][1] for x in sorted(vals)]
        m=metrics(observed,predicted,q90);m['r_squared']=statistics.correlation(observed,predicted)**2 if statistics.pstdev(predicted)>0 else None
        results[j['name']]=m
    decisions={}
    for family in FAMILIES:
        a=results[family+'_zero25_s42'];choices=[];details={}
        for recipe in RECIPES[1:]:
            b=results[family+'_'+recipe+'_s42'];g=plan['gates']
            checks=dict(nse=b['nse']-a['nse']>=g['nse_gain'],r2=b['r_squared'] is not None and a['r_squared'] is not None and b['r_squared']-a['r_squared']>=g['r_squared_gain'],
                        mae=b['mae']<=a['mae']*(1+g['maximum_mae_relative_increase']),
                        high_flow=b['high_flow_rmse']<=a['high_flow_rmse']*(1+g['maximum_high_flow_rmse_relative_increase']),
                        bias=abs(b['pbias_percent'])<=abs(a['pbias_percent'])+g['maximum_absolute_pbias_increase_percentage_points'])
            details[recipe]=dict(checks=checks,passed=all(checks.values()),goal_met=b['r_squared'] is not None and b['nse']>=plan.get('goal',dict(nse=.8,r_squared=.8))['nse'] and b['r_squared']>=plan.get('goal',dict(nse=.8,r_squared=.8))['r_squared'])
            if all(checks.values()):choices.append((b['nse'],-b['mae'],-int(recipe=='anchored25'),recipe))
        decisions[family]=dict(variants=details,selected_recipe=max(choices)[-1] if choices else None,confirmation_launched=False)
    write_new(root/'results_summary.json',dict(validation=results,decisions=decisions,test_scores=0,training_q90=q90))
    return decisions

def launch(root):
    root=Path(root).resolve(strict=True);plan=json.loads((root/'screen_plan.json').read_text());verify(plan);common=support(root)
    write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_retry=True));(root/'fits').mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='');completed=[]
    try:
        for j in plan['jobs']:
            verify(plan);fit=root/'fits'/j['name'];fit.mkdir();print('START '+j['name'],flush=True)
            rec=bounded_fit([plan['binary'],j['batch'],str(fit/'bundles')],fit,env,plan['limits']);bundle=fit/'bundles'/j['name']
            with (bundle/'training_history.csv').open() as f:history=list(csv.DictReader(f))
            if len(history)!=50 or [int(r['epoch']) for r in history]!=list(range(1,51)):raise ValueError('Exactly50 epochs required')
            losses=[float(r['validation_loss']) for r in history]
            if not all(math.isfinite(v) for v in losses+[float(r['training_loss']) for r in history]):raise ValueError('Nonfinite losses')
            if [i for i,r in enumerate(history) if r['selected_checkpoint']=='1']!=[min(range(50),key=lambda i:losses[i])]:raise ValueError('Checkpoint selection mismatch')
            with (fit/'fresh_reload.log').open('x') as log:
                subprocess.run([plan['verifier'],str(bundle),str(fit/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
            for path in [bundle/'predictions.csv',fit/'validation_predictions.csv']:
                rows=load_predictions(path,j['family'],'validation')
                with path.open() as f:all_rows=list(csv.DictReader(f))
                if len(rows)!=len(all_rows) or set(rows)!=set(common['validation']) or any(rows[x][0]!=common['validation'][x] for x in rows):raise ValueError('Test/extra/mismatched prediction rows')
            if directory_bytes(fit)>LIMITS['artifact_bytes_per_fit']:raise ValueError('Artifact ceiling exceeded')
            rec['sha256']={str(p.relative_to(fit)):digest(p) for p in fit.rglob('*') if p.is_file()};write_new(fit/'completion.json',rec)
            completed.append(j['name']);print('VERIFIED '+j['name'],flush=True)
        decisions=analyze(root,plan)
        write_new(root/'screen_completion.json',dict(completed=completed,plan_sha256=digest(root/'screen_plan.json'),decisions=decisions,no_automatic_expansion=True))
    except BaseException as exc:
        write_new(root/'screen_failure.json',dict(completed=completed,error=str(exc),no_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--prepare',nargs=5,type=Path,metavar=('MANIFEST','SUPPORT','BINARY','VERIFIER','PREFLIGHT'))
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if bool(a.prepare)==a.launch:p.error('Choose prepare or launch')
    if a.prepare:prepare(a.root,*a.prepare)
    else:launch(a.root)
