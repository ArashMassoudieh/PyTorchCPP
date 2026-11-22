#!/usr/bin/env python3
"""Fixed all-five matched rainfall package experiment:24fits+2physical runs.
No retries/expansion; validation-only advancement. Reused development split,
assumed release latency, changed rain source+age: no blind/general skill claim.
"""
import argparse,csv,json,math,os,shutil,statistics,subprocess,time
from pathlib import Path
from run_hydrofusion_six_fit_screen import bounded_fit,digest,write_new,directory_bytes
from evaluate_causal_forecast_methods import load_predictions,metrics
HERE=Path(__file__).resolve().parent
FAMILIES=['ffn_causal_forecast','lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast','pinn_causal_forecast']
ARMS=['aorc24_control','radar1_candidate']
LIMITS=dict(epochs=50,wall_seconds_per_fit=28800,rss_bytes=12*1024**3,threads=2,gpu=False,artifact_bytes_per_fit=250000000)
def support_rows(root):
 with (root/'input_packages/matched_support.csv').open() as f:rows=list(csv.DictReader(f))
 assert [r['split'] for r in rows].count('train')==14369
 return rows

def prepare(root,packages,binary,verifier,acceptance,legacy):
 root.mkdir(parents=True,exist_ok=False);shutil.copytree(packages,root/'input_packages');(root/'configs').mkdir();(root/'executables').mkdir();(root/'provenance').mkdir()
 for source,name in [(binary,'HydroBatch'),(verifier,'family_reload')]:shutil.copy2(source,root/'executables'/name)
 for p in (acceptance,legacy):
  d=json.loads(p.read_text());assert d.get('commands') and all(x['exit_code'] in (0,1) for x in d['commands'])
  shutil.copy2(p,root/'provenance'/p.parent.name.replace(' ','_').__add__('_acceptance.json'))
 for p in [Path(__file__),HERE/'run_hydrofusion_six_fit_screen.py',HERE/'evaluate_causal_forecast_methods.py',HERE/'prepare_hydrofusion_radar_comparison.py']:
  shutil.copy2(p,root/'provenance'/p.name)
 dispatch=HERE.parents[2]/'build-hydrobatch'/('radar_matched_'+digest(packages/'matched_support.csv')[:12]);dispatch.mkdir(exist_ok=False)
 jobs=[]
 for family in FAMILIES:
  for seed in ([42] if family=='pinn_causal_forecast' else [42,43,44]):
   for arm in ARMS:
    name=f'{arm}_{family}_s{seed}';manifest=root/'input_packages'/arm/'native_input_manifest.json'
    c=json.loads((HERE/'hydrofusion_five_method_checks'/(family+'_log_residual.json')).read_text())
    c.update(experiment_id=name,epochs=50,batch_size=64,learning_rate=.001,hidden_layers='32',random_seed=seed,train_end_x=17543,validation_end_x=26303,test_end_x=35063,hydrofusion_synthetic_verification=False,hydrofusion_target_mode='log_residual',hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=digest(manifest),hydrofusion_objective='log_mse_v1' if family=='pinn_causal_forecast' else 'log_flow_mse_v2',hydrofusion_ffn_activation='tanh')
    if 'physical' in family or family=='pinn_causal_forecast':c['physics_profile']='causal_valid_time_dry_v3'
    cp=dispatch/(name+'.json');write_new(cp,c);shutil.copy2(cp,root/'configs'/cp.name)
    bp=dispatch/(name+'.batch');bp.write_text(f'{family} {cp.name}\n');shutil.copy2(bp,root/'configs'/bp.name)
    jobs.append(dict(name=name,family=family,arm=arm,seed=seed,config=str(cp),batch=str(bp),physical_only=family=='pinn_causal_forecast'))
 assert len(jobs)==26
 pinned=[p for folder in ['input_packages','configs','executables','provenance'] for p in (root/folder).rglob('*') if p.is_file()]+[Path(j[k]) for j in jobs for k in ['config','batch']]
 plan=dict(schema='HydroFusionRadarMatchedAllFive-v1',approval='Human: No need to ask about budget, go ahead; all methods; renewed Go ahead after full-period checks',jobs=jobs,limits=LIMITS,binary=str(root/'executables/HydroBatch'),verifier=str(root/'executables/family_reload'),input_sha256={str(p):digest(p) for p in pinned},advancement=dict(positive_validation_seeds=2,mean_validation_delta_nse=.02,maximum_relative_validation_mae_increase=.02),limitations=['Exploratory reused development fold, not blind evaluation.','Rain source AND age differ; cannot isolate timing contribution.','Radar release is assumed1h; no historical operational-delivery certification.','Missing rain excluded symmetrically, no fallback/zero fill.','Physical parameters fixed; no extra calibration or automatic expansion.'])
 write_new(root/'screen_plan.json',plan);support_rows(root);print('FROZEN24fits+2fixedphysical runs, no fits launched by prepare')

def verify(root,p):
 if p['limits']!=LIMITS or len(p['jobs'])!=26:raise ValueError('Budget changed')
 if {(j['family'],j['seed'],j['arm']) for j in p['jobs']}!={(f,s,a) for f in FAMILIES for s in ([42] if f=='pinn_causal_forecast' else [42,43,44]) for a in ARMS}:raise ValueError('Job inventory changed')
 for path,h in p['input_sha256'].items():
  if digest(path)!=h:raise ValueError('Pinned input changed: '+path)

def score(root):
 p=json.loads((root/'screen_plan.json').read_text());verify(root,p);rows=support_rows(root);train=sorted(float(r['observed']) for r in rows if r['split']=='train');pos=(len(train)-1)*.9;lo=int(pos);q90=train[lo]+(train[min(lo+1,len(train)-1)]-train[lo])*(pos-lo)
 report={}
 for split in ['validation','test']:
  rr=[r for r in rows if r['split']==split];support={int(r['target_x']):float(r['observed']) for r in rr};out={}
  def evaluate(vals):
   if set(vals)!=set(support) or any(vals[x][0]!=support[x] for x in support):raise ValueError('Matched support/value mismatch')
   o=[vals[x][0] for x in support];v=[vals[x][1] for x in support];m=metrics(o,v,q90);r=statistics.correlation(o,v) if statistics.pstdev(o)>0 and statistics.pstdev(v)>0 else None;m['r_squared']=r*r if r is not None else None;return m
  out['persistence']=evaluate({int(r['target_x']):(float(r['observed']),float(r['persistence'])) for r in rr})
  for j in p['jobs']:
   fit=root/'fits'/j['name'];path=fit/'validation_predictions.csv' if split=='validation' else fit/'bundles'/j['name']/'predictions.csv'
   out[j['name']]=evaluate(load_predictions(path,j['family'],split))
  report[split]=out
 advancement={}
 for family in FAMILIES:
  seeds=[42] if family=='pinn_causal_forecast' else [42,43,44];deltas=[];a=[];b=[]
  for s in seeds:
   control=report['validation'][f'aorc24_control_{family}_s{s}'];candidate=report['validation'][f'radar1_candidate_{family}_s{s}'];deltas.append(candidate['nse']-control['nse']);a.append(control['mae']);b.append(candidate['mae'])
  advancement[family]=dict(mean_delta_nse=statistics.mean(deltas),positive_seeds=sum(d>0 for d in deltas),relative_mae_change=statistics.mean(b)/statistics.mean(a)-1,passed=(sum(d>0 for d in deltas)>= (1 if len(seeds)==1 else 2) and statistics.mean(deltas)>=.02 and statistics.mean(b)/statistics.mean(a)<=1.02))
 write_new(root/'results_summary.json',dict(splits=report,validation_advancement=advancement,automatic_next_stage=False,limitations=p['limitations']))
 lines=['# Matched rainfall package comparison','','| Family | Arm | Split | Mean NSE | Mean R² |','|---|---|---|---:|---:|']
 for f in FAMILIES:
  for a in ARMS:
   for split in report:
    vals=[report[split][j['name']] for j in p['jobs'] if j['family']==f and j['arm']==a];r2=statistics.mean(v['r_squared'] for v in vals) if all(v['r_squared'] is not None for v in vals) else float('nan');lines.append(f'| {f} | {a} | {split} | {statistics.mean(v["nse"] for v in vals):.6f} | {r2:.6f} |')
 lines+=['',*p['limitations']];(root/'RESULT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

def launch(root):
 p=json.loads((root/'screen_plan.json').read_text());verify(root,p);rows=support_rows(root);support={s:{int(r['target_x']):float(r['observed']) for r in rows if r['split']==s} for s in ['validation','test']};qref=statistics.median(float(r['observed']) for r in rows if r['split']=='train' and float(r['observed'])>0)
 write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_retries=True));(root/'fits').mkdir(exist_ok=False)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='');completed=[]
 try:
  for j in p['jobs']:
   verify(root,p);fit=root/'fits'/j['name'];fit.mkdir();print('START '+j['name'],flush=True)
   rec=bounded_fit([p['binary'],j['batch'],str(fit/'bundles')],fit,env,p['limits']);bundle=fit/'bundles'/j['name']
   if not j['physical_only']:
    with (bundle/'training_history.csv').open() as f:hist=list(csv.DictReader(f))
    if len(hist)!=50 or [int(r['epoch']) for r in hist]!=list(range(1,51)):raise ValueError('Epoch inventory mismatch')
    losses=[float(r['validation_loss']) for r in hist]
    if not all(math.isfinite(v) for v in losses+[float(r['training_loss']) for r in hist]):raise ValueError('Nonfinite loss')
    if [i for i,r in enumerate(hist) if r['selected_checkpoint']=='1']!=[min(range(50),key=lambda i:losses[i])]:raise ValueError('Wrong checkpoint selection')
   with (fit/'fresh_reload.log').open('x') as log:subprocess.run([p['verifier'],str(bundle),str(fit/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
   scale=[r for r in (bundle/'hydrofusion_metadata.txt').read_text().splitlines() if r.startswith('q_ref=')]
   if len(scale)!=1 or float(scale[0].split('=')[1])!=(1. if j['physical_only'] else qref):raise ValueError('Training-only matched q_ref mismatch')
   for s,path in [('test',bundle/'predictions.csv'),('validation',fit/'validation_predictions.csv')]:
    got=load_predictions(path,j['family'],s)
    if set(got)!=set(support[s]) or any(got[x][0]!=support[s][x] for x in got):raise ValueError('Frozen support mismatch')
   if directory_bytes(fit)>LIMITS['artifact_bytes_per_fit']:raise ValueError('Artifact ceiling')
   rec['sha256']={str(f.relative_to(fit)):digest(f) for f in fit.rglob('*') if f.is_file()};write_new(fit/'completion.json',rec);completed.append(j['name']);print('VERIFIED '+j['name'],flush=True)
  write_new(root/'screen_completion.json',dict(completed=completed,plan_sha256=digest(root/'screen_plan.json')));score(root)
 except BaseException as exc:
  write_new(root/'screen_failure.json',dict(completed=completed,error=str(exc),no_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',nargs=5,type=Path,metavar=('PACKAGES','BINARY','VERIFIER','MASK_ACCEPTANCE','LEGACY_ACCEPTANCE'));p.add_argument('--launch',action='store_true');p.add_argument('--analyze',action='store_true');a=p.parse_args()
 if sum([bool(a.prepare),a.launch,a.analyze])!=1:p.error('Select exactly one action')
 if a.prepare:prepare(a.root,*a.prepare)
 elif a.launch:launch(a.root)
 else:score(a.root)
