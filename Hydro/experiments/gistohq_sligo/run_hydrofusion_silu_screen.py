#!/usr/bin/env python3
"""Three NEW FFN fits changing only tanh to SiLU after the closed context pilot."""
import argparse,csv,hashlib,json,math,os,shutil,statistics,subprocess,time
from pathlib import Path
from run_hydrofusion_six_fit_screen import bounded_fit,digest,write_new,directory_bytes
from run_hydrofusion_context_pilot import support,LIMITS
from evaluate_causal_forecast_methods import metrics,load_predictions
HERE=Path(__file__).resolve().parent
RECIPES=('zero25','absolute25','anchored25')
FAMILY='ffn_physical_forecast'

def canonical(c):
 return {k:v for k,v in c.items() if k not in ['experiment_id','hydrofusion_native_manifest_path','hydrofusion_ffn_activation']}

def verify(p):
 if p['limits']!=LIMITS or len(p['jobs'])!=3 or {(j['family'],j['recipe'],j['seed']) for j in p['jobs']}!={(FAMILY,r,42) for r in RECIPES}:raise ValueError('Fixed three-fit budget changed')
 for f,h in p['pins'].items():
  if digest(f)!=h:raise ValueError('Pinned input changed: '+f)
 for j in p['jobs']:
  c=json.loads(Path(j['config']).read_text());a=json.loads(Path(j['reference_config']).read_text())
  if c['hydrofusion_ffn_activation']!='silu' or a.get('hydrofusion_ffn_activation','tanh')!='tanh' or canonical(a)!=canonical(c):raise ValueError('Activation-only isolation failed')
  if c['hydrofusion_native_manifest_path']!=p['manifest'] or c['hydrofusion_native_manifest_sha256']!=digest(p['manifest']):raise ValueError('Native input mismatch')
  if c['hydrofusion_context_recipe']!=j['recipe']:raise ValueError('Recipe/config identity mismatch')
  if c['epochs']!=50 or c['random_seed']!=42 or c['hidden_layers']!='32' or not c['hydrofusion_development_only']:raise ValueError('Frozen training settings changed')

def prepare(root,previous,binary,verifier,acceptance):
 root,previous,binary,verifier,acceptance=map(lambda p:Path(p).absolute(),(root,previous,binary,verifier,acceptance));root.mkdir(parents=True,exist_ok=False)
 old=json.loads((previous/'screen_plan.json').read_text());check=json.loads((previous/'independent_final_check.json').read_text());accept=json.loads(acceptance.read_text())
 if check['plan_sha256']!=digest(previous/'screen_plan.json'):raise ValueError('Previous independent check does not bind plan')
 if check['status']!='COMPLETE_INDEPENDENTLY_VERIFIED_NOT_ADVANCED' or accept['status']!='VERIFIED' or accept['binary_sha256']!=digest(binary) or accept['native_verifier_sha256']!=digest(verifier):raise ValueError('Closed pilot and matching preflight required')
 shutil.copytree(previous/'prepared_input',root/'prepared_input');shutil.copy2(previous/'support.csv',root/'support.csv');support(root)
 (root/'configs').mkdir();(root/'provenance').mkdir();(root/'executables').mkdir()
 for source,name in [(binary,'HydroBatch'),(verifier,'native_reload')]:shutil.copy2(source,root/'executables'/name)
 for source,name in [(previous/'results_summary.json','tanh_results.json'),(previous/'independent_final_check.json','tanh_independent_check.json'),(previous/'screen_plan.json','tanh_plan.json'),(acceptance,'acceptance.json'),(Path(str(binary)+'.build/source_manifest.json'),'build_source_manifest.json'),(Path(str(verifier)+'.link_manifest.json'),'verifier_link_manifest.json')]:shutil.copy2(source,root/'provenance'/name)
 dispatch=HERE.parents[2]/'build-hydrobatch'/('silu_screen_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False);jobs=[];manifest=root/'prepared_input/native_input_manifest.json'
 for recipe in RECIPES:
  oldjob=next(j for j in old['jobs'] if j['family']==FAMILY and j['recipe']==recipe)
  oldconfig=previous/'configs'/Path(oldjob['config']).name;a=json.loads(oldconfig.read_text());reference=root/'provenance'/('tanh_'+recipe+'.json');write_new(reference,a)
  name='silu_'+recipe+'_s42';c=dict(a,experiment_id=name,hydrofusion_ffn_activation='silu',hydrofusion_native_manifest_path=str(manifest))
  cp=dispatch/(name+'.json');write_new(cp,c);shutil.copy2(cp,root/'configs'/cp.name);bp=dispatch/(name+'.batch');bp.write_text(FAMILY+' '+cp.name+'\n');shutil.copy2(bp,root/'configs'/bp.name)
  jobs.append(dict(name=name,recipe=recipe,family=FAMILY,seed=42,config=str(cp),batch=str(bp),reference_config=str(reference)))
 scripts=[Path(__file__),HERE/'run_hydrofusion_six_fit_screen.py',HERE/'run_hydrofusion_context_pilot.py',HERE/'evaluate_causal_forecast_methods.py']
 for script in scripts:shutil.copy2(script,root/'provenance'/script.name)
 paths=[f for folder in ['prepared_input','configs','executables','provenance'] for f in (root/folder).rglob('*') if f.is_file()]+[root/'support.csv']+scripts+[Path(j[k]) for j in jobs for k in ['config','batch']]
 p=dict(schema='HydroFusionSiLUOnlyDevelopmentScreen-v1',approval='Human: Do this work and supervise yourself till finding the solution; no routine budget questions',reason='Measured65–82percent tanh hidden saturation on top10validation flows; activation-only matched test, not proof of causation.',jobs=jobs,limits=LIMITS,manifest=str(manifest),binary=str(root/'executables/HydroBatch'),verifier=str(root/'executables/native_reload'),pins={str(f):digest(f) for f in paths},source_commit=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD'],text=True).strip(),goals=dict(nse=.8,r_squared=.8),no_automatic_expansion=True)
 write_new(root/'screen_plan.json',p);verify(p);print('FROZEN3NEW SiLU-only fits; no launch by prepare')

def analyze(root,p):
 common=support(root);q=sorted(common['train'].values());pos=(len(q)-1)*.9;i=int(pos);q90=q[i]+(q[min(i+1,len(q)-1)]-q[i])*(pos-i);scored={}
 for j in p['jobs']:
  v=load_predictions(root/'fits'/j['name']/'validation_predictions.csv',FAMILY,'validation')
  if set(v)!=set(common['validation']) or any(v[x][0]!=common['validation'][x] for x in v):raise ValueError('Unmatched validation')
  o=[v[x][0] for x in sorted(v)];pred=[v[x][1] for x in sorted(v)];m=metrics(o,pred,q90);m['r_squared']=statistics.correlation(o,pred)**2 if statistics.pstdev(pred)>0 else None;scored[j['recipe']]=m
 a=scored['zero25'];decisions={}
 for recipe in RECIPES[1:]:
  b=scored[recipe];checks=dict(nse=b['nse']-a['nse']>=.02,r2=b['r_squared'] is not None and a['r_squared'] is not None and b['r_squared']-a['r_squared']>=.01,mae=b['mae']<=a['mae']*1.02,high_flow=b['high_flow_rmse']<=a['high_flow_rmse']*1.02,bias=abs(b['pbias_percent'])<=abs(a['pbias_percent'])+2)
  decisions[recipe]=dict(checks=checks,passed=all(checks.values()),goal_met=b['r_squared'] is not None and b['nse']>=p['goals']['nse'] and b['r_squared']>=p['goals']['r_squared'])
 prior=json.loads((root/'provenance/tanh_results.json').read_text())['validation'];deltas={r:{k:(scored[r][k]-prior[FAMILY+'_'+r+'_s42'][k] if scored[r][k] is not None and prior[FAMILY+'_'+r+'_s42'][k] is not None else None) for k in ['nse','r_squared','mae']} for r in RECIPES}
 write_new(root/'results_summary.json',dict(validation=scored,hybrid_decisions=decisions,activation_only_deltas=deltas,test_scores=0,training_q90=q90,no_automatic_expansion=True))
 return decisions

def launch(root):
 root=Path(root).resolve(strict=True);p=json.loads((root/'screen_plan.json').read_text());verify(p);common=support(root)
 write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_retries=True));(root/'fits').mkdir(exist_ok=False)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='');done=[]
 try:
  for j in p['jobs']:
   verify(p);fit=root/'fits'/j['name'];fit.mkdir();print('START '+j['name'],flush=True);rec=bounded_fit([p['binary'],j['batch'],str(fit/'bundles')],fit,env,LIMITS);bundle=fit/'bundles'/j['name']
   with (bundle/'training_history.csv').open() as f:h=list(csv.DictReader(f))
   losses=[float(row['validation_loss']) for row in h]
   if len(h)!=50 or [int(row['epoch']) for row in h]!=list(range(1,51)) or not all(math.isfinite(v) for v in losses+[float(row['training_loss']) for row in h]):raise ValueError('Epoch/loss mismatch')
   if [i for i,row in enumerate(h) if row['selected_checkpoint']=='1']!=[min(range(50),key=lambda i:losses[i])]:raise ValueError('Checkpoint selection mismatch')
   with (fit/'fresh_reload.log').open('x') as log:subprocess.run([p['verifier'],str(bundle),str(fit/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
   for path in [bundle/'predictions.csv',fit/'validation_predictions.csv']:
    v=load_predictions(path,FAMILY,'validation');rows=list(csv.DictReader(path.open()))
    if len(v)!=len(rows) or set(v)!=set(common['validation']) or any(v[x][0]!=common['validation'][x] for x in v):raise ValueError('Wrong support/split')
   if directory_bytes(fit)>LIMITS['artifact_bytes_per_fit']:raise ValueError('Artifact ceiling')
   rec['sha256']={str(f.relative_to(fit)):digest(f) for f in fit.rglob('*') if f.is_file()};write_new(fit/'completion.json',rec);done.append(j['name']);print('VERIFIED '+j['name'],flush=True)
  decisions=analyze(root,p);write_new(root/'screen_completion.json',dict(completed=done,plan_sha256=digest(root/'screen_plan.json'),decisions=decisions,no_automatic_expansion=True))
 except BaseException as e:write_new(root/'screen_failure.json',dict(completed=done,error=str(e),no_retry=True));raise

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',nargs=4,type=Path,metavar=('PREVIOUS','BINARY','VERIFIER','ACCEPTANCE'));p.add_argument('--launch',action='store_true');a=p.parse_args()
 if bool(a.prepare)==a.launch:p.error('Choose prepare or launch')
 if a.prepare:prepare(a.root,*a.prepare)
 else:launch(a.root)
