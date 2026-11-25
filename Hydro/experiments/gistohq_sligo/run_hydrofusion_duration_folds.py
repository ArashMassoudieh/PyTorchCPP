#!/usr/bin/env python3
"""Six NEW TRAIN-only chronological diagnostic fits; no outer refit or retry."""
import argparse,csv,json,math,os,shutil,subprocess,time
from pathlib import Path
from prepare_hydrofusion_search_slice import prepare as crop
from run_hydrofusion_six_fit_screen import bounded_fit,digest,write_new,directory_bytes
from run_hydrofusion_preissue_screen import LIMITS
HERE=Path(__file__).resolve().parent
FOLDS=(('2015Q4',6551,8759),('2016H1',8759,13127),('2016H2',13127,17543))
FAMILIES=('ffn_physical_forecast','lstm_physical_forecast')

def stability(histories):
 bands=[];epochs=[]
 for h in histories:
  if len(h)!=50 or any(not math.isfinite(v) or v<0 for v in h):raise ValueError('50 finite nonnegative losses required')
  best=min(range(50),key=lambda i:h[i]);epochs.append(best+1);bands.append({i+1 for i,v in enumerate(h) if v<=h[best]*1.02})
 if len(bands)!=3:raise ValueError('Three chronological folds required')
 common=sorted(set.intersection(*bands));censored=any(e>=46 for e in epochs)
 return dict(selected_epochs=epochs,near_optimal_common_epochs=common,earliest_common_epoch=common[0] if common else None,late_optimum_flag=censored,status='NO_STABLE_EPOCH' if not common else 'PROVISIONAL_LATE_OPTIMUM' if censored else 'COMMON_NEAR_OPTIMAL_EPOCH',rule='Intersection of within2percent minimum validation loss epochs across3folds; no forced median or outer refit')

def read_support(path,train,end):
 out={k:{} for k in ['train','validation']}
 with Path(path).open() as f:
  for z in csv.DictReader(f):
   s=z['split'];x=int(z['target_x'])
   if s not in out or not (0<x<=train if s=='train' else train<x<=end) or end>17543 or x in out[s]:raise ValueError('Fold support identity before values')
   v=tuple(float(z[k]) for k in ['observed','persistence'])
   if any(not math.isfinite(a) or a<0 for a in v):raise ValueError('Fold support values')
   out[s][x]=v
 if min(map(len,out.values()))<100:raise ValueError('Insufficient chronological support')
 return out

def check(p):
 if len(p['jobs'])!=6 or {(j['fold'],j['family']) for j in p['jobs']}!={(f,n) for f,_,_ in FOLDS for n in FAMILIES} or p['limits']!=LIMITS:raise ValueError('Finite fold inventory changed')
 for path,h in p['pins'].items():
  if digest(path)!=h:raise ValueError('Pinned fold source changed '+path)
 for j in p['jobs']:
  c=json.loads(Path(j['config']).read_text());_,a,b=next(f for f in FOLDS if f[0]==j['fold'])
  if any(c[k]!=v for k,v in dict(train_end_x=a,validation_end_x=b,test_end_x=17544,hidden_layers='32',epochs=50,random_seed=42,hydrofusion_checkpoint_selection='best_validation_v1',hydrofusion_development_only=True).items()):raise ValueError('Fold training declaration changed')

def prepare(root,inner,binary,verifier,probe):
 root=Path(root).absolute();root.mkdir(exist_ok=False);inner=Path(inner)
 review=json.loads((inner/'independent_final_check.json').read_text())
 if review['status']!='COMPLETE_INDEPENDENTLY_VERIFIED_INTERNAL_ONLY':raise ValueError('Accepted TRAIN-only parent required')
 for d in ['executables','provenance','configs','folds']:(root/d).mkdir()
 for p,n in [(binary,'HydroBatch'),(verifier,'native_reload'),(probe,'fold_preflight')]:shutil.copy2(p,root/'executables'/n)
 dispatch=HERE.parents[2]/'build-hydrobatch'/('duration_'+digest(root/'executables/HydroBatch')[:8]+'_'+root.name);dispatch.mkdir(exist_ok=False)
 jobs=[];env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
 for name,a,b in FOLDS:
  fold=root/'folds'/name;fold.mkdir();manifest=crop(inner/'prepared_input/native_input_manifest.json',fold/'native',b);weather=fold/'preissue_rain.csv'
  with (inner/'weather/preissue_rain.csv').open() as inp,weather.open('x') as out:
   out.write(next(inp));last=0
   for line in inp:
    x=int(line.partition(',')[0])
    if x!=last+1 or x>17543:raise ValueError('TRAIN weather identity')
    last=x
    if x<=b:out.write(line)
   if last!=17543:raise ValueError('TRAIN weather coverage')
  provenance=fold/'preissue_rain_provenance.json';write_new(provenance,dict(parent_csv_sha256=digest(inner/'weather/preissue_rain.csv'),parent_provenance_sha256=digest(inner/'weather/preissue_rain_provenance.json'),target_cutoff=b,scope='TRAIN-only chronology; no2017'))
  for family in FAMILIES:
   n=f'{name}_{family}_s42';c=json.loads((inner/'configs'/f'inner_{family}_h32_s42.json').read_text());c.update(experiment_id=n,train_end_x=a,validation_end_x=b,test_end_x=17544,epochs=50,hidden_layers='32',hydrofusion_checkpoint_selection='best_validation_v1',hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=digest(manifest),hydrofusion_preissue_rain_path=str(weather),hydrofusion_preissue_rain_sha256=digest(weather),hydrofusion_preissue_rain_provenance_sha256=digest(provenance))
   cp=dispatch/(n+'.json');bp=dispatch/(n+'.batch');write_new(cp,c);bp.write_text(f'{family} {cp.name}\n');shutil.copy2(cp,root/'configs'/cp.name);shutil.copy2(bp,root/'configs'/bp.name)
   if family==FAMILIES[0]:
    with (fold/'preflight.log').open('x') as log:subprocess.run([str(root/'executables/fold_preflight'),str(cp),str(fold/'support.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=300)
   support=read_support(fold/'support.csv',a,b);jobs.append(dict(name=n,fold=name,family=family,config=str(cp),batch=str(bp),support=str(fold/'support.csv'),support_counts={k:len(v) for k,v in support.items()}))
 for n in ['run_hydrofusion_duration_folds.py','prepare_hydrofusion_search_slice.py','run_hydrofusion_six_fit_screen.py','run_hydrofusion_preissue_screen.py']:shutil.copy2(HERE/n,root/'provenance'/n)
 paths=[q for q in root.rglob('*') if q.is_file()]+[Path(j[k]) for j in jobs for k in ['config','batch']]+[HERE/n for n in ['run_hydrofusion_duration_folds.py','prepare_hydrofusion_search_slice.py','run_hydrofusion_six_fit_screen.py','run_hydrofusion_preissue_screen.py']]
 plan=dict(schema='HydroFusionTRAINOnlyDuration-v1',approval='Latest human Go with TRAIN-only chronological duration assessment; standing finite research authorization',jobs=jobs,limits=LIMITS,pins={str(q):digest(q) for q in paths},binary=str(root/'executables/HydroBatch'),verifier=str(root/'executables/native_reload'),neural_fits=6,no_retries=True,no_expansion=True,selection_rule='Three nonoverlapping chronological validation blocks whollywithin2015/2016; report intersection of epochs within2percent each fold minimum loss. No forced consensus. Flag optimum>=46 as late; no automatic extension.',limitations=['One seed and fixed32width; duration diagnosis, not architecture search.','No2017or test targets; not10year study.','Fold training sets overlap; validation blocks do not.','No automatic outer refit, claim of skill or manuscript promotion.'])
 write_new(root/'screen_plan.json',plan);check(plan);print('PREPARED6NEWTRAIN-only duration fits',[(j['name'],j['support_counts']) for j in jobs])

def launch(root):
 root=Path(root).resolve(strict=True);p=json.loads((root/'screen_plan.json').read_text());check(p);write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_retry=True));(root/'fits').mkdir(exist_ok=False)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',MALLOC_ARENA_MAX='2');done=[];histories={k:[] for k in FAMILIES}
 try:
  for j in p['jobs']:
   check(p);fit=root/'fits'/j['name'];fit.mkdir();print('START '+j['name'],flush=True);rec=bounded_fit([p['binary'],j['batch'],str(fit/'bundles')],fit,env,LIMITS);bundle=fit/'bundles'/j['name'];c=json.loads((bundle/'experiment_config.json').read_text());support=read_support(j['support'],c['train_end_x'],c['validation_end_x'])
   with (bundle/'training_history.csv').open() as f:h=list(csv.DictReader(f))
   losses=[float(z['validation_loss']) for z in h]
   if len(h)!=50 or any(not math.isfinite(v) or v<0 for v in losses):raise ValueError('Invalid fold history')
   best=min(range(50),key=lambda i:losses[i])+1
   if [int(z['epoch']) for z in h if z['selected_checkpoint']=='1']!=[best]:raise ValueError('Fold selected epoch mismatch')
   with (fit/'reload.log').open('x') as log:subprocess.run([p['verifier'],str(bundle),str(fit/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
   seen=set()
   with (fit/'validation_predictions.csv').open() as f:
    for z in csv.DictReader(f):
     x=float(z['x'])
     if not math.isfinite(x) or not x.is_integer() or not c['train_end_x']<x<=c['validation_end_x'] or z['split']!='validation' or z['approach']!=j['family'] or int(x) in seen:raise ValueError('Fold prediction identity before values')
     x=int(x);y,v=float(z['observed']),float(z['predicted'])
     if x not in support['validation'] or y!=support['validation'][x][0] or not math.isfinite(v) or v<0:raise ValueError('Fold prediction values')
     seen.add(x)
   if seen!=set(support['validation']) or directory_bytes(fit)>LIMITS['artifact_bytes_per_fit']:raise ValueError('Fold support/artifact cap')
   rec.update(selected_epoch=best,validation_rows=len(seen),sha256={str(q.relative_to(fit)):digest(q) for q in fit.rglob('*') if q.is_file()});write_new(fit/'completion.json',rec);histories[j['family']].append(losses);done.append(j['name']);print('VERIFIED '+j['name']+' epoch='+str(best),flush=True)
  write_new(root/'screen_completion.json',dict(status='COMPLETE_TRAIN_ONLY_DURATION_DIAGNOSTIC',completed=done,stability={k:stability(v) for k,v in histories.items()},plan_sha256=digest(root/'screen_plan.json'),outer_refits=0,test_rows=0))
 except BaseException as e:write_new(root/'screen_failure.json',dict(completed=done,error=str(e),no_retry=True));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--prepare',nargs=4,type=Path);p.add_argument('--launch',action='store_true');a=p.parse_args()
 if bool(a.prepare)==a.launch:p.error('prepare OR launch')
 if a.prepare:prepare(a.root,*a.prepare)
 else:launch(a.root)
