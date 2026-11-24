#!/usr/bin/env python3
"""Separate finite final-TRAIN refit: four neural fits + one physical prototype.
Choices/epochs come ONLY from the completed2015/2016inner screen. No grid,
retry, best-2017-epoch selection, test scoring, or automatic follow-on stage.
"""
import argparse,csv,json,math,os,shutil,subprocess,time,hashlib
from pathlib import Path
import numpy as np
from run_hydrofusion_six_fit_screen import bounded_fit,digest,write_new,directory_bytes
from run_hydrofusion_preissue_screen import LIMITS,FAMILIES
from run_discharge_state_screen import predict,fit_rain_gain
from run_hydrofusion_joint_inner_screen import metric,gates
HERE=Path(__file__).resolve().parent
NEURAL=FAMILIES[:-1]

def common_support(path):
 out={s:{} for s in ('train','validation')}
 with Path(path).open() as f:
  for r in csv.DictReader(f):
   s=r['split'];x=int(r['target_x'])
   if s not in out or not (0<x<=17543 if s=='train' else 17543<x<=26303) or x in out[s]:raise ValueError('Invalid final support identity before value access')
   v=(float(r['observed']),float(r['persistence']))
   if not all(math.isfinite(a) and a>=0 for a in v):raise ValueError('Invalid final support value')
   out[s][x]=v
 if list(map(len,out.values()))!=[13072,7112]:raise ValueError('Unexpected final support')
 return out

def validate(plan):
 if len(plan['jobs'])!=4 or {j['family'] for j in plan['jobs']}!=set(NEURAL) or plan['limits']!=LIMITS:raise ValueError('Final finite inventory changed')
 for name,h in plan['pins'].items():
  if digest(name)!=h:raise ValueError('Pinned final input/source changed '+name)
 decisions=json.loads(Path(plan['inner_completion']).read_text())['decisions']
 for j in plan['jobs']:
  c=json.loads(Path(j['config']).read_text());d=decisions[j['family']]
  expected=dict(hidden_layers=str(d['width']),epochs=d['selected_epoch'],hydrofusion_checkpoint_selection='fixed_last_epoch_v1',train_end_x=17543,validation_end_x=26303,test_end_x=35063,random_seed=42,hydrofusion_development_only=True,hydrofusion_objective='log_flow_mse_v2',hydrofusion_preissue_rain_recipe='rate_hold_v1')
  if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Inner-chosen final config changed')
  for key,name in [('hydrofusion_native_manifest','manifest'),('hydrofusion_preissue_rain','weather')]:
   if c[key+'_path']!=plan[name] or c[key+'_sha256']!=digest(plan[name]):raise ValueError('Final input binding changed')

def prepare(root,inner,pilot,binary,verifier,preflight,state_features,state_binary):
 root=Path(root).absolute();root.mkdir(exist_ok=False);inner=Path(inner);pilot=Path(pilot)
 review=json.loads((inner/'independent_final_check.json').read_text());completion=json.loads((inner/'screen_completion.json').read_text());accept=json.loads(Path(preflight).read_text())
 if review['status']!='COMPLETE_INDEPENDENTLY_VERIFIED_INTERNAL_ONLY' or review['completed_jobs']!=8:raise ValueError('Closed independent internal screen required')
 if accept['status']!='VERIFIED' or accept['binary_sha256']!=digest(binary) or accept['native_verifier_sha256']!=digest(verifier):raise ValueError('Accepted fixed-epoch preflight required')
 for name in ('executables','configs','provenance'):(root/name).mkdir()
 shutil.copytree(pilot/'prepared_input',root/'prepared_input');shutil.copytree(pilot/'weather',root/'weather');shutil.copy2(pilot/'support.csv',root/'support.csv')
 manifest=root/'prepared_input/native_input_manifest.json';weather=root/'weather/preissue_rain.csv';common=common_support(root/'support.csv')
 for src,name in [(binary,'HydroBatch'),(verifier,'native_reload'),(state_binary,'physical_observer')]:shutil.copy2(src,root/'executables'/name)
 for src,name in [(preflight,'fixed_epoch_acceptance.json'),(inner/'screen_completion.json','inner_completion.json'),(inner/'independent_final_check.json','inner_independent_check.json')]:shutil.copy2(src,root/'provenance'/name)
 for name in ['run_hydrofusion_joint_fixed_epoch_refits.py','run_hydrofusion_six_fit_screen.py','run_hydrofusion_preissue_screen.py','run_discharge_state_screen.py','run_preissue_physical_calibration.py','run_hydrofusion_joint_inner_screen.py']:shutil.copy2(HERE/name,root/'provenance'/name)
 dispatch=HERE.parents[2]/'build-hydrobatch'/('joint_fixed_refits_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False);jobs=[]
 for family in NEURAL:
  decision=completion['decisions'][family];name='fixed_'+family+'_s42';c=json.loads((pilot/'configs'/(family+'_rate_hold_v1_s42.json')).read_text());c.update(experiment_id=name,hidden_layers=str(decision['width']),epochs=decision['selected_epoch'],hydrofusion_checkpoint_selection='fixed_last_epoch_v1',hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=digest(manifest),hydrofusion_preissue_rain_path=str(weather),hydrofusion_preissue_rain_sha256=digest(weather),hydrofusion_preissue_rain_provenance_sha256=digest(weather.parent/'preissue_rain_provenance.json'))
  cp=dispatch/(name+'.json');bp=dispatch/(name+'.batch');write_new(cp,c);bp.write_text(f'{family} {cp.name}\n');shutil.copy2(cp,root/'configs'/cp.name);shutil.copy2(bp,root/'configs'/bp.name);jobs.append(dict(name=name,family=family,config=str(cp),batch=str(bp)))
 # Refit only gain on all formerTRAIN, after shape selection is already frozen
 # from internal2016. No outer validation label is used in this fitting step.
 shape=completion['physical']['selected']
 if shape is None:raise ValueError('No internally accepted physical prototype')
 values={}
 with Path(state_features).open() as f:
  for r in csv.DictReader(f):
   x=int(r['target_x'])
   if not 0<x<=17543 or x in values:raise ValueError('Physical training source scope')
   values[x]=[float(r[k]) for k in ['observed','origin_q','available_effective_rain']]
 ordered=sorted(common['train']);data=np.array([values[x] for x in ordered])
 if set(values)!=set(ordered) or any(tuple(values[x][:2])!=common['train'][x] for x in ordered):raise ValueError('Physical training support mismatch')
 y,q,r=data.T;_,base,response=predict(q,r,shape['params'],0.);gain=fit_rain_gain(y,base,response)
 if gain is None:raise ValueError('Physical refit gain infeasible; no final validation launch')
 write_new(root/'physical_model.json',dict(recipe='observed_discharge_initialized_physical_research_v1',params=shape['params'],gain=gain,training_rows=len(y),training_features_sha256=digest(state_features),selected_shape_source_sha256=digest(inner/'screen_completion.json'),limitations='Not original free-runningPINN; analysis initial water, assumedzeroQlatency; no productiondispatch'))
 pins=[p for folder in ('prepared_input','weather','executables','configs','provenance') for p in (root/folder).rglob('*') if p.is_file()]+[root/'physical_model.json',root/'support.csv',Path(state_features)]+[Path(j[k]) for j in jobs for k in ('config','batch')]
 pins += [HERE/n for n in ['run_hydrofusion_joint_fixed_epoch_refits.py','run_hydrofusion_six_fit_screen.py','run_hydrofusion_preissue_screen.py','run_discharge_state_screen.py','run_preissue_physical_calibration.py','run_hydrofusion_joint_inner_screen.py']]
 plan=dict(schema='HydroFusionJointFixedEpochFinalRefit-v1',approval='Latest human Goahead joint recovery, standing autonomous research authorization; new distinct fixed-last checkpoint protocol',jobs=jobs,limits=LIMITS,inner_completion=str(root/'provenance/inner_completion.json'),manifest=str(manifest),weather=str(weather),binary=str(root/'executables/HydroBatch'),verifier=str(root/'executables/native_reload'),physical_binary=str(root/'executables/physical_observer'),physical_model_sha256=digest(root/'physical_model.json'),pilot=str(pilot),support_counts={k:len(v) for k,v in common.items()},neural_fits=4,physical_replays=1,no_retries=True,no_automatic_expansion=True,
  gates=dict(nse_gain=.02,r_squared_gain=.01,mae_max_ratio=1.02,high_flow_max_ratio=1.02,absolute_bias_increase=2),goals=dict(neural_acceptable=.7,hybrid_stretch=.8,physical=.6),selection='Widths/epochs/physicalshape frozen internally; final TRAIN gain-only refit; outer2017diagnostics neverselectweights or retryconfigurations',
  limitations=['Reused2017validation, not blind generalization or10years.','No test/reservedobservations; no manuscript promotion.','Compare each neural candidate with archived v5seed42on identicalsupport; protocol differs from old2017checkpointselection.','Physical observer remains a different research method; compare persistence and retain original dryphysicalreference.'],pins={str(p):digest(p) for p in pins})
 write_new(root/'screen_plan.json',plan);validate(plan);print('FROZEN4NEWfixed-epoch neuralrefits+1physicalreplay; NOTLAUNCHED')

def read(path,family,expected):
 values={}
 with Path(path).open() as f:
  for r in csv.DictReader(f):
   x=float(r['x'])
   if not math.isfinite(x) or not x.is_integer() or not 17543<x<=26303 or r['approach']!=family or r['split']!='validation' or int(x) in values:raise ValueError('Final prediction identity')
   y,p=float(r['observed']),float(r['predicted'])
   if not math.isfinite(y) or not math.isfinite(p) or min(y,p)<0 or int(x) not in expected or y!=expected[int(x)][0]:raise ValueError('Final prediction/value mismatch')
   values[int(x)]=(y,p)
 if set(values)!=set(expected):raise ValueError('Final prediction support mismatch')
 return values

def launch(root):
 root=Path(root).resolve(strict=True);plan=json.loads((root/'screen_plan.json').read_text());validate(plan);common=common_support(root/'support.csv')
 write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_retry=True));(root/'fits').mkdir(exist_ok=False)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',MALLOC_ARENA_MAX='2');done=[];models={};q90=float(np.quantile([v[0] for v in common['train'].values()],.9))
 try:
  for job in plan['jobs']:
   validate(plan);fit=root/'fits'/job['name'];fit.mkdir();print('START '+job['name'],flush=True);rec=bounded_fit([plan['binary'],job['batch'],str(fit/'bundles')],fit,env,LIMITS);bundle=fit/'bundles'/job['name'];c=json.loads((bundle/'experiment_config.json').read_text())
   with (bundle/'training_history.csv').open() as f:history=list(csv.DictReader(f))
   if c['hydrofusion_checkpoint_selection']!='fixed_last_epoch_v1' or len(history)!=c['epochs'] or [int(r['epoch']) for r in history if r['selected_checkpoint']=='1']!=[c['epochs']]:raise ValueError('Final fixed epoch not exported')
   if not all(math.isfinite(float(r[k])) for r in history for k in ('training_loss','validation_loss')):raise ValueError('Nonfinite final losses')
   with (fit/'fresh_reload.log').open('x') as log:subprocess.run([plan['verifier'],str(bundle),str(fit/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=3600,check=True)
   rows=read(fit/'validation_predictions.csv',job['family'],common['validation']);ordered=sorted(rows);y=np.array([rows[x][0] for x in ordered]);p=np.array([rows[x][1] for x in ordered]);candidate=metric(y,p,q90)
   old=Path(plan['pilot'])/'fits'/(job['family']+'_rate_hold_v1_s42')/'validation_predictions.csv';prior=read(old,job['family'],common['validation']);control=metric(y,np.array([prior[x][1] for x in ordered]),q90);g=gates(control,candidate)
   if directory_bytes(fit)>LIMITS['artifact_bytes_per_fit']:raise ValueError('Final artifact cap')
   rec.update(validation_metrics=candidate,archived_v5_control=control,archived_control_sha256=digest(old),checks=g,advanced=all(g.values()),sha256={str(p.relative_to(fit)):digest(p) for p in fit.rglob('*') if p.is_file()});write_new(fit/'completion.json',rec);models[job['family']]=dict(candidate=candidate,control=control,checks=g,advanced=all(g.values()));done.append(job['name']);print('VERIFIED '+job['name']+' NSE='+str(candidate['nse']),flush=True)
  model=json.loads((root/'physical_model.json').read_text());output=root/'physical_validation.csv'
  with (root/'physical_validation.log').open('x') as log:subprocess.run([plan['physical_binary'],plan['manifest'],digest(plan['manifest']),plan['weather'],str(root/'support.csv'),'validation',str(output),*map(str,model['params']),str(model['gain'])],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=300,check=True)
  with output.open() as f:physical_rows=list(csv.DictReader(f))
  ordered=sorted(common['validation']);x=[int(r['target_x']) for r in physical_rows]
  if x!=ordered or any(float(r['observed'])!=common['validation'][xx][0] for xx,r in zip(x,physical_rows)):raise ValueError('Physical final support mismatch')
  y=np.array([common['validation'][xx][0] for xx in ordered]);p=np.array([float(r['predicted']) for r in physical_rows]);persistence=np.array([common['validation'][xx][1] for xx in ordered]);m=metric(y,p,q90);b=metric(y,persistence,q90);g=gates(b,m)
  write_new(root/'screen_completion.json',dict(status='COMPLETE_FIXED_EPOCH_REUSED_VALIDATION',completed=done,models=models,physical=dict(method=model['recipe'],candidate=m,persistence=b,checks=g,advanced=all(g.values()),predictions_sha256=digest(output)),plan_sha256=digest(root/'screen_plan.json'),test_rows=0,no_retries=True,no_automatic_expansion=True))
 except BaseException as e:write_new(root/'screen_failure.json',dict(completed=done,error=str(e),no_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',nargs=7,type=Path);p.add_argument('--launch',action='store_true');a=p.parse_args()
 if bool(a.prepare)==a.launch:p.error('Prepare OR launch')
 if a.prepare:prepare(a.root,*a.prepare)
 else:launch(a.root)
