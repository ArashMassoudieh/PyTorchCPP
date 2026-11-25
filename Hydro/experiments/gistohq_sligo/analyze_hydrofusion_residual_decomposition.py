#!/usr/bin/env python3
"""Audit saved physical-reference / learned-correction identities, no fitting."""
import argparse,csv,json,hashlib,math
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):
 out={}
 with Path(path).open() as f:
  for z in csv.DictReader(f):
   x=float(z['target_x'])
   if not math.isfinite(x) or not x.is_integer() or not 17543<x<=26303 or int(x) in out:raise ValueError('Decomposition identity before values')
   v={k:float(z[k]) for k in ['observed','reference','predicted','learned_log_delta','required_log_delta','physical_correction','required_physical_correction']}
   if any(not math.isfinite(a) for a in v.values()) or min(v[k] for k in ['observed','reference','predicted'])<0:raise ValueError('Nonfinite/domain decomposition')
   if z['clipped'] not in ['0','1'] or not math.isclose(v['physical_correction'],v['predicted']-v['reference'],abs_tol=1e-12) or not math.isclose(v['required_physical_correction'],v['observed']-v['reference'],abs_tol=1e-12):raise ValueError('Correction arithmetic')
   out[int(x)]=dict(v,clipped=int(z['clipped']))
 return out
def sums(rows):
 if not rows:raise ValueError('Empty decomposition stratum')
 pe=[r['reference']-r['observed'] for r in rows];cor=[r['physical_correction'] for r in rows];err=[r['predicted']-r['observed'] for r in rows]
 a=math.fsum(v*v for v in pe);b=math.fsum(v*v for v in cor);cross=2*math.fsum(x*y for x,y in zip(pe,cor));total=math.fsum(v*v for v in err)
 if not math.isclose(a+b+cross,total,rel_tol=1e-11,abs_tol=1e-12):raise ValueError('Physical/correction SSE identity')
 return dict(n=len(rows),observed_sum=math.fsum(r['observed'] for r in rows),physical_reference_sse=a,correction_squared_sum=b,twice_error_correction_cross_term=cross,total_sse=total,correction_net_sse_reduction=a-total,reference_signed_error_sum=math.fsum(pe),prediction_signed_error_sum=math.fsum(err),clipped_rows=sum(r['clipped'] for r in rows),wrong_correction_direction_rows=sum(r['physical_correction']*r['required_physical_correction']<0 for r in rows),mean_log_residual_error=math.fsum(r['learned_log_delta']-r['required_log_delta'] for r in rows)/len(rows))
def analyze(root,refits):
 root=Path(root);refits=Path(refits);p=json.loads((root/'plan.json').read_text());c=json.loads((root/'completion.json').read_text())
 if c['status']!='COMPLETE_READ_ONLY_DECOMPOSITION' or not c['no_fits'] or sha(p['binary'])!=p['binary_sha256']:raise ValueError('Verified decomposition completion required')
 for z in c['outputs']:
  path=next(x['output'] for x in p['commands'] if x['name']==z['name'])
  if sha(path)!=z['output_sha256']:raise ValueError('Decomposition output changed')
 frozen=json.loads((refits/'screen_plan.json').read_text())
 if sha(refits/'support.csv')!=frozen['pins'][str(refits/'support.csv')]:raise ValueError('Support hash changed')
 diagnosis=json.loads((refits/'transfer_diagnosis_v2.json').read_text());q90=diagnosis['training_q90'];truth={}
 with (refits/'support.csv').open() as f:
  for z in csv.DictReader(f):
   x=int(z['target_x']);split=z['split']
   if split not in ['train','validation'] or not (0<x<=17543 if split=='train' else 17543<x<=26303):raise ValueError('Support identity')
   if split=='validation':truth[x]=float(z['observed'])
 models={};allrows={}
 for job in p['commands']:
  data=read(job['output']);name=job['name'];allrows[name]=data
  if set(data)!=set(truth) or len(data)!=7112 or any(v['observed']!=truth[x] for x,v in data.items()):raise ValueError('Exact matched decomposition support')
  rows=list(data.values());full=sums(rows);high=sums([v for v in rows if v['observed']>q90]);low=sums([v for v in rows if v['observed']<=q90])
  if not math.isclose(high['total_sse']+low['total_sse'],full['total_sse'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Tail partition')
  mean=math.fsum(truth.values())/len(truth);sst=math.fsum((v-mean)**2 for v in truth.values());models[name]=dict(full=full,high=high,remaining=low,reference_nse=1-full['physical_reference_sse']/sst,prediction_nse=1-full['total_sse']/sst,dominant_peak=data[22565],output_sha256=sha(job['output']))
 for family in ['ffn_physical_forecast','lstm_physical_forecast']:
  a,b=allrows[family+'_fixed'],allrows[family+'_archived']
  if any(a[x]['reference']!=b[x]['reference'] or a[x]['required_log_delta']!=b[x]['required_log_delta'] for x in a):raise ValueError('Physical reference/target changed across checkpoint comparison')
 return dict(status='VERIFIED_READ_ONLY_RESIDUAL_DECOMPOSITION',models=models,matched_rows=7112,training_q90=q90,physical_references_identical=True,new_fits=0,test_rows=0,limitations=['Physical correction is discharge-space predicted minus reference, not raw network output.','Network is a log-space residual; required residual is diagnostic truth, never an input.','SSE cross term couples reference errors and learned corrections; terms are not independent causal contributions.','Reused2017validation; no storm-specific calibration or prospective skill claim.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('refits',type=Path);p.add_argument('new_output',type=Path);a=p.parse_args();r=analyze(a.root,a.refits)
 with a.new_output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
 print(r['status'])
