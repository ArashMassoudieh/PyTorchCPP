#!/usr/bin/env python3
"""Independent scalar receipt/score/selection review of the bounded inner screen.
Only completed jobs are read. This never fits or launches a model.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
from verify_preissue_physical_calibration import metrics,check

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gated(a,b):
 return dict(nse=b['nse']>=a['nse']+.02,r_squared=b['r_squared']>=a['r_squared']+.01,mae=b['mae']<=a['mae']*1.02,high_flow=b['high_flow_rmse']<=a['high_flow_rmse']*1.02,bias=abs(b['pbias_percent'])<=abs(a['pbias_percent'])+2)
def metric(y,p,q90):
 m=metrics(y,p);high=[i for i,v in enumerate(y) if v>q90]
 if not high:raise ValueError('No high flows')
 m['high_flow_rmse']=math.sqrt(math.fsum((p[i]-y[i])**2 for i in high)/len(high));return m

def read_predictions(path,family,expected):
 rows={}
 with path.open() as f:
  for r in csv.DictReader(f):
   x=float(r['x'])
   if not math.isfinite(x) or not x.is_integer() or not 8759<x<=17543 or r['split']!='validation' or r['approach']!=family or int(x) in rows:raise ValueError('Prediction identity mismatch')
   y,p=float(r['observed']),float(r['predicted'])
   if not math.isfinite(y) or not math.isfinite(p) or min(y,p)<0 or int(x) not in expected or y!=expected[int(x)][0]:raise ValueError('Prediction value/support mismatch')
   rows[int(x)]=(y,p)
 if set(rows)!=set(expected):raise ValueError('Prediction support mismatch')
 return rows

def verify(root):
 plan=json.loads((root/'screen_plan.json').read_text());claim=json.loads((root/'launch_claim.json').read_text())
 if sha(root/'screen_plan.json')!=claim['plan_sha256'] or len(plan['jobs'])!=8 or not plan['no_retries'] or not plan['no_automatic_expansion']:raise ValueError('Plan/claim changed')
 for name,h in plan['pins'].items():
  if sha(name)!=h:raise ValueError('Pinned input/source changed '+name)
 common={s:{} for s in ['train','validation']}
 with (root/'support.csv').open() as f:
  for row in csv.DictReader(f):
   s=row['split'];x=int(row['target_x'])
   if s not in common or not (0<x<=8759 if s=='train' else 8759<x<=17543) or x in common[s]:raise ValueError('Support identity changed')
   values=(float(row['observed']),float(row['persistence']))
   if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('Invalid support values')
   common[s][x]=values
 if {k:len(v) for k,v in common.items()}!=plan['support_counts']:raise ValueError('Support counts changed')
 q=sorted(v[0] for v in common['train'].values());pos=(len(q)-1)*.9;i=int(pos);q90=q[i]+(q[min(i+1,len(q)-1)]-q[i])*(pos-i)
 models={}
 for job in plan['jobs']:
  fit=root/'fits'/job['name'];receipt=fit/'completion.json'
  if not receipt.exists():continue # Never consume an active output.
  rec=json.loads(receipt.read_text())
  for name,h in rec['sha256'].items():
   if sha(fit/name)!=h:raise ValueError('Completed artifact changed '+name)
  bundle=fit/'bundles'/job['name']
  with (bundle/'training_history.csv').open() as f:history=list(csv.DictReader(f))
  if len(history)!=50 or [int(h['epoch']) for h in history]!=list(range(1,51)):raise ValueError('Epoch inventory changed')
  losses=[float(h['validation_loss']) for h in history]
  if not all(math.isfinite(x) for x in losses):raise ValueError('Invalid history values')
  selected=min(range(50),key=lambda i:losses[i])
  if [i for i,h in enumerate(history) if h['selected_checkpoint']=='1']!=[selected] or rec['selected_epoch']!=selected+1:raise ValueError('Checkpoint epoch mismatch')
  fresh=read_predictions(fit/'validation_predictions.csv',job['family'],common['validation']);original=read_predictions(bundle/'predictions.csv',job['family'],common['validation'])
  if any(fresh[x]!=original[x] for x in fresh):raise ValueError('Fresh/native prediction mismatch')
  log=(fit/'fresh_reload.log').read_text()
  if 'PASS all_validation_rows='+str(len(fresh)) not in log or 'production_target_free_checks=12' not in log:raise ValueError('Semantic reload evidence missing')
  ordered=sorted(fresh);m=metric([fresh[x][0] for x in ordered],[fresh[x][1] for x in ordered],q90);check(rec['validation_metrics'],m)
  models[job['name']]=dict(metrics=m,selected_epoch=selected+1)
 report=dict(status='PARTIAL_INDEPENDENT_REVIEW',completed_jobs=len(models),planned_jobs=8,models=models,support_counts=plan['support_counts'],training_q90=q90,outer_validation_rows=0,test_rows=0)
 if len(models)<8:return report
 complete=json.loads((root/'screen_completion.json').read_text())
 if complete['plan_sha256']!=claim['plan_sha256'] or complete['outer_refits_launched']!=0 or complete['outer_validation_rows']!=0 or complete['test_rows']!=0:raise ValueError('Completion scope mismatch')
 if set(complete['models'])!=set(models):raise ValueError('Completion model inventory mismatch')
 for name,model in models.items():
  check(complete['models'][name]['metrics'],model['metrics'])
  if complete['models'][name]['selected_epoch']!=model['selected_epoch']:raise ValueError('Completion epoch mismatch')
 if not math.isclose(complete['training_q90'],q90,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Threshold mismatch')
 decisions={}
 for family in sorted({j['family'] for j in plan['jobs']}):
  a,b=[models[f'inner_{family}_h{w}_s42'] for w in [32,64]];g=gated(a['metrics'],b['metrics']);width=64 if all(g.values()) else 32
  decisions[family]=dict(width=width,selected_epoch=models[f'inner_{family}_h{width}_s42']['selected_epoch'],width64_checks=g)
 if decisions!=complete['decisions']:raise ValueError('Neural internal selection mismatch')
 # The physical observer is a separate experimental method. Verify gain fits
 # using only internal TRAIN, then the predeclared internal validation choice.
 values={}
 with (root/'physical_train_features.csv').open() as f:
  for r in csv.DictReader(f):
   x=int(r['target_x'])
   if not 0<x<=17543 or x in values:raise ValueError('Physical source scope')
   values[x]=[float(r[k]) for k in ['observed','origin_q','available_effective_rain']]
 def subset(split):
  rows=[values[x] for x in sorted(common[split])]
  if any(tuple(v[:2])!=common[split][x] for x,v in zip(sorted(common[split]),rows)):raise ValueError('Physical source/support mismatch')
  return [v[0] for v in rows],[v[1] for v in rows],[v[2] for v in rows]
 ty,tq,tr=subset('train');vy,vq,vr=subset('validation');baseline=metric(vy,vq,q90);ph=complete['physical'];check(ph['baseline_persistence'],baseline)
 if len(plan['physical_shapes'])!=27 or len(ph['candidates'])!=27:raise ValueError('Physical candidate inventory mismatch')
 for index,(params,candidate) in enumerate(zip(plan['physical_shapes'],ph['candidates'])):
  if index!=candidate['index'] or params!=candidate['params']:raise ValueError('Physical shape drift')
  k,l,f=params;decay=f*math.exp(-k)+(1-f)*math.exp(-l);response=f*(1+math.expm1(-k)/k)+(1-f)*(1+math.expm1(-l)/l)
  base=[v*decay for v in tq];rain=[v*response for v in tr];sy=math.fsum(ty);total=math.fsum(rain)
  lo=max(0.,(.95*sy-math.fsum(base))/total);hi=min(1.,(1.05*sy-math.fsum(base))/total)
  if lo>hi:
   if candidate['eligible'] or 'gain' in candidate:raise ValueError('Infeasible physical gain accepted')
   continue
  gain=max(lo,min(hi,math.fsum(a*(y-b) for a,y,b in zip(rain,ty,base))/math.fsum(a*a for a in rain)))
  if abs(gain-candidate['gain'])>1e-10:raise ValueError('Physical training gain mismatch')
  pred=[q*decay+gain*r*response for q,r in zip(vq,vr)];m=metric(vy,pred,q90);check(candidate['validation_metrics'],m);g=gated(baseline,m)
  if g!=candidate['checks'] or all(g.values())!=candidate['eligible']:raise ValueError('Physical validation gates mismatch')
 eligible=[c for c in ph['candidates'] if c['eligible']];best=max(eligible,key=lambda c:(c['validation_metrics']['nse'],-c['index'])) if eligible else None
 if best!=ph['selected']:raise ValueError('Physical internal choice mismatch')
 report.update(status='COMPLETE_INDEPENDENTLY_VERIFIED_INTERNAL_ONLY',decisions=decisions,physical=ph,outer_refits_launched=0)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('new_output',type=Path);a=p.parse_args();r=verify(a.root)
 with a.new_output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
 print(r['status'],r['completed_jobs'])
