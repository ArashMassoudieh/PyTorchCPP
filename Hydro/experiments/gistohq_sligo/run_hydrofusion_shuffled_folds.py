#!/usr/bin/env python3
"""Six prospective shuffled TRAIN-only fits against immutable archived controls."""
import argparse,csv,json,math,shutil
from pathlib import Path
import run_hydrofusion_duration_folds as duration
from run_hydrofusion_six_fit_screen import digest,write_new
from verify_hydrofusion_joint_inner_screen import metric,gated
HERE=Path(__file__).resolve().parent
BINARY_SHA='0e91f08910c243936e2c60805e5ac4444a075269e4dd9de51d88b3d4a7080736'
VERIFIER_SHA='17f969c7d103a26130eba56f85f5d9e4343b3ff84d449d5184c7e69b88096348'

def delta(old,new):
 if old.get('shuffle_training') is not False or new.get('shuffle_training') is not True:raise ValueError('Order flag must change false to true')
 if {k:v for k,v in old.items() if k not in ('experiment_id','shuffle_training')}!={k:v for k,v in new.items() if k not in ('experiment_id','shuffle_training')}:raise ValueError('Non-order config change')

def check(p):
 duration.check(p)
 for j in p['jobs']:
  delta(json.loads(Path(j['control_config']).read_text()),json.loads(Path(j['config']).read_text()))
 if p['binary_sha256']!=BINARY_SHA or p['verifier_sha256']!=VERIFIER_SHA:raise ValueError('Unaccepted executables')
 if digest(p['binary'])!=BINARY_SHA or digest(p['verifier'])!=VERIFIER_SHA:raise ValueError('Executable mismatch')

def prepare(root,parent,binary,verifier):
 parent=Path(parent).resolve();old=json.loads((parent/'screen_plan.json').read_text());duration.check(old)
 if json.loads((parent/'independent_final_check.json').read_text())['status']!='COMPLETE_INDEPENDENTLY_VERIFIED_TRAIN_ONLY':raise ValueError('Unverified control campaign')
 if digest(binary)!=BINARY_SHA or digest(verifier)!=VERIFIER_SHA:raise ValueError('Unaccepted executables')
 root.mkdir(exist_ok=False)
 for d in ('executables','configs','provenance'):(root/d).mkdir()
 for src,n in ((binary,'HydroBatch'),(verifier,'native_reload')):shutil.copy2(src,root/'executables'/n)
 dispatch=HERE.parents[2]/'build-hydrobatch'/root.name;dispatch.mkdir(exist_ok=False)
 jobs=[];pins=dict(old['pins'])
 for j in old['jobs']:
  n='shuffled_'+j['name'];c=json.loads(Path(j['config']).read_text());new=dict(c,experiment_id=n,shuffle_training=True);delta(c,new)
  cp=dispatch/(n+'.json');bp=dispatch/(n+'.batch');write_new(cp,new);bp.write_text(j['family']+' '+cp.name+'\n')
  shutil.copy2(cp,root/'configs'/cp.name);shutil.copy2(bp,root/'configs'/bp.name)
  control=parent/'fits'/j['name'];rec=json.loads((control/'completion.json').read_text())
  for rel,h in rec['sha256'].items():
   if digest(control/rel)!=h:raise ValueError('Control receipt changed')
   pins[str(control/rel)]=h
  pins[str(control/'completion.json')]=digest(control/'completion.json')
  jobs.append(dict(j,name=n,config=str(cp),batch=str(bp),control_config=j['config'],control_predictions=str(control/'validation_predictions.csv')))
  for x in (cp,bp):pins[str(x)]=digest(x)
 for n in ('run_hydrofusion_shuffled_folds.py','run_hydrofusion_duration_folds.py','run_hydrofusion_six_fit_screen.py','run_hydrofusion_preissue_screen.py','verify_hydrofusion_joint_inner_screen.py','verify_preissue_physical_calibration.py','test_hydrofusion_shuffled_folds.py'):
  shutil.copy2(HERE/n,root/'provenance'/n);pins[str(HERE/n)]=digest(HERE/n)
 for q in root.rglob('*'):
  if q.is_file():pins[str(q)]=digest(q)
 for q in (parent/'independent_final_check.json',parent/'screen_plan.json'):pins[str(q)]=digest(q)
 p=dict(old,schema='HydroFusionShuffledChronologicalFolds-v1',approval='Human Go ahead after opt-in seeded-order preflight; six new fits only',jobs=jobs,pins=pins,binary=str(root/'executables/HydroBatch'),verifier=str(root/'executables/native_reload'),binary_sha256=BINARY_SHA,verifier_sha256=VERIFIER_SHA,control_campaign=str(parent),selection_rule='Retain50epoch best_validation_v1; no patience tuning. Compare archived chronological vs shuffled on exact fold support.',advancement_rule='Each fold: NSE+0.02, squaredPearsonR2+0.01, MAE<=1.02control, highflowRMSE<=1.02control, absPBIAS<=abscontrol+2points. Family advances only if all five checks pass in at least2of3folds and no fold NSE decreases. No automatic outer refits.',limitations=['One seed42 and width32; overlapping TRAIN sets; reused development folds; no blind or all-method superiority claim.','Only training-window order changes. No2017/test targets or manuscript promotion.'])
 write_new(root/'screen_plan.json',p);check(p);write_new(root/'independent_plan_review.json',dict(status='VERIFIED_EXACT_SIX_ORDER_ONLY_DELTAS',plan_sha256=digest(root/'screen_plan.json'),fits=6,test_rows=0));print('PREPARED',root,flush=True)

def predictions(path,family,support,a,b):
 if not 0<a<b<=17543:raise ValueError('TRAIN-only cutoff context required')
 out={}
 with Path(path).open() as f:
  for r in csv.DictReader(f):
   x=float(r['x'])
   if not math.isfinite(x) or not x.is_integer() or not a<x<=b or r['split']!='validation' or r['approach']!=family or int(x) in out:raise ValueError('Prediction identity before values')
   x=int(x)
   if x not in support:raise ValueError('Unknown target')
   y,v=float(r['observed']),float(r['predicted'])
   if y!=support[x][0] or not math.isfinite(v) or v<0:raise ValueError('Prediction values')
   out[x]=(y,v)
 if set(out)!=set(support):raise ValueError('Unmatched support')
 return out

def analyze(root):
 p=json.loads((root/'screen_plan.json').read_text());check(p);pairs=[]
 for j in p['jobs']:
  _,a,b=next(f for f in duration.FOLDS if f[0]==j['fold']);s=duration.read_support(j['support'],a,b);q=sorted(v[0] for v in s['train'].values());pos=(len(q)-1)*.9;i=int(pos);threshold=q[i]+(q[min(i+1,len(q)-1)]-q[i])*(pos-i)
  fit=root/'fits'/j['name'];rec=json.loads((fit/'completion.json').read_text())
  for rel,h in rec['sha256'].items():
   if digest(fit/rel)!=h:raise ValueError('New receipt changed')
  log=(fit/'reload.log').read_text()
  if 'production_target_free_checks=12' not in log:raise ValueError('Target-free proof missing')
  scores=[]
  for path in (j['control_predictions'],fit/'validation_predictions.csv'):
   rows=predictions(path,j['family'],s['validation'],a,b);xs=sorted(rows);scores.append(metric([rows[x][0] for x in xs],[rows[x][1] for x in xs],threshold))
  g=gated(*scores);pairs.append(dict(fold=j['fold'],family=j['family'],control=scores[0],shuffled=scores[1],checks=g,pass_all=all(g.values()),selected_epoch=rec['selected_epoch'],training_q90=threshold))
 decisions={}
 for family in duration.FAMILIES:
  rows=[z for z in pairs if z['family']==family];decisions[family]=sum(z['pass_all'] for z in rows)>=2 and all(z['shuffled']['nse']>=z['control']['nse'] for z in rows)
 write_new(root/'comparison.json',dict(status='COMPLETE_MATCHED_TRAIN_ONLY_COMPARISON',pairs=pairs,family_advances=decisions,test_rows=0,outer_refits=0,plan_sha256=digest(root/'screen_plan.json')))
 print(json.dumps(decisions),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--prepare',nargs=3,type=Path);ap.add_argument('--launch',action='store_true');ap.add_argument('--analyze',action='store_true');args=ap.parse_args()
 if sum((bool(args.prepare),args.launch,args.analyze))!=1:ap.error('Choose one operation')
 if args.prepare:prepare(args.root,*args.prepare)
 elif args.launch:
  check(json.loads((args.root/'screen_plan.json').read_text()));duration.launch(args.root);analyze(args.root)
 else:analyze(args.root)
