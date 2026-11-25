#!/usr/bin/env python3
"""Read-only paired TRAIN-fold error and learning-curve diagnosis; never fits."""
import argparse,csv,json,math
from pathlib import Path
from run_hydrofusion_shuffled_folds import check,predictions
from run_hydrofusion_duration_folds import read_support,FOLDS
from run_hydrofusion_six_fit_screen import digest,write_new

def partition(rows,threshold):
 high=[];low=[]
 for x,(y,p) in rows.items():(high if y>threshold else low).append((p-y)**2)
 h,l=math.fsum(high),math.fsum(low);total=math.fsum((p-y)**2 for y,p in rows.values())
 if not math.isclose(h+l,total,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('SSE partition mismatch')
 return dict(sse=total,high_sse=h,remaining_sse=l,high_sse_fraction=h/total if total else 0,high_rows=len(high),remaining_rows=len(low))

def curve(path):
 with path.open() as f:h=list(csv.DictReader(f))
 if [int(r['epoch']) for r in h]!=list(range(1,51)):raise ValueError('Epoch inventory')
 train=[float(r['training_loss']) for r in h];val=[float(r['validation_loss']) for r in h]
 if any(not math.isfinite(v) or v<0 for v in train+val):raise ValueError('Loss domain')
 i=min(range(50),key=lambda k:val[k])
 if [k for k,r in enumerate(h) if r['selected_checkpoint']=='1']!=[i]:raise ValueError('Selection mismatch')
 return dict(selected_epoch=i+1,selected_validation_loss=val[i],training_loss_at_selected=train[i],final_training_loss=train[-1],final_validation_loss=val[-1],late_optimum=i>=45)

def analyze(root,out):
 p=json.loads((root/'screen_plan.json').read_text());check(p);comparison=json.loads((root/'comparison.json').read_text());results=[];pins={str(root/'comparison.json'):digest(root/'comparison.json'),str(root/'screen_plan.json'):digest(root/'screen_plan.json'),str(Path(__file__).resolve()):digest(__file__)}
 for j in p['jobs']:
  _,a,b=next(f for f in FOLDS if f[0]==j['fold']);s=read_support(j['support'],a,b);record=next(z for z in comparison['pairs'] if z['fold']==j['fold'] and z['family']==j['family']);q=sorted(y for y,_ in s['train'].values());pos=(len(q)-1)*.9;i=int(pos);threshold=q[i]+(q[min(i+1,len(q)-1)]-q[i])*(pos-i)
  if threshold!=record['training_q90']:raise ValueError('TRAIN threshold mismatch')
  new=root/'fits'/j['name'];old=Path(j['control_predictions']).parent
  for fit in (old,new):
   receipt=fit/'completion.json';rec=json.loads(receipt.read_text());pins[str(receipt)]=digest(receipt)
   for rel,h in rec['sha256'].items():
    if digest(fit/rel)!=h:raise ValueError('Artifact changed')
  arms={}
  for name,fit,pred in [('chronological',old,Path(j['control_predictions'])),('shuffled',new,new/'validation_predictions.csv')]:
   rows=predictions(pred,j['family'],s['validation'],a,b);bundles=[v for v in (fit/'bundles').iterdir() if v.is_dir()]
   if len(bundles)!=1:raise ValueError('Ambiguous bundle')
   arms[name]=dict(errors=partition(rows,threshold),curve=curve(bundles[0]/'training_history.csv'))
  delta=arms['shuffled']['errors']['sse']-arms['chronological']['errors']['sse'];dh=arms['shuffled']['errors']['high_sse']-arms['chronological']['errors']['high_sse']
  results.append(dict(fold=j['fold'],family=j['family'],training_q90=threshold,arms=arms,total_sse_change=delta,high_sse_change=dh,remaining_sse_change=delta-dh))
 write_new(out,dict(status='VERIFIED_READ_ONLY_TRAIN_FOLD_DIAGNOSIS',pairs=results,pins=pins,test_rows=0,new_fits=0,limitations=['Training_loss is an online minibatch average, not fixed-checkpoint TRAIN reevaluation.','High-flow partition is descriptive; it does not establish the cause of optimization differences.','Three reused folds and one seed; no generalization or superiority claim.']))
 for r in results:print(r['fold'],r['family'],'SSE delta',round(r['total_sse_change'],6),'high delta',round(r['high_sse_change'],6),'epochs',*[r['arms'][a]['curve']['selected_epoch'] for a in ('chronological','shuffled')])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();analyze(a.root,a.output)
