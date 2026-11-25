#!/usr/bin/env python3
"""Read-only attribution of a CLOSED fixed-epoch campaign; never fits a model."""
import argparse,csv,hashlib,json,math
from datetime import datetime,timedelta,timezone
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(path,truth):
 out={}
 with Path(path).open() as f:
  for z in csv.DictReader(f):
   x=float(z['x'])
   if not math.isfinite(x) or not x.is_integer() or not 17543<x<=26303 or z['split']!='validation' or int(x) in out:raise ValueError('Prediction identity before values')
   x=int(x)
   if x not in truth:raise ValueError('Unexpected support')
   y,p=float(z['observed']),float(z['predicted'])
   if y!=truth[x] or not math.isfinite(p) or p<0:raise ValueError('Prediction values')
   out[x]=p
 if set(out)!=set(truth):raise ValueError('Incomplete support')
 return out

def event_windows(truth,threshold,pad=6):
 # Observed threshold crossings determine windows before comparing model errors.
 peaks=[x for x in sorted(truth) if truth[x]>threshold];runs=[]
 for x in peaks:
  a,b=x-pad,x+pad
  if runs and a<=runs[-1][1]+1:runs[-1][1]=b
  else:runs.append([a,b])
 accepted=[];excluded=[]
 for a,b in runs:
  (accepted if all(x in truth for x in range(a,b+1)) else excluded).append([a,b])
 return accepted,excluded

def partitions(truth,old,new,q90):
 total=math.fsum((new[x]-truth[x])**2-(old[x]-truth[x])**2 for x in truth)
 high=math.fsum((new[x]-truth[x])**2-(old[x]-truth[x])**2 for x in truth if truth[x]>q90)
 low=math.fsum((new[x]-truth[x])**2-(old[x]-truth[x])**2 for x in truth if truth[x]<=q90)
 if not math.isclose(high+low,total,rel_tol=1e-11,abs_tol=1e-12):raise ValueError('SSE partition')
 return dict(delta_sse=total,high_flow_delta_sse=high,remaining_delta_sse=low,high_flow_fraction_of_delta=high/total if total else None)

def diagnose(root):
 root=Path(root);plan=json.loads((root/'screen_plan.json').read_text());closed=json.loads((root/'independent_final_check.json').read_text())
 if closed['status']!='COMPLETE_INDEPENDENTLY_VERIFIED_NOT_ADVANCED' or closed['plan_sha256']!=sha(root/'screen_plan.json'):raise ValueError('Independently closed campaign required')
 for p,h in plan['pins'].items():
  if sha(p)!=h:raise ValueError('Pinned source changed '+p)
 train=[];truth={};training_years={2015:[],2016:[]}
 with (root/'support.csv').open() as f:
  for z in csv.DictReader(f):
   x=int(z['target_x']);s=z['split']
   if s not in ['train','validation'] or not (0<x<=17543 if s=='train' else 17543<x<=26303):raise ValueError('Support identity')
   y=float(z['observed'])
   if not math.isfinite(y) or y<0:raise ValueError('Support values')
   if s=='train':train.append(y);training_years[2015 if x<=8759 else 2016].append(y)
   elif x in truth:raise ValueError('Duplicate support')
   else:truth[x]=y
 if len(train)!=13072 or len(truth)!=7112:raise ValueError('Support counts')
 train.sort();pos=(len(train)-1)*.9;i=int(pos);q90=train[i]+(train[i+1]-train[i])*(pos-i)
 events,excluded=event_windows(truth,q90);models={};pins={}
 for j in plan['jobs']:
  fit=root/'fits'/j['name'];receipt=fit/'completion.json';rec=json.loads(receipt.read_text())
  for p,h in rec['sha256'].items():
   if sha(fit/p)!=h:raise ValueError('Completed artifact changed')
  prior=Path(plan['pilot'])/'fits'/(j['family']+'_rate_hold_v1_s42');prior_rec=json.loads((prior/'completion.json').read_text())
  for p,h in prior_rec['sha256'].items():
   if sha(prior/p)!=h:raise ValueError('Archived artifact changed')
  bundles=[fit/'bundles'/j['name'],prior/'bundles'/(j['family']+'_rate_hold_v1_s42')]
  configs=[json.loads((b/'experiment_config.json').read_text()) for b in bundles]
  histories=[]
  for b in bundles:
   with (b/'training_history.csv').open() as f:histories.append(list(csv.DictReader(f)))
  new=rows(fit/'validation_predictions.csv',truth);old=rows(prior/'validation_predictions.csv',truth);n,o=histories
  same_width=configs[0]['hidden_layers']==configs[1]['hidden_layers'];prefix=len(n)<=len(o) and all(a[k]==b[k] for a,b in zip(n,o) for k in ['epoch','training_loss','validation_loss'])
  changes={k:[configs[1].get(k),configs[0].get(k)] for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
  summaries=[]
  for a,b in events:
   xs=list(range(a,b+1));peak=max(xs,key=lambda x:truth[x]);oldpeak=max(xs,key=lambda x:old[x]);newpeak=max(xs,key=lambda x:new[x])
   summaries.append(dict(start_x=a,end_x=b,observed_peak_x=peak,observed_peak_utc=(datetime(2015,1,1,tzinfo=timezone.utc)+timedelta(hours=peak)).isoformat(),observed_peak=truth[peak],old_peak_lag=oldpeak-peak,new_peak_lag=newpeak-peak,old_at_observed_peak=old[peak],new_at_observed_peak=new[peak],delta_sse=math.fsum((new[x]-truth[x])**2-(old[x]-truth[x])**2 for x in xs)))
  monthly={}
  for x in truth:
   m=(datetime(2015,1,1,tzinfo=timezone.utc)+timedelta(hours=x)).strftime('%Y-%m')
   monthly[m]=monthly.get(m,0.)+(new[x]-truth[x])**2-(old[x]-truth[x])**2
  attribution=partitions(truth,old,new,q90)
  if not math.isclose(math.fsum(monthly.values()),attribution['delta_sse'],rel_tol=1e-11,abs_tol=1e-12):raise ValueError('Monthly partition')
  event_sum=math.fsum(v['delta_sse'] for v in summaries)
  models[j['family']]=dict(new_epoch=int(next(h['epoch'] for h in n if h['selected_checkpoint']=='1')),old_epoch=int(next(h['epoch'] for h in o if h['selected_checkpoint']=='1')),same_width=same_width,training_and_validation_history_prefix_exact=prefix,changed_config_fields=changes,attribution=attribution,monthly_delta_sse=monthly,event_delta_sse=event_sum,non_event_delta_sse=attribution['delta_sse']-event_sum,largest_worsened_events=sorted(summaries,key=lambda z:z['delta_sse'],reverse=True)[:10],largest_improved_events=sorted(summaries,key=lambda z:z['delta_sse'])[:5],interpretation='Identical history prefix supports checkpoint-duration attribution; no intermediate prediction snapshots available.' if same_width and prefix else 'Width and saved-checkpoint changes are confounded; cannot isolate causal width effect from these outputs.')
  for p in [receipt,prior/'completion.json',*[(b/'training_history.csv') for b in bundles],*[(b/'experiment_config.json') for b in bundles]]:pins[str(p)]=sha(p)
 return dict(status='READ_ONLY_TRANSFER_DIAGNOSIS',models=models,training_q90=q90,training_year_support={str(k):dict(n=len(v),maximum=max(v),count_above_validation_maximum=sum(y>=max(truth.values()) for y in v)) for k,v in training_years.items()},accepted_event_windows=len(events),excluded_incomplete_event_windows=len(excluded),event_window_rule='Observed TRAIN-Q90 exceedances padded6hours, merged before scoring; any missing hour excludes complete window',pins=pins,new_fits=0,test_rows=0,limitations=['Reused validation; diagnostic not prospective model selection.','Peak timing is measured within bounded observed-event windows, not operational storm detection.','No full checkpoint trajectory, no claim that a post-hoc epoch fixes generalization.'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('new_output',type=Path);a=p.parse_args();r=diagnose(a.root)
 with a.new_output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:dict(epoch=v['new_epoch'],old_epoch=v['old_epoch'],prefix=v['training_and_validation_history_prefix_exact'],attribution=v['attribution']) for k,v in r['models'].items()},indent=2))
