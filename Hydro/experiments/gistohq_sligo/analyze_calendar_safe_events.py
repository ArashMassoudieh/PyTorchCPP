from pathlib import Path
import json,csv,statistics,hashlib
root=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023');out=root/'calendar_safe_regional_analysis'
sources=json.load(open(out/'sources.json'));audit=[];events=[]
for s in sources:
 c=Path(s['control']);p=Path(s['candidate']);a=json.load(open(c.parent/'experiment_config.json'));b=json.load(open(p.parent/'experiment_config.json'))
 differences={k:[a[k],b[k]] for k in a.keys()&b.keys() if a[k]!=b[k] and k not in ['experiment_id','lstm_warm_start_checkpoint_path']};assert not differences,differences
 assert hashlib.sha256((c.parent/'dataset_manifest.json').read_bytes()).hexdigest()==hashlib.sha256((p.parent/'dataset_manifest.json').read_bytes()).hexdigest()
 audit.append(dict(fold=s['fold'],seed=s['seed'],shared_config_differences=differences,dataset_manifest_identical=True,limitation='Archived controls use an earlier executable; missing export fields cannot establish full executable equivalence.'))
 def load(arm):
  with (out/f'fold{s["fold"]}_s{s["seed"]}_{arm}.csv').open() as f:return {float(r['x']):(float(r['observed']),float(r['predicted'])) for r in csv.DictReader(f)}
 ds={k:load(k) for k in ['control','calendar_safe']};xs=sorted(ds['control']);obs=sorted(ds['control'][x][0] for x in xs);threshold=obs[int(.95*(len(obs)-1))]
 high=[x for x in xs if ds['control'][x][0]>=threshold];clusters=[]
 for x in high:
  if not clusters or x-clusters[-1][-1]>24:clusters.append([x])
  else:clusters[-1].append(x)
 for n,cl in enumerate(clusters,1):
  support=[x for x in xs if cl[0]-12<=x<=cl[-1]+24];op=max(support,key=lambda x:ds['control'][x][0]);peak=ds['control'][op][0]
  for arm,d in ds.items():
   pp=max(support,key=lambda x:d[x][1]);events.append(dict(fold=s['fold'],seed=s['seed'],event=n,arm=arm,start_x=min(support),end_x=max(support),n=len(support),sse=sum((d[x][1]-d[x][0])**2 for x in support),peak_error_percent=100*(d[pp][1]-peak)/peak,timing_error_hours=pp-op,volume_error_percent=100*sum(d[x][1]-d[x][0] for x in support)/sum(d[x][0] for x in support)))
with (out/'event_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(events[0]));w.writeheader();w.writerows(events)
(out/'control_provenance_audit.json').write_text(json.dumps(audit,indent=2))
for fold in 'ABC':
 for arm in ['control','calendar_safe']:
  rs=[r for r in events if r['fold']==fold and r['arm']==arm];print(fold,arm,'eventSSE',sum(r['sse'] for r in rs),'medianAbsPeak',statistics.median(abs(r['peak_error_percent']) for r in rs))
