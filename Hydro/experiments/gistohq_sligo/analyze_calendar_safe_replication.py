import sys,json,csv,hashlib
from pathlib import Path
sys.path.insert(0,'Hydro/experiments/gistohq_sligo')
from audit_development_predictions import metrics
root=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023')
out=root/'calendar_safe_regional_analysis';out.mkdir(exist_ok=True)
manifest=json.load(open('Hydro/experiments/gistohq_sligo/improvement_loop_20261023/calendar_safe_regional_configs/run_manifest.json'))
native=set(json.load(open(root.parent/'native_discharge_support_v2.json'))['strict_observed_hour_x'])
results=[];sources=[]
def load(p):
 with p.open() as f:return {float(r['x']):(float(r['observed']),float(r['predicted'])) for r in csv.DictReader(f) if r['split']=='test' and float(r['x']) in native}
for j in manifest['jobs']:
 run=root/'calendar_safe_regional_runs/finetune'/j['finetune_id'];p=run/j['finetune_id']/'predictions.csv'
 v=json.load(open(run/'VERIFIED.json'));assert hashlib.sha256(p.read_bytes()).hexdigest()==v['predictions_sha256']
 candidates=list(Path('.scratch_pilot_run').glob(f'out_finetune*/pilot_local_only_control_fold{j["fold"]}_s{j["seed"]}/predictions.csv'))
 assert len(candidates)==1,(j,candidates)
 c=candidates[0]; a,b=load(c),load(p);xs=sorted(a.keys()&b.keys());assert xs
 for x in xs:assert abs(a[x][0]-b[x][0])<1e-6
 for arm,d in [('control',a),('calendar_safe',b)]:
  rows=[(x,a[x][0],d[x][1]) for x in xs];r=dict(fold=j['fold'],seed=j['seed'],arm=arm,**metrics(rows));results.append(r)
  with (out/f'fold{j["fold"]}_s{j["seed"]}_{arm}.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['x','observed','predicted']);w.writerows(rows)
 sources.append(dict(fold=j['fold'],seed=j['seed'],control=str(c.resolve()),control_sha256=hashlib.sha256(c.read_bytes()).hexdigest(),candidate=str(p),common_hours=len(xs)))
with (out/'paired_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
(out/'sources.json').write_text(json.dumps(sources,indent=2))
for i in range(0,len(results),2):
 a,b=results[i:i+2];print(a['fold'],a['seed'],'NSE',round(a['nse'],4),round(b['nse'],4),'delta',round(b['nse']-a['nse'],4),'KGE',round(a['kge'],3),round(b['kge'],3),'MAE',round(a['mae'],4),round(b['mae'],4))
