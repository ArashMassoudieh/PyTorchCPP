from pathlib import Path
import csv,json,statistics
r=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023');d=r/'component_diagnostics';atlas=r/'six_year_physics_failure_atlas'
rows=list(csv.DictReader((atlas/'event_metrics.csv').open()));top=sorted([x for x in rows if x['model']=='urban_no_detention'],key=lambda x:float(x['sse']),reverse=True)[:5]
meta={x['event']:x for x in csv.DictReader((atlas/'event_summary.csv').open())};result=[]
for event in top:
 year=meta[event['event']]['observed_peak_utc'][:4];base={float(x['x']):x for x in csv.DictReader((d/f'baseline_{year}.csv').open())};lo,hi=float(event['start_x']),float(event['end_x']);preds=[]
 for seed in [42,123,27182]:
  p=r/f'six_year_physics_analysis/out_of_sample_urban_no_detention_s{seed}.csv';preds.append({float(x['x']):x for x in csv.DictReader(p.open()) if lo<=float(x['x'])<=hi})
 xs=sorted(set.intersection(*(set(x) for x in preds))); obs={x:float(preds[0][x]['observed']) for x in xs};op=max(xs,key=lambda x:obs[x]);pred={x:statistics.median(float(z[x]['predicted']) for z in preds) for x in xs}
 for x in xs:assert abs(float(base[x]['observed'])-obs[x])<1e-6
 bp=max(xs,key=lambda x:float(base[x]['baseline']));rain=[float(base[x]['precipitation']) for x in xs]
 result.append(dict(event=event['event'],year=year,observed_peak_utc=meta[event['event']]['observed_peak_utc'],observed_peak=obs[op],baseline_at_observed_peak=float(base[op]['baseline']),baseline_max=float(base[bp]['baseline']),hybrid_at_observed_peak=pred[op],residual_at_observed_peak=pred[op]-float(base[op]['baseline']),rain_max_mm_h=max(rain),common_native_hours=len(xs),missing_common_hours=int(hi-lo+1)-len(xs),segments=len({base[x]['segment'] for x in xs})))
with (d/'top_five_components.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
for x in result:print(x)
