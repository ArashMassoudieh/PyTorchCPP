#!/usr/bin/env python3
"""Descriptive preissue storm diagnostics; observation-defined events, never promotion gates."""
import argparse,csv,json,math
from pathlib import Path
from verify_hydrofusion_preissue_results import identity,number,predictions,SHA

def event_bank(train,validation):
    q=sorted(train.values());position=(len(q)-1)*.9;i=int(position);threshold=q[i]+(q[min(i+1,len(q)-1)]-q[i])*(position-i)
    high=sorted(x for x,v in validation.items() if v>=threshold);clusters=[]
    for x in high:
        if not clusters or x-clusters[-1][-1]>24:clusters.append([])
        clusters[-1].append(x)
    events=[];excluded=[]
    for cluster in clusters:
        start,end=cluster[0]-6,cluster[-1]+6
        if not set(range(start,end+1)).issubset(validation):excluded.append(dict(start_x=start,end_x=end,reason='incomplete_common_support'));continue
        times=list(range(start,end+1));peak=max(times,key=lambda x:validation[x])
        events.append(dict(start_x=start,end_x=end,observed_peak_x=peak,observed_peak=validation[peak],hours=len(times)))
    return threshold,events,excluded

def run(root,output):
    root=Path(root).resolve(strict=True);plan=json.loads((root/'screen_plan.json').read_text());train={};validation={}
    if not plan['development_only'] or plan['test_scores']!=0:raise ValueError('Development-only analysis required')
    with (root/'support.csv').open() as f:
        for r in csv.DictReader(f):
            if r['split']=='train':target=train;x=identity(r['target_x'],0,17543)
            elif r['split']=='validation':target=validation;x=identity(r['target_x'],17543,26303)
            else:raise ValueError('Forbidden split before value access')
            if x in target:raise ValueError('Duplicate support')
            target[x]=number(r['observed'])
    threshold,events,excluded=event_bank(train,validation);results={}
    for j in plan['jobs']:
        fit=root/'fits'/j['name']
        if not (fit/'completion.json').exists():continue
        values=predictions(fit/'validation_predictions.csv',j['family'],validation);items=[]
        for event in events:
            times=range(event['start_x'],event['end_x']+1);predicted_peak=max(times,key=lambda x:values[x][1]);observed_volume=math.fsum(validation[x] for x in times)
            items.append(dict(**event,predicted_peak_x=predicted_peak,predicted_peak=values[predicted_peak][1],peak_timing_error_hours=predicted_peak-event['observed_peak_x'],peak_error=values[predicted_peak][1]-event['observed_peak'],volume_bias_percent=100*(math.fsum(values[x][1] for x in times)/observed_volume-1)))
        results[j['name']]=items
    report=dict(schema='HydroFusionPreissueDescriptiveEvents-v1',plan_sha256=SHA(root/'screen_plan.json'),support_sha256=SHA(root/'support.csv'),training_q90=threshold,discharge_units='mm/h of runoff depth',catchment_area_m2=json.loads((root/'prepared_input/native_input_manifest.json').read_text())['catchment_area_m2'],event_definition='Validation Q>=training Q90; merge high-flow hours separated by<=24h; add6h each end; reject incomplete windows',events=events,excluded_incomplete_events=excluded,completed_models=results,test_rows=0,not_a_promotion_gate=True,limitations=['Exploratory reused-validation diagnostics, not event-selected model promotion.','No interpolation, truncated storms or incomplete-window scoring.'])
    with Path(output).open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print('SAVED descriptive events='+str(len(events))+' excluded='+str(len(excluded))+' completed_models='+str(len(results)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('new_output',type=Path);a=p.parse_args();run(a.root,a.new_output)
