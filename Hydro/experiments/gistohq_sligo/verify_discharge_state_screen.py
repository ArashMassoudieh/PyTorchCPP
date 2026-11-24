#!/usr/bin/env python3
"""Independent scalar verification and descriptive events for discharge-state screen."""
import csv,hashlib,itertools,json,math,sys
from pathlib import Path
from verify_preissue_physical_calibration import metrics,check
from analyze_preissue_validation_events import event_bank


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,split):
    with p.open() as f:rows=list(csv.DictReader(f))
    x=[int(v['target_x']) for v in rows];lo,hi=(0,17543) if split=='train' else (17543,26303)
    if len(set(x))!=len(x) or x!=sorted(x) or len(x)!=(13072 if split=='train' else 7112) or min(x)<=lo or max(x)>hi:raise ValueError('Invalid support')
    values=[[float(v[k]) for v in rows] for k in ['observed','origin_q','available_effective_rain','predicted']]
    if any(not math.isfinite(v) or v<0 for a in values for v in a):raise ValueError('Invalid numeric value')
    return x,*values

def forecast(q,r,params,gain):
    k,l,f=params
    base=[a*(f*math.exp(-k)+(1-f)*math.exp(-l)) for a in q]
    response=[a*(f*(1+math.expm1(-k)/k)+(1-f)*(1+math.expm1(-l)/l)) for a in r]
    return [a+gain*b for a,b in zip(base,response)],base,response

def verify(root):
    plan=json.loads((root/'plan.json').read_text());s=json.loads((root/'training_selection.json').read_text());c=json.loads((root/'completion.json').read_text())
    for name,h in plan['hashes'].items():
        saved=root/'provenance'/Path(name).name
        if sha(saved if saved.exists() else name)!=h:raise ValueError('Input/source hash mismatch')
    for name,h in c['artifacts'].items():
        if sha(root/name)!=h:raise ValueError('Artifact hash mismatch')
    if sha(root/'training_selection.json')!=c['selection_sha256']:raise ValueError('Selection changed')
    grid=list(itertools.product([.1,.4,1.],[.005,.025,.05],[.3,.7,.9]))
    if [list(p) for p in grid]!=plan['shapes'] or len(s['candidates'])!=27:raise ValueError('Grid drift')
    x,y,q,r,_=read(root/'train_native.csv','train');b=metrics(y,q);check(s['baseline'],b)
    for index,candidate in enumerate(s['candidates']):
        if candidate['index']!=index or candidate['params']!=list(grid[index]):raise ValueError('Candidate identity mismatch')
        _,base,response=forecast(q,r,candidate['params'],0);total=math.fsum(response);sy=math.fsum(y);sb=math.fsum(base)
        lo=max(0.,(.95*sy-sb)/total);hi=min(1.,(1.05*sy-sb)/total)
        if lo>hi:
            if candidate['eligible'] or 'gain' in candidate:raise ValueError('Infeasible gain accepted')
            continue
        gain=max(lo,min(hi,math.fsum(a*(v-p) for a,v,p in zip(response,y,base))/math.fsum(a*a for a in response)))
        if abs(gain-candidate['gain'])>1e-11:raise ValueError('Gain mismatch')
        prediction,_,_=forecast(q,r,candidate['params'],gain);m=metrics(y,prediction);check(candidate['metrics'],m)
        if (m['nse']>=b['nse']+.02 and m['mae']<=b['mae']*1.02)!=candidate['eligible']:raise ValueError('Training gate mismatch')
    best=max((v for v in s['candidates'] if v['eligible']),key=lambda v:(v['metrics']['nse'],-v['index']))
    if best!=s['selected'] or best!=c['selected']:raise ValueError('Selection mismatch')
    vx,vy,vq,vr,actual=read(root/'validation_replay.csv','validation');prediction,_,_=forecast(vq,vr,best['params'],best['gain'])
    error=max(abs(a-p) for a,p in zip(actual,prediction))
    if error>1e-10:raise ValueError('Native C++/independent equation mismatch')
    baseline=metrics(vy,vq);candidate=metrics(vy,actual)
    yy=sorted(y);pos=(len(y)-1)*.9;i=int(pos);q90=yy[i]+(yy[min(i+1,len(y)-1)]-yy[i])*(pos-i)
    high=[i for i,v in enumerate(vy) if v>q90]
    for m,p in [(baseline,vq),(candidate,actual)]:m['high_flow_rmse']=math.sqrt(math.fsum((p[i]-vy[i])**2 for i in high)/len(high))
    check(c['validation_persistence'],baseline);check(c['validation_candidate'],candidate)
    gates=dict(nse=candidate['nse']>=baseline['nse']+.02,r_squared=candidate['r_squared']>=baseline['r_squared']+.01,mae=candidate['mae']<=baseline['mae']*1.02,high_flow=candidate['high_flow_rmse']<=baseline['high_flow_rmse']*1.02,bias=abs(candidate['pbias_percent'])<=abs(baseline['pbias_percent'])+2)
    if gates!=c['gates'] or all(gates.values())!=c['advanced']:raise ValueError('Validation gates mismatch')
    target=candidate['nse']>=.6 and candidate['r_squared']>=.6
    if target!=c['target_reached']:raise ValueError('Target mismatch')
    # Descriptive only, after immutable selection: same observation-defined
    # storm windows used in the prior v5 diagnostics, never event-wise selection.
    _,events,excluded=event_bank(dict(zip(x,y)),dict(zip(vx,vy)));lookup={v:i for i,v in enumerate(vx)};detail={}
    for name,p in [('persistence',vq),('discharge_initialized',actual)]:
        items=[]
        for e in events:
            times=list(range(e['start_x'],e['end_x']+1));peak=max(times,key=lambda t:p[lookup[t]])
            items.append(dict(**e,predicted_peak_x=peak,peak_timing_error_hours=peak-e['observed_peak_x'],peak_error=p[lookup[peak]]-e['observed_peak']))
        detail[name]=dict(events=items,mean_absolute_peak_timing_hours=math.fsum(abs(e['peak_timing_error_hours']) for e in items)/len(items))
    return dict(status='INDEPENDENTLY_VERIFIED_RESEARCH_PROTOTYPE',training_rows=len(x),validation_rows=len(vx),test_rows=0,training_candidates=27,target_reached=target,advanced=all(gates.values()),gates=gates,validation_candidate=candidate,validation_persistence=baseline,cpp_formula_max_abs=error,complete_events=len(events),excluded_incomplete_events=len(excluded),descriptive_events=detail,production_integration=False,operational_delivery_certified=False)

if __name__=='__main__':
    root=Path(sys.argv[1]);report=verify(root)
    with (root/'independent_review.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print('PASS independent hashes, all 27 train fits, selection,7112 equations, metrics and frozen gates')
