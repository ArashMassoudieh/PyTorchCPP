#!/usr/bin/env python3
"""Independent read-only result/receipt verifier. Does not launch models or read test data."""
import argparse,csv,hashlib,json,math
from pathlib import Path
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def number(s):
    v=float(s)
    if not math.isfinite(v) or v<0:raise ValueError('Nonfinite/negative discharge')
    return v
def identity(s,low,high):
    x=float(s)
    if not math.isfinite(x) or not x.is_integer() or not low<x<=high or x>=77016:raise ValueError('Forbidden prediction identity before observed parsing')
    return int(x)
def predictions(path,family,expected):
    rows={}
    with Path(path).open() as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['approach','index','split','x','observed','predicted','residual']:raise ValueError('Prediction schema')
        for i,r in enumerate(reader):
            if None in r or any(v is None for v in r.values()):raise ValueError('Ragged prediction row')
            x=identity(r['x'],17543,26303)
            if x not in expected or x in rows or r['split']!='validation' or r['approach']!=family or r['index']!=str(i):raise ValueError('Prediction identity/index/support mismatch')
            o,p=number(r['observed']),number(r['predicted']);error=float(r['residual'])
            if o!=expected[x] or not math.isfinite(error) or not math.isclose(error,p-o,rel_tol=1e-9,abs_tol=1e-9):raise ValueError('Observation/residual mismatch')
            rows[x]=(o,p)
    if set(rows)!=set(expected):raise ValueError('Missing prediction rows')
    return rows

def independent_metrics(observed,predicted,q90):
    if len(observed)!=len(predicted) or not observed:raise ValueError('Metric support')
    n=len(observed);mo=math.fsum(observed)/n;mp=math.fsum(predicted)/n
    vo=math.fsum((o-mo)**2 for o in observed);vp=math.fsum((p-mp)**2 for p in predicted)
    cross=math.fsum((o-mo)*(p-mp) for o,p in zip(observed,predicted));sse=math.fsum((p-o)**2 for o,p in zip(observed,predicted))
    if vo==0:raise ValueError('Undefined observed variance')
    high=[(p-o)**2 for o,p in zip(observed,predicted) if o>=q90]
    correlation=cross/math.sqrt(vo*vp) if vp else None
    kge=1-math.sqrt((correlation-1)**2+(math.sqrt(vp/vo)-1)**2+(mp/mo-1)**2) if correlation is not None and mo else None
    low=[(p-o)**2 for o,p in zip(observed,predicted) if o<q90]
    bias=100*math.fsum(p-o for o,p in zip(observed,predicted))/math.fsum(observed)
    return dict(n=n,kge=kge,nse=1-sse/vo,r_squared=cross*cross/(vo*vp) if vp else None,rmse=math.sqrt(sse/n),mae=math.fsum(abs(p-o) for o,p in zip(observed,predicted))/n,pbias_percent=bias,absolute_pbias_percent=abs(bias),high_flow_n=len(high),remaining_n=len(low),remaining_rmse=math.sqrt(math.fsum(low)/len(low)) if low else None,high_flow_rmse=math.sqrt(math.fsum(high)/len(high)) if high else None)

def verify(root,partial=False):
    root=Path(root).resolve(strict=True);plan=json.loads((root/'screen_plan.json').read_text());claim=json.loads((root/'launch_claim.json').read_text())
    if claim['plan_sha256']!=SHA(root/'screen_plan.json'):raise ValueError('Plan/claim changed')
    for p,h in plan['pins'].items():
        if SHA(p)!=h:raise ValueError('Pinned input changed: '+p)
    families={'ffn_causal_forecast','lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast','pinn_causal_forecast'}
    expected={(f,r) for f in families for r in ('zero_channel_v1','rate_hold_v1')}
    if len(plan['jobs'])!=10 or {(j['family'],j['recipe']) for j in plan['jobs']}!=expected or len({j['name'] for j in plan['jobs']})!=10 or any(j['neural']!=(j['family']!='pinn_causal_forecast') for j in plan['jobs']) or not plan['development_only'] or plan['test_scores']!=0:raise ValueError('Prospective inventory/scope mismatch')
    train={};val={};persist={}
    with (root/'support.csv').open() as f:
        for r in csv.DictReader(f):
            if r['split']=='train':target=train;x=identity(r['target_x'],0,17543)
            elif r['split']=='validation':target=val;x=identity(r['target_x'],17543,26303)
            else:raise ValueError('Forbidden support split')
            if x in target:raise ValueError('Duplicate support')
            target[x]=number(r['observed'])
            if target is val:persist[x]=number(r['persistence'])
    if (len(train),len(val))!=(13072,7112):raise ValueError('Support counts')
    ordered=sorted(train.values());k=(len(ordered)-1)*.9;i=int(k);q90=ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(k-i)
    results={};finished=[]
    for j in plan['jobs']:
        fit=root/'fits'/j['name'];receipt=fit/'completion.json'
        if not receipt.exists():
            if partial:continue
            raise ValueError('Job not complete: '+j['name'])
        rec=json.loads(receipt.read_text())
        if rec['exit_code']!=0 or rec['wall_seconds']>28800 or rec['peak_sampled_rss_bytes']>12*1024**3:raise ValueError('Job resource/exit receipt')
        actual={str(p.relative_to(fit)) for p in fit.rglob('*') if p.is_file() and p!=receipt}
        if actual!=set(rec['sha256']):raise ValueError('Unhashed/missing job artifacts')
        for p,h in rec['sha256'].items():
            if SHA(fit/p)!=h:raise ValueError('Job artifact changed: '+p)
        log=(fit/'fresh_reload.log').read_text()
        if 'PASS all_validation_rows=7112 production_target_free_checks=12' not in log:raise ValueError('Missing full reload/API proof')
        bundle=fit/'bundles'/j['name'];a=predictions(bundle/'predictions.csv',j['family'],val);b=predictions(fit/'validation_predictions.csv',j['family'],val)
        for x in a:
            if abs(a[x][1]-b[x][1])>1e-5+1e-5*abs(a[x][1]):raise ValueError('Raw/fresh prediction mismatch')
        keys=sorted(b);results[j['name']]=independent_metrics([val[x] for x in keys],[b[x][1] for x in keys],q90);finished.append(j['name'])
    out=dict(status='PARTIAL_INDEPENDENTLY_VERIFIED' if partial else 'COMPLETE_INDEPENDENTLY_VERIFIED',plan_sha256=SHA(root/'screen_plan.json'),completed=finished,validation=results,test_rows=0,persistence=independent_metrics([val[x] for x in sorted(val)],[persist[x] for x in sorted(val)],q90),q90_training_only=q90)
    if not partial:
        saved=json.loads((root/'results_summary.json').read_text());complete=json.loads((root/'screen_completion.json').read_text())
        if set(complete['completed'])!=set(finished) or complete['plan_sha256']!=out['plan_sha256']:raise ValueError('Completion inventory/plan')
        for name,item in results.items():
            for k,v in item.items():
                actual=saved['validation'][name][k]
                if v is None:
                    if actual is not None:raise ValueError('Undefined metric mismatch')
                elif not math.isclose(v,actual,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Independent raw score mismatch: '+name+'/'+k)
        decisions={};g=plan['gates']
        for family in {j['family'] for j in plan['jobs']}:
            a=results[family+'_zero_channel_v1_s42'];b=results[family+'_rate_hold_v1_s42']
            checks=dict(nse=b['nse']-a['nse']>=g['nse_gain'],r_squared=b['r_squared'] is not None and a['r_squared'] is not None and b['r_squared']-a['r_squared']>=g['r_squared_gain'],mae=b['mae']<=a['mae']*(1+g['mae_relative_increase']),high_flow=b['high_flow_rmse']<=a['high_flow_rmse']*(1+g['high_flow_rmse_relative_increase']),bias=abs(b['pbias_percent'])<=abs(a['pbias_percent'])+g['absolute_pbias_increase_points'])
            if checks!=saved['decisions'][family]['checks'] or all(checks.values())!=saved['decisions'][family]['passed']:raise ValueError('Independent gate mismatch')
            decisions[family]=dict(checks=checks,passed=all(checks.values()))
        out['decisions']=decisions
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--partial',action='store_true');p.add_argument('--new-report',type=Path);a=p.parse_args();out=verify(a.root,a.partial)
    if a.new_report:
        with a.new_report.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out,indent=2,allow_nan=False))
