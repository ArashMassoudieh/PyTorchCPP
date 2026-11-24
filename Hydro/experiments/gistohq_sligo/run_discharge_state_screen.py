#!/usr/bin/env python3
"""Finite observed-discharge initialized physical forecast screen, TRAIN-selected.
This is not the original free-running physical method or a neural PINN.
"""
import argparse,csv,itertools,json,shutil,subprocess
from pathlib import Path
import numpy as np
from run_preissue_physical_calibration import scores,sha,save
HERE=Path(__file__).resolve().parent


def predict(q,r,params,gain):
    fast,slow,fraction=params
    if q.ndim!=1 or q.shape!=r.shape or not np.isfinite(q).all() or not np.isfinite(r).all() or (q<0).any() or (r<0).any() or not all(np.isfinite(v) for v in (fast,slow,fraction,gain)) or fast<=0 or slow<=0 or not 0<=fraction<=1 or not 0<=gain<=1:
        raise ValueError('Invalid forecast inputs')
    # Exact one-hour outflow from dry reconstructed end-of-hour analysis stores.
    base=q*(fraction*np.exp(-fast)+(1-fraction)*np.exp(-slow))
    response=r*(fraction*(1+np.expm1(-fast)/fast)+(1-fraction)*(1+np.expm1(-slow)/slow))
    p=base+gain*response
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Nonfinite/domain prediction')
    return p,base,response


def read(path,split):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    x=np.array([int(v['target_x']) for v in rows]);lower,upper=(0,17543) if split=='train' else (17543,26303)
    if len(x)!=(13072 if split=='train' else 7112) or len(set(x))!=len(x) or (np.diff(x)<=0).any() or x.min()<=lower or x.max()>upper:raise ValueError('Support identity mismatch')
    values=[np.array([float(v[k]) for v in rows]) for k in ['observed','origin_q','available_effective_rain','predicted']]
    if any(not np.isfinite(v).all() or (v<0).any() for v in values):raise ValueError('Invalid native feature/value')
    return x,*values


def fit_rain_gain(y,base,response):
    if not np.isfinite(y).all() or not np.isfinite(base).all() or not np.isfinite(response).all() or any((v<0).any() for v in (y,base,response)) or y.sum()<=0:
        raise ValueError('Invalid training gain arrays')
    total=float(response.sum())
    if total<=0:return 0. if abs(100*(base.sum()-y.sum())/y.sum())<=5 else None
    lo=max(0.,float((.95*y.sum()-base.sum())/total));hi=min(1.,float((1.05*y.sum()-base.sum())/total))
    if lo>hi:return None
    denominator=float(response@response)
    if denominator<=0:raise ValueError('Zero response norm')
    return float(np.clip(response@(y-base)/denominator,lo,hi))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('previous',type=Path);ap.add_argument('binary',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    a.output.mkdir(exist_ok=False);oldplan=json.loads((a.previous/'plan.json').read_text())
    native=next(Path(p) for p in oldplan['hashes'] if p.endswith('native_input_manifest.json'))
    weather=next(Path(p) for p in oldplan['hashes'] if p.endswith('preissue_rain.csv'))
    support=next(Path(p) for p in oldplan['hashes'] if p.endswith('support.csv'))
    training=a.previous/'train_input/native_input_manifest.json'
    grid=list(itertools.product([.1,.4,1.],[.005,.025,.05],[.3,.7,.9]))
    files=[native,weather,support,training,a.binary,Path(__file__),HERE/'hydrofusion_discharge_state_forecast.h',HERE/'extract_discharge_state_features.cpp']
    plan=dict(schema='discharge_initialized_physical_screen_v1',shapes=grid,
        initialization='Qorigin is previous-hour mean; reconstruct dry previous-hour end stores by fixed fast/slow discharge fraction; count initial water as analysis storage',
        inputs='observed discharge at origin plus preissue available rainfall/PET only; no target Q enters prediction',
        scope='idealized retrospective zero-latency observed-Q analysis, archive rain release proxy; not free-running physical model or operational certification',
        training='bounded gain[0,1] least squares with abs training volume bias<=5%; MAE<=persistence*1.02 and NSE>=persistence+.02',
        selection='max TRAIN NSE among eligible; ties earlier frozen shape order',
        validation_gates=dict(nse_gain=.02,r_squared_gain=.01,mae_max_ratio=1.02,high_flow_rmse_max_ratio=1.02,absolute_bias_increase_points=2),
        target=dict(nse=.60,pearson_r_squared=.60),bounds=dict(shape_candidates=27,validation_candidates=1,neural_fits=0,test_rows=0,retries=0),hashes={str(p):sha(p) for p in files})
    save(a.output/'plan.json',plan);(a.output/'provenance').mkdir()
    for p in files:
        if p.suffix in ('.py','.cpp','.h') or p==a.binary:shutil.copy2(p,a.output/'provenance'/p.name)
    def extract(split,params=None,gain=None):
        m=training if split=='train' else native;file=a.output/(split+'_native.csv' if params is None else split+'_replay.csv')
        args=[str(a.binary),str(m),sha(m),str(weather),str(support),split,str(file)]
        if params is not None:args+=list(map(str,params))+[str(gain)]
        with (a.output/(file.stem+'.log')).open('x') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        return read(file,split)
    x,y,q,r,_=extract('train');baseline=scores(y,q);candidates=[];q90=float(np.quantile(y,.9))
    for index,params in enumerate(grid):
        _,base,response=predict(q,r,params,0.)
        gain=fit_rain_gain(y,base,response)
        if gain is None:candidates.append(dict(index=index,params=params,eligible=False,reason='No volume-feasible gain'));continue
        prediction,_,_=predict(q,r,params,gain);metrics=scores(y,prediction)
        eligible=metrics['nse']>=baseline['nse']+.02 and metrics['mae']<=baseline['mae']*1.02
        candidates.append(dict(index=index,params=params,gain=gain,metrics=metrics,eligible=eligible))
    eligible=[c for c in candidates if c['eligible']]
    selected=max(eligible,key=lambda c:(c['metrics']['nse'],-c['index'])) if eligible else None
    save(a.output/'training_selection.json',dict(baseline=baseline,training_q90=q90,candidates=candidates,selected=selected))
    if selected is None:
        save(a.output/'completion.json',dict(status='CLOSED_TRAINING_GATE_FAILED',test_rows=0,validation_rows=0));print('No training-eligible configuration; no validation read');return
    frozen=sha(a.output/'training_selection.json')
    # Selected formula must agree with the C++ reservoir implementation on ALL
    # training rows before loading validation observations.
    tx,ty,tq,tr,actual=extract('train',selected['params'],selected['gain']);expected,_,_=predict(q,r,selected['params'],selected['gain'])
    if not np.array_equal(x,tx) or not np.array_equal(y,ty) or not np.array_equal(q,tq) or not np.array_equal(r,tr) or np.max(np.abs(actual-expected))>1e-10:raise ValueError('Train C++/formula parity failed')
    vx,vy,vq,vr,actual=extract('validation',selected['params'],selected['gain']);expected,_,_=predict(vq,vr,selected['params'],selected['gain'])
    error=float(np.max(np.abs(actual-expected)))
    if error>1e-10:raise ValueError('Validation C++/formula parity failed')
    b=scores(vy,vq);c=scores(vy,actual);high=vy>q90
    for m,p in [(b,vq),(c,actual)]:m['high_flow_rmse']=float(np.sqrt(np.mean((p[high]-vy[high])**2)))
    gates=dict(nse=c['nse']>=b['nse']+.02,r_squared=c['r_squared']>=b['r_squared']+.01,mae=c['mae']<=b['mae']*1.02,high_flow=c['high_flow_rmse']<=b['high_flow_rmse']*1.02,bias=abs(c['pbias_percent'])<=abs(b['pbias_percent'])+2)
    if sha(a.output/'training_selection.json')!=frozen:raise ValueError('Selection changed')
    save(a.output/'completion.json',dict(status='COMPLETE_RESEARCH_PROTOTYPE',selected=selected,selection_sha256=frozen,validation_persistence=b,validation_candidate=c,gates=gates,advanced=all(gates.values()),target_reached=c['nse']>=.6 and c['r_squared']>=.6,validation_rows=len(vx),test_rows=0,cpp_formula_max_abs=error,
        artifacts={p.name:sha(p) for p in a.output.glob('*.csv')},warning='Observed-discharge initialized physical prototype; not original free-running PINN, no production dispatch or operational/generalization claim.'))
    print(json.dumps(json.loads((a.output/'completion.json').read_text()),indent=2))

if __name__=='__main__':main()
