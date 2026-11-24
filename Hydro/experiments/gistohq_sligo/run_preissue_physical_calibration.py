#!/usr/bin/env python3
"""Finite TRAIN-only routing/gain selection, followed by one frozen validation evaluation.
No test loading, retries, old campaign mutation, or neural fitting. This is a new
v5 held-rate calibration, not a relaunch of earlier forcing-recipe searches.
"""
import argparse, csv, hashlib, itertools, json, subprocess
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write('\n')


def scores(y, p):
    if len(y) < 2 or not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError('Invalid scoring vectors')
    sst = np.sum((y-y.mean())**2)
    if sst <= 0 or y.sum() <= 0 or np.var(p) <= 0:
        raise ValueError('Undefined metrics')
    return dict(n=len(y), nse=float(1-np.sum((p-y)**2)/sst),
                r_squared=float(np.corrcoef(y,p)[0,1]**2),
                mae=float(np.mean(np.abs(p-y))),
                pbias_percent=float(100*np.sum(p-y)/y.sum()))


def fit_gain(y, p):
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y<0).any() or (p<0).any() or np.dot(p,p)<=0:
        raise ValueError('Invalid gain fitting inputs')
    return float(np.clip(np.dot(y,p)/np.dot(p,p), 0., 1.))


def load(p):
    with Path(p).open() as f:
        rows=list(csv.DictReader(f))
    x=np.array([int(r['target_x']) for r in rows])
    if len(set(x))!=len(x) or np.any(np.diff(x)<=0) or np.any(x>=77016):
        raise ValueError('Invalid target identities')
    return x, *[np.array([float(r[k]) for r in rows]) for k in ['observed','wet_unit_gain','dry_unit_gain']]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('pilot',type=Path); ap.add_argument('binary',type=Path); ap.add_argument('output',type=Path)
    a=ap.parse_args(); a.output.mkdir(exist_ok=False)
    config=json.loads((a.pilot/'configs/pinn_causal_forecast_rate_hold_v1_s42.json').read_text())
    manifest=Path(config['hydrofusion_native_manifest_path']); weather=Path(config['hydrofusion_preissue_rain_path']); support=a.pilot/'support.csv'
    grid=list(itertools.product([.25,1.,4.],[.025,.1,.4],[.3,.7,.9]))
    plan=dict(schema='preissue_physical_train_calibration_v1', grid=grid, gain_bounds=[0,1],
        selection='TRAIN least squares gain per shape; max TRAIN NSE among MAE<=default*1.02 and abs(PBIAS)<=abs(default)+5; ties grid order',
        validation='one frozen selected configuration versus original wet/dry on all 7112 matched rows; no subsequent tuning',
        gates=dict(nse_improvement_over_dry=.005,mae_ratio_max=1.02,absolute_bias_increase_max=5),
        bounds=dict(shape_candidates=27,neural_fits=0,test_rows=0,retries=0),
        hashes={str(p):sha(p) for p in [manifest,weather,support,a.binary,Path(__file__),Path(__file__).with_name('calibrate_preissue_physical.cpp')]})
    save(a.output/'plan.json',plan)
    # Crop physically before the calibration executable can parse any validation Q.
    subprocess.run(['python3',str(Path(__file__).with_name('prepare_hydrofusion_search_slice.py')),str(manifest),str(a.output/'train_input'),'17543'],check=True)
    trainmanifest=a.output/'train_input/native_input_manifest.json'
    def run(split,params,name):
        m=trainmanifest if split=='train' else manifest
        p=a.output/(name+'.csv')
        with (a.output/(name+'.log')).open('x') as log:
            subprocess.run([str(a.binary),str(m),sha(m),str(weather),str(support),split,*map(str,params),str(p)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        return load(p)
    baseline=run('train',(1.,.1,.7),'train_default'); x,y,p,d=baseline
    base=scores(y,p); candidates=[]
    for i,params in enumerate(grid):
        xx,yy,pp,dd=baseline if params==(1.,.1,.7) else run('train',params,f'train_{i:02d}')
        if not np.array_equal(x,xx) or not np.array_equal(y,yy): raise ValueError('Support drift')
        gain=fit_gain(y,pp); metrics=scores(y,gain*pp)
        eligible=metrics['mae']<=base['mae']*1.02 and abs(metrics['pbias_percent'])<=abs(base['pbias_percent'])+5
        candidates.append(dict(index=i,params=params,gain=gain,metrics=metrics,eligible=eligible))
    eligible=[c for c in candidates if c['eligible']]
    if not eligible: raise ValueError('No training-eligible candidate; do not evaluate validation')
    selected=max(eligible,key=lambda c:(c['metrics']['nse'],-c['index']))
    save(a.output/'training_selection.json',dict(default=base,candidates=candidates,selected=selected,training_manifest_sha256=sha(trainmanifest)))
    # The immutable selection file exists BEFORE the first validation execution.
    frozen=sha(a.output/'training_selection.json')
    vx,vy,vp,vd=run('validation',selected['params'],'validation_selected')
    bx,by,bp,bd=run('validation',(1.,.1,.7),'validation_default')
    if not np.array_equal(vx,bx) or not np.array_equal(vy,by): raise ValueError('Validation support drift')
    dry=scores(by,bd); wet=scores(by,bp); candidate=scores(vy,vp*selected['gain'])
    passed=candidate['nse']>=dry['nse']+.005 and candidate['mae']<=dry['mae']*1.02 and abs(candidate['pbias_percent'])<=abs(dry['pbias_percent'])+5
    if sha(a.output/'training_selection.json')!=frozen: raise ValueError('Selection changed')
    save(a.output/'result.json',dict(status='COMPLETE',selection_sha256=frozen,selected=selected,validation_default_dry=dry,validation_default_wet=wet,validation_selected=candidate,gate_pass=passed,
        interpretation='Reused validation, train-calibrated physical-only settings; no hybrid checkpoint change or operational delivery claim.',
        artifacts={p.name:sha(p) for p in a.output.glob('*.csv')}))
    print(json.dumps(json.loads((a.output/'result.json').read_text()),indent=2))

if __name__=='__main__': main()
