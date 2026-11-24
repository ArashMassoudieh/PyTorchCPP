#!/usr/bin/env python3
"""A new finite structural screen using saved TRAIN-only routing components.
Never reruns old fits. Selection cannot load validation until saved and frozen.
No production recipe change is implied by this research calculation.
"""
import argparse,json,subprocess,shutil
from pathlib import Path
import numpy as np
from preissue_physical_components import box_least_squares,component_design
from run_preissue_physical_calibration import load,save,scores,sha

HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('previous',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();a.output.mkdir(exist_ok=False)
    old=json.loads((a.previous/'training_selection.json').read_text())
    inputs={}
    for c in old['candidates']:
        inputs[c['index']]=a.previous/('train_default.csv' if c['params']==[1.,.1,.7] else f"train_{c['index']:02d}.csv")
    plan=dict(schema='preissue_physical_components_research_v1',model='past_gain*dry + storm_gain*(wet-dry) + background',
        parameters=dict(past_gain=[0,1],storm_gain=[0,1],background='0..median positive TRAIN target, optional separate empirical source'),
        candidates=54,shapes=[c['params'] for c in old['candidates']],families=['separate_storm','separate_storm_background'],
        fitting='exact box least squares on TRAIN only; no observed-Q inference input',
        train_gates=dict(mae_max_dry_ratio=1.02,absolute_pbias_max_percent=20,nse_min_gain=.005),
        selection='max TRAIN NSE among eligible; ties simpler model then earlier shape',
        validation_gates=dict(nse_min_gain=.02,r_squared_min_gain=.01,mae_max_ratio=1.02,high_flow_rmse_max_ratio=1.02,absolute_bias_increase_max=2),
        goal=dict(nse=.60,pearson_r_squared=.60),
        bounds=dict(neural_fits=0,new_routing_searches=0,test_rows=0,validation_candidates_max=1,retries=0),
        input_sha256={str(p):sha(p) for p in set(inputs.values())|{a.previous/'training_selection.json'}},
        source_sha256={str(p):sha(p) for p in [Path(__file__),HERE/'preissue_physical_components.py',HERE/'run_preissue_physical_calibration.py']})
    save(a.output/'plan.json',plan)
    (a.output/'provenance').mkdir()
    for p in plan['source_sha256']:shutil.copy2(p,a.output/'provenance'/Path(p).name)
    x,y,wet,dry=load(a.previous/'train_default.csv');base=scores(y,dry)
    positive=y[y>0]
    if not len(positive):raise ValueError('No training scale')
    cap=float(np.median(positive));q90=float(np.quantile(y,.9));candidates=[]
    for c in old['candidates']:
        xx,yy,w,d=load(inputs[c['index']])
        if not np.array_equal(x,xx) or not np.array_equal(y,yy) or x.min()<1 or x.max()>17543:raise ValueError('Train identity drift')
        for background in [False,True]:
            design=component_design(d,w,background);upper=[1,1,cap] if background else [1,1]
            coefficients=box_least_squares(design,y,upper);prediction=design@coefficients;m=scores(y,prediction)
            eligible=m['mae']<=base['mae']*1.02 and abs(m['pbias_percent'])<=20 and m['nse']>=base['nse']+.005
            candidates.append(dict(shape_index=c['index'],params=c['params'],background=background,coefficients=coefficients.tolist(),metrics=m,eligible=eligible))
    eligible=[c for c in candidates if c['eligible']]
    selected=max(eligible,key=lambda c:(c['metrics']['nse'],-int(c['background']),-c['shape_index'])) if eligible else None
    selection=dict(training_default_dry=base,background_upper_mm_h=cap,training_q90=q90,candidates=candidates,selected=selected)
    save(a.output/'training_selection.json',selection)
    if selected is None:
        save(a.output/'completion.json',dict(status='CLOSED_TRAINING_GATE_FAILED',validation_rows=0,test_rows=0));print('No training-eligible model; no validation loaded');return
    frozen=sha(a.output/'training_selection.json')
    # Only NOW load the old plan to locate the development manifest. No validation
    # prediction/observed value has entered fitting or selection above.
    oldplan=json.loads((a.previous/'plan.json').read_text())
    manifest=next(Path(p) for p in oldplan['hashes'] if p.endswith('native_input_manifest.json'))
    weather=next(Path(p) for p in oldplan['hashes'] if p.endswith('preissue_rain.csv'))
    support=next(Path(p) for p in oldplan['hashes'] if p.endswith('support.csv'))
    binary=a.previous/'provenance/calibrate_preissue_physical_20261124'
    if sha(binary)!=oldplan['hashes'][next(p for p in oldplan['hashes'] if Path(p).name=='calibrate_preissue_physical_20261124')]:raise ValueError('Probe changed')
    vp=a.output/'validation_components.csv'
    with (a.output/'validation_components.log').open('x') as log:
        subprocess.run([str(binary),str(manifest),sha(manifest),str(weather),str(support),'validation',*map(str,selected['params']),str(vp)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    vx,vy,vwet,vdry=load(vp);prediction=component_design(vdry,vwet,selected['background'])@np.array(selected['coefficients'])
    bx,by,bwet,bdry=load(a.previous/'validation_default.csv')
    if not np.array_equal(vx,bx) or not np.array_equal(vy,by) or vx.min()<=17543 or vx.max()>26303:raise ValueError('Validation identity drift')
    baseline=scores(vy,bdry);candidate=scores(vy,prediction);high=vy>q90
    if not high.any():raise ValueError('No high flow support')
    baseline['high_flow_rmse']=float(np.sqrt(np.mean((bdry[high]-vy[high])**2)))
    candidate['high_flow_rmse']=float(np.sqrt(np.mean((prediction[high]-vy[high])**2)))
    checks=dict(nse=candidate['nse']>=baseline['nse']+.02,r_squared=candidate['r_squared']>=baseline['r_squared']+.01,
        mae=candidate['mae']<=baseline['mae']*1.02,high_flow=candidate['high_flow_rmse']<=baseline['high_flow_rmse']*1.02,
        bias=abs(candidate['pbias_percent'])<=abs(baseline['pbias_percent'])+2)
    if sha(a.output/'training_selection.json')!=frozen:raise ValueError('Selection changed')
    with (a.output/'validation_predictions.csv').open('x') as f:
        f.write('target_x,observed,predicted\n')
        for xx,yy,pp in zip(vx,vy,prediction):f.write(f'{xx},{yy:.17g},{pp:.17g}\n')
    save(a.output/'completion.json',dict(status='COMPLETE_RESEARCH_ONLY',selection_sha256=frozen,selected=selected,validation_baseline=baseline,validation_candidate=candidate,gates=checks,advanced=all(checks.values()),goal_reached=candidate['nse']>=.6 and candidate['r_squared']>=.6,
        validation_rows=len(vx),test_rows=0,validation_components_sha256=sha(vp),validation_predictions_sha256=sha(a.output/'validation_predictions.csv'),
        warning='Reused-validation diagnostic, new component recipe not production-integrated; background is an empirical source outside reservoir ledger.'))
    print(json.dumps(json.loads((a.output/'completion.json').read_text()),indent=2))

if __name__=='__main__':main()
