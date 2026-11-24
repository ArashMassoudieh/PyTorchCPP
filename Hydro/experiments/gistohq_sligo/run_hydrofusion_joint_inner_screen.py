#!/usr/bin/env python3
"""NEW eight-fit internal chronological selection, plus bounded physical analysis.
2015 inner training / 2016 inner validation only. No 2017/test observations.
No outer refits, retries, or expansion are authorized by this driver's claim.
"""
import argparse,csv,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from run_hydrofusion_six_fit_screen import bounded_fit,digest,write_new,directory_bytes
from run_hydrofusion_preissue_screen import LIMITS,FAMILIES
from run_discharge_state_screen import predict,fit_rain_gain
from run_preissue_physical_calibration import scores
HERE=Path(__file__).resolve().parent
NEURAL=FAMILIES[:-1]

def support(root):
    result={s:{} for s in ('train','validation')}
    with (root/'support.csv').open() as f:
        for r in csv.DictReader(f):
            split=r['split'];x=int(r['target_x'])
            if split not in result or not (0<x<=8759 if split=='train' else 8759<x<=17543) or x in result[split]:raise ValueError('Invalid internal support before value access')
            y=float(r['observed']);q=float(r['persistence'])
            if not math.isfinite(y) or not math.isfinite(q) or min(y,q)<0:raise ValueError('Invalid internal observations')
            result[split][x]=(y,q)
    if min(map(len,result.values()))<100:raise ValueError('Insufficient internal support')
    return result

def check(plan):
    if plan['limits']!=LIMITS or len(plan['jobs'])!=8 or {(j['family'],j['width']) for j in plan['jobs']}!={(f,w) for f in NEURAL for w in (32,64)}:raise ValueError('Finite inventory changed')
    for p,h in plan['pins'].items():
        if digest(p)!=h:raise ValueError('Pinned source/input changed '+p)
    for j in plan['jobs']:
        c=json.loads(Path(j['config']).read_text())
        expected={'train_end_x':8759,'validation_end_x':17543,'test_end_x':26303,'epochs':50,'hidden_layers':str(j['width']),'random_seed':42,'hydrofusion_development_only':True,'hydrofusion_preissue_rain_recipe':'rate_hold_v1','hydrofusion_synthetic_verification':False,'hydrofusion_objective':'log_flow_mse_v2'}
        if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Prospective config changed')
        for key,path in [('hydrofusion_native_manifest',plan['manifest']),('hydrofusion_preissue_rain',plan['weather'])]:
            if c[key+'_path']!=path or c[key+'_sha256']!=digest(path):raise ValueError('Config/input binding changed')

def prepare(root,pilot,training_manifest,binary,verifier,feature_probe,state_features):
    root=Path(root).absolute();root.mkdir(exist_ok=False)
    for folder in ['prepared_input','weather','executables','configs','provenance']:(root/folder).mkdir()
    for p in Path(training_manifest).parent.iterdir():
        if p.is_file():shutil.copy2(p,root/'prepared_input'/p.name)
    manifest=root/'prepared_input/native_input_manifest.json';m=json.loads(manifest.read_text())
    if m['target_cutoff_x']!=17543 or m['end_interval_x_exclusive']!=17543 or m['execution_scope']!='train_validation_only':raise ValueError('Physically cropped training package required')
    source=Path(pilot)/'weather/preissue_rain.csv';weather=root/'weather/preissue_rain.csv'
    with source.open() as inp,weather.open('x') as out:
        out.write(next(inp));expected=1
        for line in inp:
            target=int(line.partition(',')[0])
            if target!=expected:raise ValueError('Weather identities not ordered')
            if target<=17543:out.write(line)
            expected+=1
        if expected!=26304:raise ValueError('Weather parent inventory mismatch')
    write_new(root/'weather/preissue_rain_provenance.json',dict(schema='PreissueRainChronologicalPrefix-v1',parent_csv_sha256=digest(source),parent_provenance_sha256=digest(source.parent/'preissue_rain_provenance.json'),prefix_target_end_x=17543,parent_provenance=json.loads((source.parent/'preissue_rain_provenance.json').read_text()),purpose='Internal2015training2016validation; weather-only prefix, no observation transformation'))
    for src,name in [(binary,'HydroBatch'),(verifier,'native_reload'),(feature_probe,'feature_preflight')]:shutil.copy2(src,root/'executables'/name)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    with (root/'feature_preflight.log').open('x') as log:subprocess.run([str(root/'executables/feature_preflight'),str(manifest),digest(manifest),str(weather),digest(weather.parent/'preissue_rain_provenance.json'),str(root/'support.csv')],stdout=log,stderr=subprocess.STDOUT,check=True,env=env,timeout=300)
    counts={k:len(v) for k,v in support(root).items()}
    # This file contains only former TRAIN targets, never 2017; enforce raw
    # identity before reading observation/features and persist a bounded copy.
    with Path(state_features).open() as inp,(root/'physical_train_features.csv').open('x') as out:
        out.write(next(inp));previous=0
        for line in inp:
            target=int(line.partition(',')[0])
            if not previous<target<=17543:raise ValueError('Physical source extends beyond former TRAIN')
            previous=target;out.write(line)
    dispatch=HERE.parents[2]/'build-hydrobatch'/('joint_inner_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
    jobs=[]
    for family in NEURAL:
        template=json.loads((Path(pilot)/'configs'/(family+'_rate_hold_v1_s42.json')).read_text())
        for width in [32,64]:
            name=f'inner_{family}_h{width}_s42';c=dict(template,experiment_id=name,hidden_layers=str(width),train_end_x=8759,validation_end_x=17543,test_end_x=26303,
                hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=digest(manifest),hydrofusion_preissue_rain_path=str(weather),hydrofusion_preissue_rain_sha256=digest(weather),hydrofusion_preissue_rain_provenance_sha256=digest(weather.parent/'preissue_rain_provenance.json'))
            cp=dispatch/(name+'.json');bp=dispatch/(name+'.batch');write_new(cp,c);bp.write_text(f'{family} {cp.name}\n');shutil.copy2(cp,root/'configs'/cp.name);shutil.copy2(bp,root/'configs'/bp.name)
            jobs.append(dict(name=name,family=family,width=width,config=str(cp),batch=str(bp)))
    for name in ['run_hydrofusion_joint_inner_screen.py','run_hydrofusion_six_fit_screen.py','run_hydrofusion_preissue_screen.py','run_discharge_state_screen.py','run_preissue_physical_calibration.py']:
        shutil.copy2(HERE/name,root/'provenance'/name)
    paths=[p for folder in ['prepared_input','weather','executables','configs','provenance'] for p in (root/folder).rglob('*') if p.is_file()]
    paths += [root/'support.csv',root/'physical_train_features.csv']+[Path(j[k]) for j in jobs for k in ('config','batch')]+[HERE/n for n in ['run_hydrofusion_joint_inner_screen.py','run_hydrofusion_six_fit_screen.py','run_hydrofusion_preissue_screen.py','run_discharge_state_screen.py','run_preissue_physical_calibration.py']]
    plan=dict(schema='HydroFusionJointInternalSelection-v1',approval='Latest human Go ahead joint all-method improvement, standing autonomous research authorization',jobs=jobs,limits=LIMITS,manifest=str(manifest),weather=str(weather),binary=str(root/'executables/HydroBatch'),verifier=str(root/'executables/native_reload'),support_counts=counts,
        scope='2015innertrain2016innervalidation, no2017ortestobservations',neural_fits=8,physical_gain_fits=27,no_retries=True,no_automatic_expansion=True,
        goals=dict(neural_acceptable=.7,hybrid_stretch=.8,physical=.6),gates=dict(nse_gain=.02,r_squared_gain=.01,mae_ratio=1.02,high_flow_ratio=1.02,absolute_bias_increase=2),
        neural_selection='64 only if all internal validation gates beat32; otherwise retain32; freeze width and selected epoch for later separately verified final TRAIN refit',
        physical_selection='27 finite discharge-initialized shapes, gain fit internalTRAIN only with abs bias<=5%; choose max internal validation NSE among models passing all gates vs persistence; otherwise retain persistence for analysis and original dry production reference',
        physical_shapes=[list(p) for p in __import__('itertools').product([.1,.4,1.],[.005,.025,.05],[.3,.7,.9])],
        later_stage='No outer refits in this claim. Must first support fixed epoch/last checkpoint without2017selection; retain existing completed seed42fits rather than duplicate identical jobs.',
        limitations=['Internal development selection, not10yeartraining or blind generalization.','Physical initialized variant is experimental/not original PINN and has no production dispatch.','2017was already used historically; no future paper promotion from these inner scores.','Archive rainreleaseproxy and assumedzeroQlatency remain.'],pins={str(p):digest(p) for p in paths})
    write_new(root/'screen_plan.json',plan);check(plan);print('FROZEN8internalfits+27physicalgainfits',counts)

def raw_predictions(path,family,expected):
    rows={}
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            x=float(r['x'])
            if not math.isfinite(x) or not x.is_integer() or not 8759<x<=17543 or r['split']!='validation' or r['approach']!=family or int(x) in rows:raise ValueError('Invalid internal validation identity')
            y=float(r['observed']);p=float(r['predicted'])
            if not math.isfinite(y) or not math.isfinite(p) or min(y,p)<0 or int(x) not in expected or y!=expected[int(x)][0]:raise ValueError('Invalid internal value/support')
            rows[int(x)]=(y,p)
    if set(rows)!=set(expected):raise ValueError('Internal support mismatch')
    return rows

def metric(y,p,q90):
    m=scores(y,p);high=y>q90
    if not high.any():raise ValueError('No internal high-flow support')
    m['high_flow_rmse']=float(np.sqrt(np.mean((p[high]-y[high])**2)));return m

def gates(a,b):
    return dict(nse=b['nse']>=a['nse']+.02,r_squared=b['r_squared']>=a['r_squared']+.01,mae=b['mae']<=a['mae']*1.02,high_flow=b['high_flow_rmse']<=a['high_flow_rmse']*1.02,bias=abs(b['pbias_percent'])<=abs(a['pbias_percent'])+2)

def physical(root,plan,common,q90):
    values={}
    with (root/'physical_train_features.csv').open() as f:
        for row in csv.DictReader(f):
            x=int(row['target_x'])
            if not 0<x<=17543 or x in values:raise ValueError('Physical training source identity')
            values[x]=[float(row[k]) for k in ['observed','origin_q','available_effective_rain']]
    def subset(split):
        data=[]
        for x in sorted(common[split]):
            v=values[x]
            if tuple(v[:2])!=common[split][x] or not all(math.isfinite(a) and a>=0 for a in v):raise ValueError('Physical support/value mismatch')
            data.append(v)
        return np.array(data).T
    y,q,r=subset('train');vy,vq,vr=subset('validation');baseline=metric(vy,vq,q90);candidates=[]
    for i,params in enumerate(plan['physical_shapes']):
        _,base,response=predict(q,r,params,0);gain=fit_rain_gain(y,base,response)
        if gain is None:candidates.append(dict(index=i,params=params,eligible=False));continue
        prediction,_,_=predict(vq,vr,params,gain);m=metric(vy,prediction,q90);g=gates(baseline,m)
        candidates.append(dict(index=i,params=params,gain=gain,validation_metrics=m,checks=g,eligible=all(g.values())))
    eligible=[c for c in candidates if c['eligible']];selected=max(eligible,key=lambda c:(c['validation_metrics']['nse'],-c['index'])) if eligible else None
    return dict(method='discharge_initialized_physical_prototype_not_original_PINN',baseline_persistence=baseline,candidates=candidates,selected=selected)

def launch(root):
    root=Path(root).resolve(strict=True);plan=json.loads((root/'screen_plan.json').read_text());check(plan);common=support(root)
    write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_retries=True));(root/'fits').mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',MALLOC_ARENA_MAX='2');done=[]
    train_y=np.array([v[0] for _,v in sorted(common['train'].items())]);q90=float(np.quantile(train_y,.9));models={}
    try:
        for j in plan['jobs']:
            check(plan);fit=root/'fits'/j['name'];fit.mkdir();print('START '+j['name'],flush=True)
            rec=bounded_fit([plan['binary'],j['batch'],str(fit/'bundles')],fit,env,LIMITS);bundle=fit/'bundles'/j['name']
            with (bundle/'training_history.csv').open() as f:history=list(csv.DictReader(f))
            if len(history)!=50 or [int(h['epoch']) for h in history]!=list(range(1,51)):raise ValueError('Invalid internal epoch history')
            losses=[float(h['validation_loss']) for h in history]
            if not all(math.isfinite(v) for v in losses+[float(h['training_loss']) for h in history]):raise ValueError('Nonfinite losses')
            epoch=min(range(50),key=lambda i:losses[i])
            if [i for i,h in enumerate(history) if h['selected_checkpoint']=='1']!=[epoch]:raise ValueError('Epoch selection mismatch')
            with (fit/'fresh_reload.log').open('x') as log:subprocess.run([plan['verifier'],str(bundle),str(fit/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=3600,check=True)
            rows=raw_predictions(fit/'validation_predictions.csv',j['family'],common['validation']);ordered=sorted(rows);y=np.array([rows[x][0] for x in ordered]);p=np.array([rows[x][1] for x in ordered]);m=metric(y,p,q90)
            if directory_bytes(fit)>LIMITS['artifact_bytes_per_fit']:raise ValueError('Artifact cap exceeded')
            rec.update(selected_epoch=epoch+1,validation_metrics=m,sha256={str(p.relative_to(fit)):digest(p) for p in fit.rglob('*') if p.is_file()});write_new(fit/'completion.json',rec);done.append(j['name']);models[j['name']]=dict(metrics=m,selected_epoch=epoch+1);print('VERIFIED '+j['name']+' NSE='+str(m['nse']),flush=True)
        decisions={}
        for family in NEURAL:
            a,b=[models[f'inner_{family}_h{w}_s42'] for w in (32,64)];g=gates(a['metrics'],b['metrics']);width=64 if all(g.values()) else 32
            decisions[family]=dict(width=width,selected_epoch=models[f'inner_{family}_h{width}_s42']['selected_epoch'],width64_checks=g)
        ph=physical(root,plan,common,q90)
        write_new(root/'screen_completion.json',dict(status='COMPLETE_INTERNAL_SELECTION_ONLY',completed=done,plan_sha256=digest(root/'screen_plan.json'),models=models,decisions=decisions,physical=ph,training_q90=q90,test_rows=0,outer_validation_rows=0,outer_refits_launched=0,no_automatic_expansion=True))
    except BaseException as e:write_new(root/'screen_failure.json',dict(completed=done,error=str(e),no_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',nargs=6,type=Path);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if bool(a.prepare)==a.launch:p.error('Choose prepare OR launch')
    if a.prepare:prepare(a.root,*a.prepare)
    else:launch(a.root)
