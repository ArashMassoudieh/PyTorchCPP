#!/usr/bin/env python3
"""Hash-checked matched-support diagnostics; predictions only, no fitting/tuning."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,pstdev
import evaluate_causal_forecast_methods as e


def sha(path):
    with Path(path).open('rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
        return h.hexdigest()


def verified_jobs(root):
    root=Path(root).resolve(strict=True);plan=json.loads((root/'screen_plan.json').read_text());done=json.loads((root/'screen_completion.json').read_text())
    if done['plan_sha256']!=sha(root/'screen_plan.json') or done['completed']!=[j['name'] for j in plan['jobs']]:raise ValueError('Incomplete or changed campaign')
    for path,h in plan['input_sha256'].items():
        if sha(path)!=h:raise ValueError('Changed prospective input: '+path)
    output=[]
    for j in plan['jobs']:
        fit=root/'fits'/j['name'];record=json.loads((fit/'completion.json').read_text())
        for path,h in record['file_sha256'].items():
            if sha(fit/path)!=h:raise ValueError('Changed completed artifact: '+path)
        output.append(dict(j,root=root,fit=fit,bundle=fit/'bundles'/j['name'],record=record,
                           family=j.get('family','ffn_causal_forecast')))
    return plan,output


def quantile(values,p):
    values=sorted(values);position=(len(values)-1)*p;i=int(position);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(position-i)


def kge_components(observed,predicted):
    if not observed or len(observed)!=len(predicted):raise ValueError('Nonempty matched series required')
    if not all(math.isfinite(v) and v>=0 for v in observed+predicted):raise ValueError('Finite nonnegative series required')
    om,pm=mean(observed),mean(predicted);os,ps=pstdev(observed),pstdev(predicted)
    r=sum((a-om)*(b-pm) for a,b in zip(observed,predicted))/(len(observed)*os*ps) if os and ps else None
    alpha=ps/os if os else None;beta=pm/om if om else None
    terms=dict(correlation=(r-1)**2 if r is not None else None,variability=(alpha-1)**2 if alpha is not None else None,mean=(beta-1)**2 if beta is not None else None)
    kge=1-math.sqrt(sum(terms.values())) if all(v is not None for v in terms.values()) else None
    return dict(correlation=r,variability_ratio=alpha,mean_ratio=beta,squared_kge_penalties=terms,kge=kge)


def diagnostic(values,support,thresholds):
    obs=[values[x][0] for x in support];pred=[values[x][1] for x in support]
    out=dict(metrics=e.metrics(obs,pred,thresholds['q90']),kge_components=kge_components(obs,pred),regimes={})
    for label,indices in {
        'below_training_q50':[i for i,o in enumerate(obs) if o<thresholds['q50']],
        'training_q50_to_q90':[i for i,o in enumerate(obs) if thresholds['q50']<=o<thresholds['q90']],
        'at_or_above_training_q90':[i for i,o in enumerate(obs) if o>=thresholds['q90']]}.items():
        out['regimes'][label]=e.metrics([obs[i] for i in indices],[pred[i] for i in indices],thresholds['q90'])
    out['flow_duration_quantiles_mm_h']={str(p):dict(observed=quantile(obs,p),predicted=quantile(pred,p)) for p in (.1,.5,.9,.95,.99)}
    return out


def analyze(source,output,extension=None):
    output=Path(output)
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError(output)
    source=Path(source).resolve(strict=True);plan,jobs=verified_jobs(source);plans=[dict(path=str(source/'screen_plan.json'),sha256=sha(source/'screen_plan.json'))]
    if extension:
        extension=Path(extension).resolve(strict=True);ext_plan,extra=verified_jobs(extension)
        if ext_plan['source_plan_sha256']!=sha(source/'screen_plan.json'):raise ValueError('Extension from different source campaign')
        for p in (source/'frozen_support').glob('*.csv'):
            if sha(p)!=sha(extension/'frozen_support'/p.name):raise ValueError('Support changed between stages')
        for name in ('intervals.csv','native_input_manifest.json'):
            if sha(source/'prepared_input'/name)!=sha(extension/'prepared_input'/name):raise ValueError('Native source changed')
        jobs+=extra;plans.append(dict(path=str(extension/'screen_plan.json'),sha256=sha(extension/'screen_plan.json')))
    train=e.csv_rows(source/'frozen_support/train_observations.csv',['x','observed'])
    times=[e.timestamp(r['x']) for r in train]
    if times!=sorted(set(times)) or any(x>17543 for x in times):raise ValueError('Training-only threshold support required')
    thresholds={f'q{p}':quantile([e.finite_nonnegative(r['observed']) for r in train],p/100) for p in (10,50,90)}
    scalers=[];qrefs=[]
    for j in jobs:
        if j['family']=='pinn_causal_forecast':continue
        with (j['bundle']/'scalers.csv').open(newline='') as f:
            scalers.append([{k:v for k,v in row.items() if k!='approach'} for row in csv.DictReader(f) if row['kind']=='input'])
        qrefs.append([r for r in (j['bundle']/'hydrofusion_metadata.txt').read_text().splitlines() if r.startswith('q_ref=')])
    if not scalers[0] or any(s!=scalers[0] for s in scalers) or not qrefs[0] or any(q!=qrefs[0] for q in qrefs):raise ValueError('Neural scaler/q_ref mismatch')
    result=dict(schema='HydroFusionMatchedDiagnostics-v1',plans=plans,script_sha256=sha(__file__),training_thresholds_mm_h=thresholds,
                catchment_area_m2=json.loads((source/'prepared_input/native_input_manifest.json').read_text())['catchment_area_m2'],
                splits={},resources={j['name']:{k:v for k,v in j['record'].items() if k!='file_sha256'} for j in jobs},
                limitations=['Exploratory reused 2018 development fold; no new blind generalization claim.',
                    'Idealized retrospective availability, not verified operational publication latency.',
                    'Fixed uncalibrated physical recipe; no parameter search or observed-state initialization.',
                    'FFN705 and LSTM6945 parameters: architecture and capacity differ.',
                    'No event windows prospectively frozen. No event-specific claims.',
                    'Regime diagnostics are descriptive, not permission for test-selected corrections.',
                    'Only physical router ledger conserves water; neural corrections do not inherit closure.',
                    'No automatic manuscript promotion, tuning or additional training.'])
    lines=['# Matched causal forecast comparison','', 'All scores use identical declared targets. Negative PBIAS means underprediction.','']
    for split in ('validation','test'):
        support=e.load_support(source/f'frozen_support/{split}_support.csv')
        methods={'persistence':e.load_predictions(source/f'frozen_support/{split}_persistence.csv','persistence',split)}
        groups={}
        for j in jobs:
            path=j['fit']/'validation_predictions.csv' if split=='validation' else j['bundle']/'predictions.csv'
            values=e.load_predictions(path,j['family'],split)
            if set(values)!=set(support):raise ValueError('Prediction/support mismatch')
            methods[j['name']]=values
            groups.setdefault(j['family']+' / '+j['mode'],[]).append(j['name'])
        e.evaluate(methods,support,thresholds['q90'],[]) # independently verifies observed identity across all arms
        detailed={name:diagnostic(values,support,thresholds) for name,values in methods.items()}
        groups['persistence']=['persistence']
        summaries={g:{metric:(mean(detailed[name]['metrics'][metric] for name in names) if all(detailed[name]['metrics'][metric] is not None for name in names) else None) for metric in ('nse','kge','rmse','mae','pbias_percent','high_flow_rmse')} for g,names in groups.items()}
        paired=[]
        for seed in (42,43,44):
            if extension:
                paired += [dict(name='lstm_residual_minus_direct',seed=seed,control=f'lstm_hydrofusion_direct_log_q_s{seed}',candidate=f'lstm_hydrofusion_log_residual_s{seed}'),
                    dict(name='ffn_physical_minus_direct',seed=seed,control=f'ffn_direct_log_q_s{seed}',candidate=f'ffn_physical_forecast_log_residual_s{seed}'),
                    dict(name='ffn_physical_minus_persistence_residual',seed=seed,control=f'ffn_log_residual_s{seed}',candidate=f'ffn_physical_forecast_log_residual_s{seed}'),
                    dict(name='lstm_physical_minus_direct',seed=seed,control=f'lstm_hydrofusion_direct_log_q_s{seed}',candidate=f'lstm_physical_forecast_log_residual_s{seed}'),
                    dict(name='lstm_physical_minus_persistence_residual',seed=seed,control=f'lstm_hydrofusion_log_residual_s{seed}',candidate=f'lstm_physical_forecast_log_residual_s{seed}')]
            paired.append(dict(name='ffn_residual_minus_direct',seed=seed,control=f'ffn_direct_log_q_s{seed}',candidate=f'ffn_log_residual_s{seed}'))
        paired_report=e.paired_seed_summary({name:dict(pooled=d['metrics']) for name,d in detailed.items()},paired)
        result['splits'][split]=dict(n=len(support),support_sha256=sha(source/f'frozen_support/{split}_support.csv'),methods=detailed,means=summaries,paired=paired_report)
        lines += [f'## {split.title()}', '', '| Method / target | NSE | KGE | RMSE m³/s | MAE m³/s | PBIAS % |','|---|---:|---:|---:|---:|---:|']
        factor=result['catchment_area_m2']/3600000
        def fmt(v,factor=1):return f'{v*factor:.6f}' if v is not None else 'undefined'
        for g,m in summaries.items():lines.append(f"| {g} | {fmt(m['nse'])} | {fmt(m['kge'])} | {fmt(m['rmse'],factor)} | {fmt(m['mae'],factor)} | {fmt(m['pbias_percent'])} |")
        lines += ['', 'Seed means for neural groups; physical and persistence are single deterministic results.','']
    lines+=result['limitations']
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with output.with_suffix('.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[:16]));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--extension',type=Path)
    a=p.parse_args();analyze(a.source,a.output,a.extension)
