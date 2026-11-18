#!/usr/bin/env python3
"""Close the predeclared six-fit screen; no training or model selection by test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import evaluate_causal_forecast_methods as evaluator


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root,output):
    root=Path(root).resolve(strict=True);output=Path(output)
    if output.exists():raise FileExistsError(output)
    plan=json.loads((root/'screen_plan.json').read_text())
    completion=json.loads((root/'screen_completion.json').read_text())
    if completion['completed']!=[j['name'] for j in plan['jobs']] or completion['plan_sha256']!=sha(root/'screen_plan.json'):
        raise ValueError('Incomplete or changed campaign')
    for name,digest in plan['input_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed pinned input: '+name)
    train_end=17543
    q90=evaluator.training_threshold(root/'frozen_support/train_observations.csv',train_end)
    reports={};pairs=[dict(name='residual_minus_direct',seed=seed,control=f'ffn_direct_log_q_s{seed}',candidate=f'ffn_log_residual_s{seed}') for seed in (42,43,44)]
    scalers=[];qrefs=[];resources={}
    for job in plan['jobs']:
        fit=root/'fits'/job['name'];record=json.loads((fit/'completion.json').read_text())
        for name,digest in record['file_sha256'].items():
            if sha(fit/name)!=digest:raise ValueError('Changed completed artifact: '+name)
        resources[job['name']]={k:record[k] for k in ('wall_seconds','peak_sampled_rss_bytes','retained_artifact_bytes','selected_epoch')}
        bundle=fit/'bundles'/job['name']
        with (bundle/'scalers.csv').open(newline='') as stream:
            scalers.append([row for row in csv.DictReader(stream) if row['kind']=='input'])
        metadata=(bundle/'hydrofusion_metadata.txt').read_text().splitlines()
        qrefs.append([line for line in metadata if line.startswith('q_ref=')])
    if not scalers[0] or any(s!=scalers[0] for s in scalers) or not qrefs[0] or any(q!=qrefs[0] for q in qrefs):
        raise ValueError('Arms/seeds do not share input scaling and q_ref')
    for split in ('validation','test'):
        support=evaluator.load_support(root/f'frozen_support/{split}_support.csv')
        methods={'persistence':evaluator.load_predictions(root/f'frozen_support/{split}_persistence.csv','persistence',split)}
        for job in plan['jobs']:
            fit=root/'fits'/job['name']
            path=fit/'validation_predictions.csv' if split=='validation' else fit/'bundles'/job['name']/'predictions.csv'
            values=evaluator.load_predictions(path,'ffn_causal_forecast',split)
            if set(values)!=set(support):raise ValueError('Extra/missing prospective support')
            methods[job['name']]=values
        evaluated=evaluator.evaluate(methods,support,q90,[])
        reports[split]={'n':len(support),'support_sha256':sha(root/f'frozen_support/{split}_support.csv'),
                        'methods':evaluated,'paired':evaluator.paired_seed_summary(evaluated,pairs)}
    validation=reports['validation']['methods'];test=reports['test']['methods']
    def average(split,mode,key):return mean(split[f'ffn_{mode}_s{seed}']['pooled'][key] for seed in (42,43,44))
    delta=reports['validation']['paired']['residual_minus_direct']
    mae_ratio=average(validation,'log_residual','mae')/average(validation,'direct_log_q','mae')-1
    gate=plan['advancement']
    checks={'positive_validation_seeds':delta['positive_seeds']>=gate['positive_validation_seeds'],
            'mean_validation_delta_nse':delta['mean_delta_nse']>=gate['mean_validation_delta_nse'],
            'validation_mae_increase':mae_ratio<=gate['maximum_relative_validation_mae_increase']}
    native=json.loads((root/'prepared_input/native_input_manifest.json').read_text())
    area=float(native['catchment_area_m2'])
    report={'study_type':plan['study_type'],'catchment_area_m2':area,'m3_s_per_mm_h':area/3600000,
            'units':'NSE/KGE dimensionless; RMSE/MAE raw mm/h; PBIAS percent. Convert discharge with the declared dataset area.',
            'plan_sha256':sha(root/'screen_plan.json'),
            'analysis_script_sha256':sha(__file__),'q90_training_only_mm_h':q90,
            'input_scalers_identical_all_arms_seeds':True,'training_q_ref_identical_all_arms_seeds':True,
            'resources':resources,'splits':reports,'advancement_checks':checks,'advancement_passed':all(checks.values()),
            'relative_validation_mae_increase':mae_ratio,
            'limitations':['Only six fits, one reused development fold; exploratory, no generalization claim.',
                           'Idealized retrospective availability, not operational delivery-time verification.',
                           'Native coverage/qualifier/source-prefix caveats retained in pinned inputs.',
                           'Scores apply to discharge-informed one-hour forecasts; earlier forcing-only/same-hour results use a different task.',
                           'Only FFN has been research-trained in this screen; all-five adapter tests are not all-five research results.',
                           'No event windows were prospectively frozen for this screen; event diagnostics not claimed.',
                           'Passing advancement does not authorize another training stage or manuscript promotion.']}
    with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    lines=['# Approved six-fit FFN screen\n',f"Predeclared validation advancement: {'PASS' if report['advancement_passed'] else 'FAIL'}. No automatic additional fits.\n",
           '| Split | Direct mean NSE | Residual mean NSE | Persistence NSE | Paired mean Δ NSE |',
           '|---|---:|---:|---:|---:|']
    for label,split in [('validation',validation),('test',test)]:
        paired=reports[label]['paired']['residual_minus_direct']
        lines.append(f"| {label} | {average(split,'direct_log_q','nse'):.6f} | {average(split,'log_residual','nse'):.6f} | {split['persistence']['pooled']['nse']:.6f} | {paired['mean_delta_nse']:+.6f} |")
    lines += ['',f"Validation paired positive seeds: {delta['positive_seeds']}/3. Residual/direct mean validation MAE change: {mae_ratio:+.2%}.",
              '', 'All arms share the exact frozen targets, input scalers and training-only q_ref. Each saved checkpoint was reloaded in a fresh process against all test predictions, with additional target-free production API checks.',
              '',*report['limitations']]
    md=output.with_suffix('.md')
    with md.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[:10]))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();analyze(a.root,a.output)
