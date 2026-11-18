#!/usr/bin/env python3
"""Actual synthetic CLI→managed bundle→fresh inference→common evaluation.

Only fabricated configs in this committed fixture directory are launched.
Does not build against live sources or enable a real-data source.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('batch_binary',type=Path)
    parser.add_argument('extended_reload_binary',type=Path)
    parser.add_argument('ffn_reload_binary',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--baseline-binary',type=Path)
    args=parser.parse_args()
    inputs=[args.batch_binary,args.extended_reload_binary,args.ffn_reload_binary]
    if args.baseline_binary:inputs.append(args.baseline_binary)
    identities={str(p.resolve(strict=True)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if args.output.exists():parser.error('Refusing existing output.')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'binary_hashes.json').write_text(json.dumps(identities,indent=2)+'\n')
    root=Path(__file__).resolve().parent;gist=root.parent
    fixture_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(root.glob('*.json'))+list(root.glob('*.batch'))}
    (args.output/'fixture_hashes.json').write_text(json.dumps(fixture_hashes,indent=2)+'\n')
    import os
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    logs=[]
    def run(name, command, expected=0):
        completed=subprocess.run(list(map(str,command)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
        (args.output/(name+'.log')).write_text(completed.stdout)
        logs.append({'name':name,'command':list(map(str,command)),'exit_code':completed.returncode})
        (args.output/'commands.json').write_text(json.dumps(logs,indent=2)+'\n')
        if completed.returncode!=expected:raise RuntimeError(name+' failed; see saved log')
        return completed.stdout
    run('actual_cli',[args.batch_binary.resolve(),root/'jobs.batch',args.output/'bundles'])
    specs=[]
    for line in (root/'jobs.batch').read_text().splitlines():
        approach,config=line.split();settings=json.loads((root/config).read_text())
        directory=args.output/'bundles'/settings['experiment_id']
        if approach=='ffn_hydrofusion':
            run(approach,[args.ffn_reload_binary.resolve(),directory,directory/'hydrofusion_metadata.txt',directory/'hydrofusion_origin_target_map.csv',approach])
        else:run(settings['experiment_id'],[args.extended_reload_binary.resolve(),directory,approach])
        specs+=['--method',settings['experiment_id']+'='+approach+'='+str(directory/'predictions.csv')]
        if approach!='ffn_hydrofusion':
            run(settings['experiment_id']+'_integrity',[sys.executable,root/'check_integrity.py',args.extended_reload_binary.resolve(),directory,approach,args.output/(settings['experiment_id']+'_tamper')])
    run('repeat_seed42',[args.batch_binary.resolve(),root/'repeat_seed42.batch',args.output/'repeat_seed42'])
    run('different_seed43',[args.batch_binary.resolve(),root/'different_seed43.batch',args.output/'different_seed43'])
    for mode in ['ffn_causal_forecast','lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast']:
        original=args.output/'bundles'/(mode+'_log_residual')/'predictions.csv'
        repeat=args.output/'repeat_seed42'/(mode+'_log_residual')/'predictions.csv'
        changed=args.output/'different_seed43'/(mode+'_seed43')/'predictions.csv'
        if original.read_bytes()!=repeat.read_bytes():raise RuntimeError('Same seed is not reproducible: '+mode)
        first=list(csv.DictReader(original.open()));second=list(csv.DictReader(changed.open()))
        if [row['predicted'] for row in first]==[row['predicted'] for row in second]:raise RuntimeError('Different seed had no effect: '+mode)
        history=list(csv.DictReader((original.parent/'training_history.csv').open()))
        if len(history)<2 or all(float(row['training_loss'])==float(history[0]['training_loss']) for row in history[1:]):
            raise RuntimeError('No training-loss response across gradient updates: '+mode)
        selected=[row for row in history if row['selected_checkpoint']=='1']
        if len(selected)!=1 or float(selected[0]['validation_loss'])!=min(float(row['validation_loss']) for row in history):
            raise RuntimeError('Selected checkpoint is not validation minimum: '+mode)
    physical_history=list(csv.DictReader((args.output/'bundles'/'pinn_causal_forecast_log_residual'/'training_history.csv').open()))
    if physical_history:raise RuntimeError('Physical-only baseline acquired fake training history.')
    # These test identities are frozen by cutoff 150/220/280, 72-hour feature
    # history and 24 eligible origin steps, NOT selected by observed skill.
    support=list(range(245,281))
    with (args.output/'support.csv').open('w') as f:
        f.write('x\n');f.writelines(str(x)+'\n' for x in support)
    def q(x):
        phase=x%60
        return 0.5+0.002*x+(2*math.exp(-phase/4) if phase<10 else 0)
    with (args.output/'training.csv').open('w') as f:
        f.write('x,observed\n');f.writelines(str(x)+','+repr(q(x))+'\n' for x in range(95,151))
    with (args.output/'persistence.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['approach','index','split','x','observed','predicted','residual'])
        for i,x in enumerate(support):writer.writerow(['persistence',i,'test',x,q(x),q(x-1),q(x-1)-q(x)])
    specs+=['--method','free_persistence=persistence='+str(args.output/'persistence.csv')]
    events=[{'name':'fabricated_complete_window','start_x':250,'end_x':260}]
    (args.output/'events.json').write_text(json.dumps(events)+'\n')
    run('common_evaluation',[sys.executable,gist/'evaluate_causal_forecast_methods.py',*specs,'--support',args.output/'support.csv',
        '--training-observations',args.output/'training.csv','--train-end','150','--events',args.output/'events.json','--output',args.output/'evaluation.json'])
    # Explicit source guards for every new mode, through the real CLI.
    for mode in ['ffn_causal_forecast','lstm_hydrofusion','pinn_causal_forecast','ffn_physical_forecast','lstm_physical_forecast']:
        for key in ['use_hydro_package','use_csv_data']:
            name='guard_'+mode+'_'+key
            guard=run(name,[args.batch_binary.resolve(),root/(name+'.batch'),args.output/(name+'_output')],expected=1)
            if 'reject package/CSV training' not in guard:raise RuntimeError('Wrong failure reason for source guard')
    for mode in ['ffn_causal_forecast','lstm_hydrofusion','pinn_causal_forecast','ffn_physical_forecast','lstm_physical_forecast']:
        for case in ['no_consent','extra_objective']:
            name='reject_'+mode+'_'+case
            guard=run(name,[args.batch_binary.resolve(),root/(name+'.batch'),args.output/(name+'_output')],expected=1)
            expected='explicit synthetic verification' if case=='no_consent' else 'does not support extra objectives'
            if expected not in guard:raise RuntimeError('Wrong recipe rejection reason: '+name)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in fixture_hashes.items()):
        raise RuntimeError('Fixture changed during verification.')
    if args.baseline_binary:
        run('unchanged_original_five',['bash',gist/'hydrofusion_task4_cli_checks/compare_regression_against_baseline_build.sh',
            args.batch_binary.resolve(),args.baseline_binary.resolve(),args.output/'original_five_regression'])
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in identities.items()):
        raise RuntimeError('Input executable changed during verification.')
    print('All five-method synthetic CLI/reload/support/source-guard checks passed; no real research run.')


if __name__=='__main__':main()
