#!/usr/bin/env python3
"""User-approved SIX-fit FFN screen, sequential, no retries or automatic expansion."""
import argparse
import csv
import math
import hashlib
import json
import os
import signal
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
        return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(root, binary, verifier):
    root, binary, verifier = map(lambda p: Path(p).resolve(strict=True), (root, binary, verifier))
    if (root/'screen_plan.json').exists():
        raise FileExistsError('Plan already frozen')
    native = root/'prepared_input/native_input_manifest.json'
    base = json.loads((HERE/'hydrofusion_five_method_checks/ffn_causal_forecast_direct_log_q.json').read_text())
    jobs = []
    folder = root/'configs'
    folder.mkdir(exist_ok=False)
    dispatch=HERE.parents[2]/'build-hydrobatch'/('causal_six_fit_'+hashlib.sha256(str(root).encode()).hexdigest()[:12])
    dispatch.mkdir(parents=True,exist_ok=False)
    for seed in (42, 43, 44):
        for mode in ('direct_log_q', 'log_residual'):
            name = f'ffn_{mode}_s{seed}'
            config = dict(base, experiment_id=name, epochs=50, batch_size=64, learning_rate=.001,
                          hidden_layers='32', random_seed=seed, train_end_x=17543, validation_end_x=26303,
                          test_end_x=35063, hydrofusion_synthetic_verification=False, hydrofusion_target_mode=mode,
                          hydrofusion_native_manifest_path=str(native), hydrofusion_native_manifest_sha256=digest(native))
            config_path = dispatch/(name+'.json')
            write_new(config_path, config)
            write_new(folder/config_path.name,config)
            batch = dispatch/(name+'.batch')
            batch.write_text('ffn_causal_forecast '+config_path.name+'\n')
            (folder/batch.name).write_bytes(batch.read_bytes())
            jobs.append(dict(name=name, seed=seed, mode=mode, config=str(config_path), batch=str(batch)))
    pinned = [binary, verifier, native, root/'prepared_input/intervals.csv', root/'native_policy.json',
              root/'input_prefix/manifest.json', root/'input_prefix/preseal_slice_provenance.json',
              root/'prepared_input/source_audit.json', Path(__file__).resolve(), HERE/'evaluate_causal_forecast_methods.py']
    pinned += sorted((root/'frozen_support').glob('*.csv'))
    pinned += sorted(folder.glob('*'))
    pinned += [Path(job[key]) for job in jobs for key in ('config','batch')]
    plan = dict(schema='HydroFusionSixFit-v1', approval='User explicitly approved six FFN fits in this chat',
                study_type='idealized_retrospective_causal_hindcast', binary=str(binary), verifier=str(verifier), jobs=jobs,
                limits=dict(epochs=50, wall_seconds_per_fit=28800, rss_bytes=12*1024**3, threads=2, gpu=False,
                            artifact_bytes_per_fit=250_000_000),
                advancement=dict(positive_validation_seeds=2, mean_validation_delta_nse=.02,
                                 maximum_relative_validation_mae_increase=.02),
                input_sha256={str(p):digest(p) for p in pinned},
                limitations=['Exploratory development fold reuse; no unbiased generalization claim.',
                             'Native qualifier/cadence and source-prefix limitations recorded in pinned inputs.',
                             'No operational latency proof; PET/hourly forcings explicitly lagged24h.'])
    write_new(root/'screen_plan.json', plan)
    return plan


def validate_plan(plan):
    expected={(seed, mode) for seed in (42,43,44) for mode in ('direct_log_q','log_residual')}
    jobs=plan['jobs']
    if len(jobs)!=6 or {(j['seed'],j['mode']) for j in jobs}!=expected or len({j['name'] for j in jobs})!=6:
        raise ValueError('Exactly six fixed paired fits required')
    limits=plan['limits']
    if limits!=dict(epochs=50,wall_seconds_per_fit=28800,rss_bytes=12*1024**3,threads=2,gpu=False,artifact_bytes_per_fit=250_000_000):
        raise ValueError('Approved resource limits changed')
    for name, expected_hash in plan['input_sha256'].items():
        if digest(name)!=expected_hash:
            raise ValueError('Pinned input changed: '+name)
    for job in jobs:
        c=json.loads(Path(job['config']).read_text())
        if (c['epochs'],c['hidden_layers'],c['batch_size'],c['learning_rate'],c['train_end_x'],c['validation_end_x'],c['test_end_x'],
            c['random_seed'],c['hydrofusion_target_mode'],c['hydrofusion_synthetic_verification'],c['use_csv_data'],c['use_hydro_package'])!=(50,'32',64,.001,17543,26303,35063,job['seed'],job['mode'],False,False,False):
            raise ValueError('Prospective config changed')


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def bounded_fit(command, output, env, limits):
    start=time.monotonic();peak=0
    with (output/'training.log').open('x') as log:
        def hard_limits():
            # Hard virtual-memory cap also bounds resident memory; stricter than
            # the sampled RSS watchdog, and independent of polling intervals.
            resource.setrlimit(resource.RLIMIT_AS,(limits['rss_bytes'],limits['rss_bytes']))
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            allowed=sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0,set(allowed[:limits['threads']]))
            cpu=limits['wall_seconds_per_fit']*limits['threads']
            resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu))
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=hard_limits)
        try:
            while child.poll() is None:
                try:
                    for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                        if line.startswith('VmRSS:'):
                            peak=max(peak,int(line.split()[1])*1024)
                except FileNotFoundError:
                    pass
                if time.monotonic()-start>limits['wall_seconds_per_fit'] or peak>limits['rss_bytes'] or directory_bytes(output)>limits['artifact_bytes_per_fit']:
                    raise RuntimeError('Approved fit resource ceiling exceeded')
                time.sleep(2)
            if child.returncode:
                raise RuntimeError('Training failed; no retry or remaining jobs launched')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    return dict(wall_seconds=time.monotonic()-start,peak_sampled_rss_bytes=peak,exit_code=child.returncode)


def launch(root):
    root=Path(root).resolve(strict=True)
    plan=json.loads((root/'screen_plan.json').read_text())
    validate_plan(plan)
    # Never automatically resume or repeat a fit, including after interruption.
    write_new(root/'launch_claim.json',dict(plan_sha256=digest(root/'screen_plan.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='')
    output=root/'fits';output.mkdir(exist_ok=False)
    completed=[]
    try:
        for job in plan['jobs']:
            validate_plan(plan)
            fit=output/job['name'];fit.mkdir(exist_ok=False)
            print('START '+job['name'],flush=True)
            resource=bounded_fit([plan['binary'],job['batch'],str(fit/'bundles')],fit,env,plan['limits'])
            bundle=fit/'bundles'/job['name']
            with (bundle/'training_history.csv').open(newline='') as stream:
                history=list(csv.DictReader(stream))
            if len(history)!=50 or [int(row['epoch']) for row in history]!=list(range(1,51)):
                raise RuntimeError('Expected exactly50 training epochs')
            loss=[float(row['validation_loss']) for row in history]
            if not all(math.isfinite(v) for v in loss) or not all(math.isfinite(float(row['training_loss'])) for row in history):
                raise RuntimeError('Nonfinite training/validation loss')
            selected=[i for i,row in enumerate(history) if row['selected_checkpoint']=='1']
            if selected!=[min(range(50),key=lambda i:loss[i])]:
                raise RuntimeError('Checkpoint was not earliest validation minimum')
            resource['selected_epoch']=selected[0]+1
            command=[plan['verifier'],str(bundle),str(fit/'validation_predictions.csv')]
            with (fit/'fresh_reload.log').open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
            resource['retained_artifact_bytes']=directory_bytes(fit)
            if resource['retained_artifact_bytes']>plan['limits']['artifact_bytes_per_fit']:
                raise RuntimeError('Artifact ceiling exceeded')
            resource['file_sha256']={str(p.relative_to(fit)):digest(p) for p in fit.rglob('*') if p.is_file()}
            write_new(fit/'completion.json',resource)
            completed.append(job['name'])
            print('VERIFIED '+job['name'],flush=True)
        write_new(root/'screen_completion.json',dict(completed=completed,plan_sha256=digest(root/'screen_plan.json')))
    except BaseException as exc:
        write_new(root/'screen_failure.json',dict(completed=completed,error=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',nargs=2,metavar=('BINARY','VERIFIER'))
    action.add_argument('--launch-approved-six-fits',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.root,*args.prepare)
    else:launch(args.root)
