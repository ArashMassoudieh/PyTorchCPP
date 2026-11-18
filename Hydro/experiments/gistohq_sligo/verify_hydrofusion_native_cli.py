#!/usr/bin/env python3
"""Fabricated producer-package acceptance only; never opens research inputs."""
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
from pathlib import Path
from test_hydrofusion_prepared_package import fabricated_package
from prepare_hydrofusion_causal_package import prepare

HERE=Path(__file__).resolve().parent
NAMES=['ffn_causal_forecast_direct_log_q','ffn_causal_forecast_log_residual',
       'lstm_hydrofusion_direct_log_q','lstm_hydrofusion_log_residual','pinn_causal_forecast_log_residual',
       'ffn_physical_forecast_log_residual','lstm_physical_forecast_log_residual']


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(binary,verifier,bulk_verifier,output):
    output=Path(output).absolute();output.mkdir(exist_ok=False)
    source,policy=fabricated_package(output);prepare(source,policy,output/'prepared')
    manifest=output/'prepared/native_input_manifest.json'
    dispatch=HERE.parents[2]/'build-hydrobatch'/('native_check_'+hashlib.sha256(str(output).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='')
    def limits():resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    commands=[]
    def command(name,argv,expected=0,reason=None):
        with (output/(name+'.log')).open('x') as log:
            p=subprocess.run(list(map(str,argv)),stdout=log,stderr=subprocess.STDOUT,env=env,timeout=120,preexec_fn=limits)
        text=(output/(name+'.log')).read_text()
        if p.returncode!=expected or (reason and reason not in text):raise RuntimeError(name+': unexpected result '+text[-2000:])
        commands.append(dict(name=name,command=list(map(str,argv)),exit_code=p.returncode))
    configs={};batch=[]
    for name in NAMES:
        c=json.loads((HERE/'hydrofusion_five_method_checks'/(name+'.json')).read_text())
        c.update(hydrofusion_synthetic_verification=False,hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=digest(manifest))
        p=dispatch/(name+'.json');p.write_text(json.dumps(c,indent=2)+'\n');configs[name]=c
        mode=name.removesuffix('_direct_log_q') if name.endswith('_direct_log_q') else name.removesuffix('_log_residual')
        batch.append(mode+' '+p.name)
    job=dispatch/'positive.batch';job.write_text('\n'.join(batch)+'\n')
    command('positive_cli',[binary,job,output/'bundles'])
    for name in NAMES:
        mode=name.removesuffix('_direct_log_q') if name.endswith('_direct_log_q') else name.removesuffix('_log_residual')
        command(name+'_reload',[verifier,output/'bundles'/name,mode])
    for name in NAMES[:2]:
        command(name+'_bulk',[bulk_verifier,output/'bundles'/name,output/(name+'_validation.csv')])
    changes=[('csv_tamper',None,'Native CSV hash/path mismatch'),('start_label',('time_label','interval_start'),'Unsupported native package semantics'),
             ('pet_negative_delay',('daily_pet_delay_hours',-1),'Invalid native manifest integer'),
             ('q_latency',('discharge_delay_hours',1),'Explicit bounded causal availability policy'),
             ('unsafe_path',('interval_csv','../outside.csv'),'Native CSV must be local'),
             ('row_count',('row_count',299),'Native row count mismatch')]
    for name,change,reason in changes:
        folder=output/name;shutil.copytree(output/'prepared',folder)
        p=folder/'native_input_manifest.json'
        if change:
            m=json.loads(p.read_text());m[change[0]]=change[1];p.write_text(json.dumps(m))
        else:
            with (folder/'intervals.csv').open('a') as f:f.write('corruption\n')
        c=dict(configs[NAMES[0]],experiment_id=name,hydrofusion_native_manifest_path=str(p),hydrofusion_native_manifest_sha256=digest(p))
        cfg=dispatch/(name+'.json');cfg.write_text(json.dumps(c));job=dispatch/(name+'.batch');job.write_text('ffn_causal_forecast '+cfg.name+'\n')
        command(name,[binary,job,output/(name+'_output')],1,reason)
    for name,key,value in [('ambiguous_source','hydrofusion_synthetic_verification',True),('missing_pin','hydrofusion_native_manifest_sha256','')]:
        c=dict(configs[NAMES[0]],experiment_id=name);c[key]=value
        cfg=dispatch/(name+'.json');cfg.write_text(json.dumps(c));job=dispatch/(name+'.batch');job.write_text('ffn_causal_forecast '+cfg.name+'\n')
        command(name,[binary,job,output/(name+'_output')],1,'pinned prepared native manifest')
    report=dict(binary_sha256=digest(binary),verifier_sha256=digest(verifier),bulk_verifier_sha256=digest(bulk_verifier),
                prepared_manifest_sha256=digest(manifest),commands=commands,synthetic_only=True,
                source_sha256={str(p):digest(p) for p in [Path(__file__),HERE/'test_hydrofusion_prepared_package.py',HERE/'prepare_hydrofusion_causal_package.py']})
    (output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS:7 fabricated-package modes + fresh reloads;2 all-row FFN reloads;8 CLI rejections;hard12GB address-space cap.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','verifier','bulk_verifier','output'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.binary.resolve(),a.verifier.resolve(),a.bulk_verifier.resolve(),a.output)
