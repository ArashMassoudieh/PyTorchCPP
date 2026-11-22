#!/usr/bin/env python3
"""Synthetic-only v4 CLI, archive, history, scope and corruption regression checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from test_corrected_gauge_package import inputs
from prepare_corrected_gauge_package import prepare
from prepare_hydrofusion_search_slice import prepare as crop
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(binary, verifier, native_verifier, output):
    binary,verifier,native_verifier,output=map(lambda p:Path(p).absolute(),(binary,verifier,native_verifier,output))
    output.mkdir(parents=True,exist_ok=False)
    dispatch=HERE.parents[2]/'build-hydrobatch'/('context_checks_'+hashlib.sha256(str(output).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    receipt=dict(binary_sha256=sha(binary),verifier_sha256=sha(verifier),native_verifier_sha256=sha(native_verifier),synthetic_only=True,commands=[])
    def command(name,args,expected=0,reason=None):
        with (output/(name+'.log')).open('x') as log:
            p=subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600)
        text=(output/(name+'.log')).read_text()
        if p.returncode!=expected or reason and reason not in text:raise RuntimeError(name+': '+text[-2000:])
        receipt['commands'].append(dict(name=name,args=list(map(str,args)),exit_code=p.returncode))
    def config(name,family,recipe,dev=False,**changes):
        c=json.loads((HERE/'hydrofusion_five_method_checks'/(family+'_log_residual.json')).read_text())
        c.update(experiment_id=name,epochs=3,physics_profile='causal_valid_time_dry_v3',hydrofusion_objective='log_flow_mse_v2',
                 hydrofusion_context_recipe=recipe,hydrofusion_history_hours='0,1,3,6,12,23' if family.startswith('ffn') else '0',
                 hydrofusion_development_only=dev)
        c.update(changes);return c
    def fit(name,family,c,expected=0,reason=None):
        path=dispatch/(name+'.json');path.write_text(json.dumps(c,indent=2)+'\n')
        batch=dispatch/(name+'.batch');batch.write_text(f'{family} {path.name}\n')
        command(name,[binary,batch,output/name],expected,reason)
        return output/name/name
    family='ffn_physical_forecast'
    good={}
    for f in ['ffn_physical_forecast','lstm_physical_forecast']:
        for recipe in ['zero25','absolute25','anchored25']:
            for dev in [False,True]:
                name=f+'_'+recipe+('_dev' if dev else '_all')
                bundle=fit(name,f,config(name,f,recipe,dev));good[name]=bundle
                command(name+'_reload',[verifier,bundle,f])
                c=json.loads((bundle/'experiment_config.json').read_text())
                if c['hydrofusion_context_recipe']!=recipe or c['hydrofusion_development_only']!=dev:raise ValueError('Exported recipe/scope mismatch')
    name='lstm_longer_48_dev';bundle=fit(name,'lstm_physical_forecast',config(name,'lstm_physical_forecast','anchored25',True,lstm_sequence_length=48))
    command(name+'_reload',[verifier,bundle,'lstm_physical_forecast'])
    # Real prepared-native adapter exercised on fabricated values only.
    native,weather=inputs(output)
    prepare(native,weather,output/'fabricated_native',producer_catalog=weather/'station_catalog.json',original_export_manifest=weather/'station_source_manifest.json',station_metadata=weather/'station.rdb')
    manifest=output/'fabricated_native/native_input_manifest.json'
    cropped=crop(manifest,output/'fabricated_development',220)
    name='native_context_dev';c=config(name,family,'anchored25',True,hydrofusion_synthetic_verification=False,hydrofusion_native_manifest_path=str(cropped),hydrofusion_native_manifest_sha256=sha(cropped))
    bundle=fit(name,family,c);command(name+'_reload',[native_verifier,bundle,output/'native_validation.csv'])
    name='uncropped_dev_rejected';c=config(name,family,'anchored25',True,hydrofusion_synthetic_verification=False,hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=sha(manifest))
    fit(name,family,c,1,'exceeds authorized target cutoff before observation parsing')
    for name,changes,reason in [
        ('unknown_context',dict(hydrofusion_context_recipe='unknown'),'Context recipes require'),
        ('wrong_context_type',dict(hydrofusion_context_recipe=25),'missing string'),
        ('wrong_scope_type',dict(hydrofusion_development_only='true'),'missing boolean'),
        ('wrong_history_type',dict(hydrofusion_history_hours=[0,1]),'missing string'),
        ('duplicate_history',dict(hydrofusion_history_hours='0,1,1'),'History offsets'),
        ('future_history',dict(hydrofusion_history_hours='0,-1'),'unsigned integer'),
        ('out_of_window',dict(hydrofusion_history_hours='0,24'),'History offsets'),
        ('legacy_lag_knob',dict(use_time_lagged_ffn=True),'Legacy FFN lag settings'),
        ('wrong_profile',dict(physics_profile='causal_two_reservoir_v1'),'Context recipes require')]:
        c=config(name,family,'anchored25',True,**changes);fit(name,family,c,1,reason)
    def rehash(bundle):
        path=bundle/'artifact_manifest.csv';lines=path.read_text().splitlines();out=lines[:2]
        for line in lines[2:]:
            p,_,_=line.split(',');f=bundle/p;out.append(f'{p},{f.stat().st_size},{sha(f)}')
        path.write_text('\n'.join(out)+'\n')
    source=good['ffn_physical_forecast_anchored25_dev']
    corrupt=output/'corrupt_schema';shutil.copytree(source,corrupt)
    path=corrupt/'hydrofusion_context_schema.txt';path.write_text(path.read_text().replace('physical_fast_storage','wrong_storage'))
    rehash(corrupt);command('corrupt_schema_reload',[verifier,corrupt,family],1,'Context feature schema/config mismatch')
    corrupt=output/'corrupt_recipe';shutil.copytree(source,corrupt)
    path=corrupt/'experiment_config.json';c=json.loads(path.read_text());c['hydrofusion_context_recipe']='absolute25';path.write_text(json.dumps(c,indent=2)+'\n')
    rehash(corrupt);command('corrupt_recipe_reload',[verifier,corrupt,family],1,'Context feature schema/config mismatch')
    receipt['status']='VERIFIED'
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('VERIFIED context CLI commands='+str(len(receipt['commands'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['binary','verifier','native_verifier','new_output']:p.add_argument(name,type=Path)
    a=p.parse_args();run(a.binary,a.verifier,a.native_verifier,a.new_output)
