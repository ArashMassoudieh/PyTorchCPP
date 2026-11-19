#!/usr/bin/env python3
"""Fabricated native-v2 input through all forecast CLI/reload families; no research fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
from test_corrected_gauge_package import inputs
from prepare_corrected_gauge_package import prepare

HERE=Path(__file__).resolve().parent
NAMES=['ffn_causal_forecast_direct_log_q','ffn_causal_forecast_log_residual',
       'lstm_hydrofusion_direct_log_q','lstm_hydrofusion_log_residual','pinn_causal_forecast_log_residual',
       'ffn_physical_forecast_log_residual','lstm_physical_forecast_log_residual']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(binary,verifier,output):
    binary,verifier,output=map(lambda p:Path(p).absolute(),(binary,verifier,output))
    output.mkdir(parents=True,exist_ok=False)
    native,weather=inputs(output)
    prepare(native,weather,output/'prepared_v2',producer_catalog=weather/'station_catalog.json',original_export_manifest=weather/'station_source_manifest.json',station_metadata=weather/'station.rdb')
    manifest=output/'prepared_v2/native_input_manifest.json'
    dispatch=HERE.parents[2]/'build-hydrobatch'/('reference_et_v2_'+hashlib.sha256(str(output).encode()).hexdigest()[:12])
    dispatch.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='')
    def limits():resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    receipt={'binary_sha256':sha(binary),'verifier_sha256':sha(verifier),'native_v2_sha256':sha(manifest),'commands':[],'synthetic_only':True}
    def command(name,args,expected=0,reason=None):
        with (output/(name+'.log')).open('x') as log:
            p=subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600,preexec_fn=limits)
        text=(output/(name+'.log')).read_text()
        if p.returncode!=expected or reason and reason not in text:raise RuntimeError(name+': '+text[-2000:])
        receipt['commands'].append({'name':name,'arguments':list(map(str,args)),'exit_code':p.returncode})
    configs={};jobs=[]
    for name in NAMES:
        cfg=json.loads((HERE/'hydrofusion_five_method_checks'/(name+'.json')).read_text())
        cfg.update(epochs=3,hydrofusion_synthetic_verification=False,
                   hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=sha(manifest))
        path=dispatch/(name+'.json');path.write_text(json.dumps(cfg,indent=2)+'\n');configs[name]=cfg
        mode=name.removesuffix('_direct_log_q').removesuffix('_log_residual')
        jobs.append(mode+' '+path.name)
    batch=dispatch/'positive.batch';batch.write_text('\n'.join(jobs)+'\n')
    command('positive_cli',[binary,batch,output/'bundles'])
    for name in NAMES:
        mode=name.removesuffix('_direct_log_q').removesuffix('_log_residual')
        command(name+'_reload',[verifier,output/'bundles'/name,mode])
    changes=[('bad_day_offset','daily_pet_utc_start_hour',24,'Explicit bounded causal availability policy'),
             ('wrong_et_semantics','pet_input_semantics','EVPTRNS','Unsupported native reference ET semantics'),
             ('legacy_schema_with_new_fields','schema','HydroFusionPreparedNative-v1','Versioned native ET semantics required')]
    for name,key,value,reason in changes:
        folder=output/name;shutil.copytree(output/'prepared_v2',folder)
        p=folder/'native_input_manifest.json';m=json.loads(p.read_text());m[key]=value;p.write_text(json.dumps(m))
        cfg=dict(configs[NAMES[0]],experiment_id=name,hydrofusion_native_manifest_path=str(p),hydrofusion_native_manifest_sha256=sha(p))
        path=dispatch/(name+'.json');path.write_text(json.dumps(cfg));batch=dispatch/(name+'.batch');batch.write_text('ffn_causal_forecast '+path.name+'\n')
        command(name,[binary,batch,output/(name+'_bundles')],1,reason)
    (output/'dispatch_configs').mkdir()
    for p in dispatch.iterdir():shutil.copy2(p,output/'dispatch_configs'/p.name)
    (output/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS: seven synthetic native-v2 CLI modes and fresh-process reloads; three semantic rejection checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['binary','verifier','new_output_directory']:parser.add_argument(key,type=Path)
    args=parser.parse_args();run(args.binary,args.verifier,args.new_output_directory)
