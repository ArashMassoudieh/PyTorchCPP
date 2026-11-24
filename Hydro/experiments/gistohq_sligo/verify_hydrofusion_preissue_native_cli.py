#!/usr/bin/env python3
"""Exercise v5 prepared-native assembly/reload using fabricated observations only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from test_corrected_gauge_package import inputs
from prepare_corrected_gauge_package import prepare
from prepare_hydrofusion_search_slice import prepare as crop
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(binary,verifier,synthetic_receipt,output):
    binary,verifier,synthetic_receipt,output=(Path(p).absolute() for p in (binary,verifier,synthetic_receipt,output));output.mkdir(parents=True,exist_ok=False)
    saved=json.loads(synthetic_receipt.read_text())
    if saved['status']!='VERIFIED' or saved['binary_sha256']!=sha(binary):raise ValueError('Synthetic preflight/binary mismatch')
    native,weather=inputs(output)
    prepare(native,weather,output/'fabricated_native',producer_catalog=weather/'station_catalog.json',original_export_manifest=weather/'station_source_manifest.json',station_metadata=weather/'station.rdb')
    manifest=crop(output/'fabricated_native/native_input_manifest.json',output/'fabricated_development',220)
    dispatch=HERE.parents[2]/'build-hydrobatch'/('preissue_native_checks_'+hashlib.sha256(str(output).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');commands=[]
    for record in saved['commands']:
        name=record['name']
        if name.endswith('_reload') or 's25' in name or record['exit_code']!=0:continue
        if not name.endswith(('_zero_channel_v1','_rate_hold_v1')):continue
        bp=Path(record['args'][1]);family,config=bp.read_text().split();c=json.loads((bp.parent/config).read_text())
        name='native_'+name;c.update(experiment_id=name,hydrofusion_synthetic_verification=False,hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=sha(manifest))
        cp=dispatch/(name+'.json');cp.write_text(json.dumps(c,indent=2)+'\n');batch=dispatch/(name+'.batch');batch.write_text(f'{family} {cp.name}\n');bundle=output/name/name
        for label,args in [('fit',[binary,batch,output/name]),('reload',[verifier,bundle,output/(name+'_validation.csv')])]:
            log=output/(name+'_'+label+'.log')
            with log.open('x') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600,check=True)
            commands.append(dict(name=name+'_'+label,args=list(map(str,args)),log_sha256=sha(log)))
    if len(commands)!=20:raise ValueError('Exactly all5families x2arms required')
    result=dict(status='VERIFIED',fabricated_native_only=True,binary_sha256=sha(binary),native_verifier_sha256=sha(verifier),commands=commands)
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS20fabricated-native fit/reload commands across all5families')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('binary','verifier','synthetic_receipt','output'):p.add_argument(n,type=Path)
    a=p.parse_args();run(a.binary,a.verifier,a.synthetic_receipt,a.output)
