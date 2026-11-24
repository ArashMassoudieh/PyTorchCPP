#!/usr/bin/env python3
"""Compare raw deterministic disabled-rain predictions against the accepted pre-v5 binary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(before,after,out):
    before,after,out=(Path(p).absolute() for p in (before,after,out));out.mkdir(parents=True,exist_ok=False)
    dispatch=HERE.parents[2]/'build-hydrobatch'/('rain_legacy_'+hashlib.sha256(str(out).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
    receipt=dict(before_sha256=sha(before),after_sha256=sha(after),comparisons=[])
    variants=[(f,'legacy20') for f in ('ffn_causal_forecast','lstm_hydrofusion','pinn_causal_forecast','ffn_physical_forecast','lstm_physical_forecast')]+[('ffn_physical_forecast',r) for r in ('zero25','absolute25','anchored25')]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    for family,context in variants:
        name=family+'_'+context;c=json.loads((HERE/'hydrofusion_five_method_checks'/(family+'_log_residual.json')).read_text());c.update(experiment_id=name,epochs=3)
        if context!='legacy20':c.update(hydrofusion_context_recipe=context,hydrofusion_history_hours='0,1,3,6,12,23',physics_profile='causal_valid_time_dry_v3',hydrofusion_development_only=True,hydrofusion_objective='log_flow_mse_v2')
        cp=dispatch/(name+'.json');cp.write_text(json.dumps(c,indent=2)+'\n');bp=dispatch/(name+'.batch');bp.write_text(f'{family} {cp.name}\n')
        files=[]
        for binary,label in ((before,'before'),(after,'after')):
            output=out/(name+'_'+label)
            with (out/(name+'_'+label+'.log')).open('x') as log:subprocess.run([str(binary),str(bp),str(output)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600,check=True)
            files.append(output/name/'predictions.csv')
        if files[0].read_bytes()!=files[1].read_bytes():raise RuntimeError('Legacy prediction regression: '+name)
        receipt['comparisons'].append(dict(name=name,predictions_sha256=sha(files[0]),rows=len(files[0].read_text().splitlines())-1))
    receipt['status']='VERIFIED';(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS8byte-identical disabled-rain prediction comparisons')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('before','after','output'):p.add_argument(n,type=Path)
    a=p.parse_args();run(a.before,a.after,a.output)
