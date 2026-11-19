#!/usr/bin/env python3
"""Synthetic-only CLI checks for valid-time physics and exact legacy compatibility.

Requires separately built frozen binaries. Does not accept a native package or
launch research fits. Keeps configs, commands, binary hashes and raw predictions.
"""
import argparse
import hashlib
import json
import os
import sys
import csv
import shutil
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
FAMILIES=('ffn_causal_forecast','lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast','pinn_causal_forecast')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(binary,verifier,baseline,output):
    paths=[Path(p).resolve(strict=True) for p in (binary,verifier,baseline)]
    binary,verifier,baseline=paths;output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
    # Keep dispatch beneath the project for HydroBatch repository-root discovery.
    dispatch=HERE.parents[2]/'build-hydrobatch'/('valid_time_checks_'+hashlib.sha256(str(output).encode()).hexdigest()[:16])
    dispatch.mkdir(parents=True,exist_ok=False);configs=output/'configs';configs.mkdir()
    pins={str(p):digest(p) for p in paths};commands=[];legacy=[];new=[]
    integrity=HERE/'hydrofusion_five_method_checks/check_integrity.py'
    helper=HERE/'hydrofusion_task4_cli_checks/check_hydrofusion_bundle_integrity.py'
    for p in (Path(__file__).resolve(),integrity,helper):pins[str(p)]=digest(p)
    sys.path.insert(0,str(helper.parent))
    from check_hydrofusion_bundle_integrity import refresh_manifest
    def write_config(c,family,lines):
        p=dispatch/(c['experiment_id']+'.json');p.write_text(json.dumps(c,indent=2)+'\n')
        (configs/p.name).write_bytes(p.read_bytes());pins[str(p)]=digest(p);pins[str(configs/p.name)]=digest(configs/p.name)
        lines.append(f'{family} {p.name}');return c['experiment_id']
    specs=[]
    for family in FAMILIES:
        template=HERE/'hydrofusion_five_method_checks'/(family+'_log_residual.json');pins[str(template)]=digest(template)
        c=json.loads(template.read_text())
        if c.get('hydrofusion_synthetic_verification') is not True or c.get('hydrofusion_native_manifest_path') or c.get('use_csv_data') or c.get('use_hydro_package'):
            raise ValueError('Only explicit fabricated fixtures permitted')
        c.update(epochs=2,experiment_id='legacy_'+family)
        specs.append((write_config(c,family,legacy),family))
        if family in FAMILIES[2:]:
            v2=dict(c,experiment_id='legacy_v2_'+family,physics_profile='causal_two_reservoir_mean_v2')
            specs.append((write_config(v2,family,legacy),family))
            c.update(experiment_id='v3_'+family,physics_profile='causal_valid_time_dry_v3')
            specs.append((write_config(c,family,new),family))
    batches={}
    for label,lines in (('legacy',legacy),('v3',new)):
        p=dispatch/(label+'.batch');p.write_text('\n'.join(lines)+'\n');batches[label]=p;pins[str(p)]=digest(p);(configs/p.name).write_bytes(p.read_bytes())
    (output/'input_hashes.json').write_text(json.dumps(pins,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    def run(label,args,rejection=None):
        for p,h in pins.items():
            if digest(p)!=h:raise ValueError('Changed check input: '+p)
        with (output/(label+'.log')).open('x') as f:
            completed=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env,timeout=300)
        commands.append(dict(label=label,args=list(map(str,args)),exit_code=completed.returncode))
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if rejection:
            if completed.returncode==0 or rejection not in (output/(label+'.log')).read_text():
                raise RuntimeError('Expected semantic rejection absent: '+label)
        elif completed.returncode:raise RuntimeError('Failed '+label)
    run('baseline',[baseline,batches['legacy'],output/'baseline'])
    run('after_legacy',[binary,batches['legacy'],output/'after'])
    run('after_v3',[binary,batches['v3'],output/'after'])
    for name,family in specs:
        bundle=output/'after'/name
        if name.startswith('legacy_'):
            if (bundle/'predictions.csv').read_bytes()!=(output/'baseline'/name/'predictions.csv').read_bytes():
                raise ValueError('Legacy raw prediction regression: '+name)
        run('reload_'+name,[verifier,bundle,family])
        if name.startswith('v3_'):
            run('integrity_'+name,[sys.executable,integrity,verifier,bundle,family,output/('tamper_'+name)])
            changed=output/('reinterpret_'+name);shutil.copytree(bundle,changed)
            config=changed/'experiment_config.json';c=json.loads(config.read_text());c['physics_profile']='causal_two_reservoir_mean_v2';config.write_text(json.dumps(c,indent=2)+'\n');refresh_manifest(changed,config.name)
            models=changed/'models.csv'
            with models.open() as f:records=list(csv.reader(f))
            records[1][2]='hydrofusion-forecast-v2'
            with models.open('w') as f:csv.writer(f,lineterminator='\n').writerows(records)
            refresh_manifest(changed,models.name)
            run('reinterpret_'+name,[verifier,changed,family],rejection='Forecast checkpoint recipe/scale mismatch')
    result=dict(status='PASS',legacy_methods_byte_identical=5,legacy_configurations_byte_identical=8,v3_synthetic_modes=3,all_exported_synthetic_rows_fresh_reloaded=True,research_fits=0,semantic_corruptions_rejected=18,
                inputs=pins,commands=commands,script_sha256=digest(__file__))
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS 5 byte-identical legacy modes;3 v3 CLI modes;11 fresh-process reloads;0researchfits')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','verifier','baseline','output'):p.add_argument(name,type=Path)
    a=p.parse_args();verify(a.binary,a.verifier,a.baseline,a.output)
