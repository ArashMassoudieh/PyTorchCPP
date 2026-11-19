#!/usr/bin/env python3
"""Read-only saved-checkpoint diagnostics; no optimizer or research fitting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from analyze_hydrofusion_all_methods import verified_jobs,sha


def run(source,extension,output,binary):
    output=Path(output).resolve(strict=True);binary=Path(binary).resolve(strict=True)
    jobs=verified_jobs(source)[1]+verified_jobs(extension)[1]
    jobs=[j for j in jobs if j['family']!='pinn_causal_forecast']
    if len(jobs)!=18:raise ValueError('Exactly18 existing neural bundles required')
    folder=output/'neural';folder.mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    record=dict(binary=str(binary),binary_sha256=sha(binary),script_sha256=sha(__file__),fitting_performed=False,jobs=[])
    command=[str(binary)]
    for j in jobs:command += [str(j['bundle']),str(folder/(j['name']+'.csv'))]
    master=output/'neural_batch_execution.log'
    with master.open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
    text=master.read_text()
    if text.count('PASS all_test_rows=8387 production_target_free_checks=12 ')!=18:
        raise ValueError('Expected18complete all-row parity proofs')
    sections=text.split('DIAGNOSE ')[1:]
    if len(sections)!=18:raise ValueError('Expected18 named diagnostic logs')
    for j,section in zip(jobs,sections):
        name=j['name'];csv=folder/(name+'.csv');log=folder/(name+'.log')
        if section.splitlines()[0]!=name:raise ValueError('Diagnostic order mismatch')
        log.write_text(section)
        record['jobs'].append(dict(name=name,family=j['family'],mode=j['mode'],seed=j['seed'],bundle=str(j['bundle']),csv=str(csv),csv_sha256=sha(csv),log_sha256=sha(log)))
        print('DIAGNOSED '+name,flush=True)
    record['shared_immutable_prepared_tensors']=True
    record['master_log_sha256']=sha(master)
    with (output/'neural_export_verification.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('extension');p.add_argument('output');p.add_argument('binary');a=p.parse_args();run(a.source,a.extension,a.output,a.binary)
