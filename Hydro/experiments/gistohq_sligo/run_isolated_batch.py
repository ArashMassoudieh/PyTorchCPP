#!/usr/bin/env python3
"""Run frozen experiment jobs with isolated outputs, progress and verified resume."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[3]


def validate_generated_wrapper_build(binary, repo_root=REPO_ROOT):
    """Reject a batch when generated supervised wrappers or the binary are stale."""
    repo_root = Path(repo_root)
    binary = Path(binary)
    generator = repo_root / 'Hydro/generate_reduced_reservoir_supervised_wrappers.py'
    pairs = (
        (repo_root / 'Hydro/models/ffn_wrapper.cpp',
         repo_root / 'Hydro/generated/ffn_wrapper_reduced_reservoir.cpp'),
        (repo_root / 'Hydro/models/lstmnetworkwrapper.cpp',
         repo_root / 'Hydro/generated/lstmnetworkwrapper_reduced_reservoir.cpp'),
    )
    for source, generated in pairs:
        if not generated.exists():
            raise ValueError(f'Missing generated wrapper: {generated}')
        newest_input = max(source.stat().st_mtime_ns, generator.stat().st_mtime_ns)
        if generated.stat().st_mtime_ns < newest_input:
            raise ValueError(f'Stale generated wrapper: {generated}; regenerate before running')
        if binary.stat().st_mtime_ns < generated.stat().st_mtime_ns:
            raise ValueError(f'Stale HydroBatch binary: rebuild after generating {generated}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(job, binary, root):
    output=root/job['id']
    output.mkdir(parents=True,exist_ok=True)
    marker=output/'VERIFIED.json'
    identity={'config_sha256':sha(job['config']),'binary_sha256':sha(binary)}
    if job.get('config_sha256',identity['config_sha256']) != identity['config_sha256']:
        raise ValueError('Configuration changed after manifest freeze')
    if marker.exists():
        old=json.loads(marker.read_text())
        if all(old[k]==v for k,v in identity.items()):
            prediction=output/job['id']/'predictions.csv'
            if sha(prediction)==old['predictions_sha256']:
                return dict(id=job['id'],status='verified_resume')
        raise ValueError(f'Existing run has different provenance: {output}')
    # Never silently archive/overwrite a prior partial research run.
    if (output/job['id']).exists():
        raise ValueError(f'Unverified prior output needs inspection: {output}')
    start=time.time()
    with (output/'run.log').open('w') as log:
        subprocess.run([str(binary),job['batch'],str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
    prediction=output/job['id']/'predictions.csv'
    bounds=job['bounds']
    counts={s:0 for s in ['train','validation','test','beyond_test_end']}
    previous=-math.inf
    with prediction.open() as stream:
        for row in csv.DictReader(stream):
            x=float(row['x']);split=row['split']
            if not all(math.isfinite(float(row[k])) for k in ['x','observed','predicted']):
                raise ValueError('Nonfinite prediction row')
            if x <= previous:raise ValueError('Duplicate or unordered prediction timestamp')
            previous=x
            if x >= 77016:
                raise ValueError('Reserved-period row exported')
            expected=('train' if x <= bounds['train_end_x'] else
                      'validation' if x <= bounds['validation_end_x'] else
                      'test' if x <= bounds['test_end_x'] else 'beyond_test_end')
            if expected!=split:
                raise ValueError(f'{job["id"]}: split mismatch at {x}')
            counts[split]+=1
    if not all(counts[s]>0 for s in ['train','validation','test']):
        raise ValueError('Empty partition')
    result=dict(id=job['id'],status='complete',seconds=time.time()-start,
                counts=counts,predictions_sha256=sha(prediction),**identity)
    marker.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    validate_generated_wrapper_build(args.binary)
    jobs=json.loads(args.manifest.read_text())['jobs']
    args.output.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        active={};iterator=iter(jobs);complete=0;failed=False
        def fill():
            while len(active)<args.workers:
                job=next(iterator,None)
                if job is None:return
                active[pool.submit(run,job,args.binary.resolve(),args.output.resolve())]=job['id']
        fill()
        while active:
            done,_=concurrent.futures.wait(active,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                name=active.pop(future)
                try: result=future.result();complete+=1
                except Exception as error:
                    failed=True;result=dict(id=name,status='failed',error=str(error))
                with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                print(json.dumps(result),flush=True)
            print(json.dumps(dict(completed=complete,total=len(jobs),active=list(active.values()),failed=failed)),flush=True)
            if not failed:fill()
        if failed:raise SystemExit(1)


if __name__=='__main__':main()
