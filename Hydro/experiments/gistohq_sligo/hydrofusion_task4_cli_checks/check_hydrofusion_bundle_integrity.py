#!/usr/bin/env python3
"""Tamper synthetic-only bundle copies and exercise the real fresh-load verifier."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh_manifest(bundle, name):
    path = bundle/'artifact_manifest.csv'
    lines = path.read_text().splitlines()
    rows = list(csv.reader(lines[2:]))
    matches = [r for r in rows if r[0]==name]
    if len(matches)!=1:
        raise AssertionError('Expected exactly one managed artifact entry.')
    target = bundle/name
    matches[0][1:] = [str(target.stat().st_size),sha(target)]
    with path.open('w') as out:
        out.write(lines[0]+'\n'+lines[1]+'\n')
        csv.writer(out,lineterminator='\n').writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('verifier')
    p.add_argument('synthetic_bundle')
    p.add_argument('output')
    args = p.parse_args()
    original = Path(args.synthetic_bundle).resolve(strict=True)
    config = json.loads((original/'experiment_config.json').read_text())
    if config.get('hydrofusion_synthetic_verification') is not True or config.get('use_hydro_package') or config.get('use_csv_data'):
        p.error('This test accepts only explicit synthetic HydroFusion fixture outputs.')
    verifier = Path(args.verifier).resolve(strict=True)
    output = Path(args.output).absolute()
    output.mkdir(parents=True,exist_ok=False)
    result = {'verifier_sha256':sha(verifier),'cases':{}}
    for case in ['metadata_crc','map_crc','mode_semantics','scaler_binding','nan_prediction']:
        bundle = output/case
        shutil.copytree(original,bundle)
        expected = ''
        if case=='metadata_crc':
            path = bundle/'hydrofusion_metadata.txt'
            text=path.read_text();value=next(x for x in text.splitlines() if x.startswith('q_ref='))
            path.write_text(text.replace(value,'q_ref=99'))
            expected='integrity check failed'
        elif case=='map_crc':
            path=bundle/'hydrofusion_origin_target_map.csv'
            path.write_text(path.read_text()+'0,1\n')
            expected='integrity check failed'
        elif case=='mode_semantics':
            path=bundle/'experiment_config.json';c=json.loads(path.read_text())
            c['hydrofusion_target_mode']='direct_log_q' if c['hydrofusion_target_mode']=='log_residual' else 'log_residual'
            path.write_text(json.dumps(c,indent=2)+'\n');refresh_manifest(bundle,path.name)
            expected='config and metadata mode/bounds disagree'
        elif case=='scaler_binding':
            path=bundle/'scalers.csv';rows=list(csv.reader(path.read_text().splitlines()))
            # Alter one finite offset, keep a well-formed scaler and manifest.
            rows[1][5]=str(float(rows[1][5])+1)
            with path.open('w') as f:csv.writer(f,lineterminator='\n').writerows(rows)
            refresh_manifest(bundle,path.name)
            expected='scalers_sha256 does not match'
        else:
            path=bundle/'predictions.csv';rows=list(csv.reader(path.read_text().splitlines()));rows[1][5]='nan'
            with path.open('w') as f:csv.writer(f,lineterminator='\n').writerows(rows)
            refresh_manifest(bundle,path.name)
            expected='Invalid scaler predicted'
        command=[str(verifier),str(bundle),str(bundle/'hydrofusion_metadata.txt'),
                 str(bundle/'hydrofusion_origin_target_map.csv'),'ffn_hydrofusion']
        tested=subprocess.run(command,capture_output=True,text=True)
        text=tested.stdout+tested.stderr;(output/(case+'.log')).write_text(text)
        if tested.returncode==0 or expected not in text:
            raise AssertionError(case+': expected rejection '+expected+'; got '+str(tested.returncode)+' '+text)
        result['cases'][case]={'exit_code':tested.returncode,'expected_error':expected}
        print(case+': correctly rejected')
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
