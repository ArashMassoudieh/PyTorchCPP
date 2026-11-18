#!/usr/bin/env python3
"""Probe semantic binding on owned copies of fabricated extended bundles."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hydrofusion_task4_cli_checks'))
from check_hydrofusion_bundle_integrity import refresh_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('verifier',type=Path);p.add_argument('bundle',type=Path);p.add_argument('approach');p.add_argument('output',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for kind in ['metadata_scale','recipe_config','scalers','time_map','nonfinite_prediction']:
        bundle=args.output/kind;shutil.copytree(args.bundle,bundle)
        if kind=='metadata_scale':
            path=bundle/'hydrofusion_metadata.txt';lines=path.read_text().splitlines()
            lines=[('q_ref=999' if line.startswith('q_ref=') else line) for line in lines];path.write_text('\n'.join(lines)+'\n')
        elif kind=='recipe_config':
            path=bundle/'experiment_config.json';config=json.loads(path.read_text());config['storage_coeff']=config['storage_coeff']+0.1;path.write_text(json.dumps(config,indent=2)+'\n')
        elif kind=='scalers':
            path=bundle/'scalers.csv';lines=list(csv.reader(path.open()));lines[1][5]=str(float(lines[1][5])+0.1)
            with path.open('w') as stream:csv.writer(stream,lineterminator='\n').writerows(lines)
        elif kind=='time_map':
            path=bundle/'hydrofusion_origin_target_map.csv';lines=path.read_text().splitlines();lines[1]='1,2';path.write_text('\n'.join(lines)+'\n')
        else:
            path=bundle/'predictions.csv';lines=list(csv.reader(path.open()));lines[1][5]='nan'
            with path.open('w') as stream:csv.writer(stream,lineterminator='\n').writerows(lines)
        refresh_manifest(bundle,path.name)
        result=subprocess.run([str(args.verifier.resolve()),str(bundle),args.approach],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (args.output/(kind+'.log')).write_text(result.stdout)
        if result.returncode==0:raise RuntimeError('Corrupt bundle accepted: '+kind)
    print('All five semantic corruption cases rejected for '+args.approach)


if __name__=='__main__':main()
