#!/usr/bin/env python3
"""Synthetic-only native-v5 CLI, archive, weather binding and causal reload checks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
HERE = Path(__file__).resolve().parent
FAMILIES = ('ffn_causal_forecast', 'lstm_hydrofusion', 'ffn_physical_forecast', 'lstm_physical_forecast', 'pinn_causal_forecast')
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(binary, verifier, output):
    binary, verifier, output = (Path(p).absolute() for p in (binary, verifier, output))
    output.mkdir(parents=True, exist_ok=False)
    dispatch = HERE.parents[2]/'build-hydrobatch'/('preissue_checks_'+SHA_BYTES(str(output)))
    dispatch.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    weather = output/'weather'; weather.mkdir()
    provenance = weather/'preissue_rain_provenance.json'
    provenance.write_text(json.dumps(dict(schema='SyntheticPreissueRainFixture-v1', synthetic_only=True))+'\n')
    rain = weather/'preissue_rain.csv'
    with rain.open('x') as f:
        w = csv.writer(f); w.writerow('target_x origin_x valid_seconds issue_seconds release_proxy_seconds coverage rate_mm_h usable status'.split())
        for target in range(1,221):
            issue=1420070400+(target-1)*3600
            # One explicit gap tests both arms' common exclusion.
            available=target!=180
            w.writerow([target,target-1,issue-600,issue,issue-500,1,(target%11)*.2 if available else '',int(available),'WEATHER_VALUE_VERIFIED' if available else 'SYNTHETIC_MISSING'])
    receipt=dict(binary_sha256=SHA(binary),verifier_sha256=SHA(verifier),synthetic_only=True,commands=[])
    def command(name,args,expected=0,reason=None):
        log=output/(name+'.log')
        with log.open('x') as f:
            p=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env,timeout=900)
        text=log.read_text()
        if p.returncode!=expected or reason and reason not in text:raise RuntimeError(name+': '+text[-2500:])
        receipt['commands'].append(dict(name=name,args=list(map(str,args)),exit_code=p.returncode,log_sha256=SHA(log)))
    def config(name,family,recipe,context=None):
        c=json.loads((HERE/'hydrofusion_five_method_checks'/(family+'_log_residual.json')).read_text())
        c.update(experiment_id=name,epochs=3,physics_profile='causal_valid_time_dry_v3',
                 hydrofusion_objective='log_flow_mse_v2' if family!='pinn_causal_forecast' else 'log_mse_v1',
                 hydrofusion_context_recipe=context or ('anchored25' if 'physical' in family and family!='pinn_causal_forecast' else 'legacy20'),
                 hydrofusion_history_hours='0,1,3,6,12,23' if family.startswith('ffn') and (context or 'physical' in family) else '0',
                 hydrofusion_development_only=True,hydrofusion_preissue_rain_recipe=recipe,
                 hydrofusion_preissue_rain_path=str(rain),hydrofusion_preissue_rain_sha256=SHA(rain),
                 hydrofusion_preissue_rain_provenance_sha256=SHA(provenance))
        return c
    def fit(name,family,c,expected=0,reason=None):
        cp=dispatch/(name+'.json');cp.write_text(json.dumps(c,indent=2)+'\n')
        bp=dispatch/(name+'.batch');bp.write_text(f'{family} {cp.name}\n')
        command(name,[binary,bp,output/name],expected,reason)
        return output/name/name
    good={}
    for family in FAMILIES:
        supports=[]
        for recipe in ('zero_channel_v1','rate_hold_v1'):
            name=family+'_'+recipe;bundle=fit(name,family,config(name,family,recipe));good[name]=bundle
            command(name+'_reload',[verifier,bundle,family])
            c=json.loads((bundle/'experiment_config.json').read_text())
            if c['hydrofusion_preissue_rain_recipe']!=recipe or SHA(bundle/'preissue_rain.csv')!=SHA(rain):raise ValueError('Exported weather mismatch')
            with (bundle/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
            supports.append([(r['x'],r['observed']) for r in rows])
            if len(rows)!=22:raise ValueError('Gap exclusion not enforced: '+str(len(rows)))
        if supports[0]!=supports[1]:raise ValueError('Weather arms have different support')
    # Same26-channel width as hybrids, with zero physical channels in plain arms.
    for family in ('ffn_causal_forecast','lstm_hydrofusion'):
        name=family+'_zero25_rate';bundle=fit(name,family,config(name,family,'rate_hold_v1','zero25'))
        command(name+'_reload',[verifier,bundle,family])
    family='ffn_physical_forecast'
    for name, changes in [
        ('unknown_rain',dict(hydrofusion_preissue_rain_recipe='unknown')),
        ('wrong_rain_type',dict(hydrofusion_preissue_rain_recipe=5)),
        ('wrong_rain_hash',dict(hydrofusion_preissue_rain_sha256='0'*64)),
        ('wrong_provenance_hash',dict(hydrofusion_preissue_rain_provenance_sha256='0'*64)),
        ('wrong_rain_scope',dict(hydrofusion_development_only=False)),
        ('wrong_rain_profile',dict(physics_profile='causal_hourly_mean_v2')),
        ('disabled_bound_rain',dict(hydrofusion_preissue_rain_recipe='disabled'))]:
        c=config(name,family,'rate_hold_v1');c.update(changes);fit(name,family,c,1)
    def rehash(bundle):
        p=bundle/'artifact_manifest.csv';lines=p.read_text().splitlines();out=lines[:2]
        for line in lines[2:]:
            name,_,_=line.split(',');f=bundle/name;out.append(f'{name},{f.stat().st_size},{SHA(f)}')
        p.write_text('\n'.join(out)+'\n')
    source=good[family+'_rate_hold_v1']
    for kind in ('weather','provenance','schema','recipe','bound_weather'):
        dest=output/('tamper_'+kind);shutil.copytree(source,dest)
        if kind=='bound_weather':
            p=dest/'preissue_rain.csv';p.write_text(p.read_text().replace(',1,0.2,1,',',1,0.3,1,',1))
            p=dest/'experiment_config.json';c=json.loads(p.read_text());old=c['hydrofusion_preissue_rain_sha256'];c['hydrofusion_preissue_rain_sha256']=SHA(dest/'preissue_rain.csv');p.write_text(json.dumps(c))
            p=dest/'hydrofusion_preissue_rain_schema.txt';p.write_text(p.read_text().replace(old,c['hydrofusion_preissue_rain_sha256']))
        elif kind=='weather':
            p=dest/'preissue_rain.csv';p.write_text(p.read_text().replace(',1,0.2,1,',',1,0.3,1,',1))
        elif kind=='provenance':
            p=dest/'preissue_rain_provenance.json';p.write_text(p.read_text()+' ')
        elif kind=='schema':
            p=dest/'hydrofusion_preissue_rain_schema.txt';p.write_text(p.read_text().replace('600','599'))
        else:
            p=dest/'experiment_config.json';c=json.loads(p.read_text());c['hydrofusion_preissue_rain_recipe']='zero_channel_v1';p.write_text(json.dumps(c))
        rehash(dest);command('tamper_'+kind,[verifier,dest,family],1)
    receipt['status']='VERIFIED'
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('VERIFIED synthetic preissue CLI commands='+str(len(receipt['commands'])))

def SHA_BYTES(s):return hashlib.sha256(s.encode()).hexdigest()[:12]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('binary','verifier','new_output'):p.add_argument(n,type=Path)
    a=p.parse_args();run(a.binary,a.verifier,a.new_output)
