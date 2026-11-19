#!/usr/bin/env python3
"""Saved synthetic-only v2 CLI/reload and exact v1 default compatibility check."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch'); p.add_argument('reload'); p.add_argument('baseline'); p.add_argument('output')
    a = p.parse_args()
    root = Path(__file__).resolve().parent/'hydrofusion_five_method_checks'
    out = Path(a.output).resolve()
    try: out.relative_to(root.parents[3])
    except ValueError: p.error('Output must be inside this checkout so HydroBatch can locate its repository root.')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    evidence = {'binaries': {str(Path(x).resolve()): hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in (a.batch,a.reload,a.baseline)}, 'checks': []}
    def run(name, cmd, expected=0):
        r = subprocess.run(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,timeout=180)
        (out/(name+'.log')).write_text(r.stdout)
        if r.returncode != expected: raise RuntimeError(name+' failed: '+r.stdout[-2000:])
        evidence['checks'].append(name)
    modes = [('ffn_causal_forecast','direct_log_q'),('ffn_causal_forecast','log_residual'),
             ('lstm_hydrofusion','direct_log_q'),('lstm_hydrofusion','log_residual'),
             ('pinn_causal_forecast','log_residual'),('ffn_physical_forecast','log_residual'),('lstm_physical_forecast','log_residual')]
    for mode,target in modes:
        key = mode+'_'+target
        base = json.loads((root/(key+'.json')).read_text())
        # Fabricated fixture only; never permit caller-supplied native inputs.
        if not base['hydrofusion_synthetic_verification'] or base['use_csv_data'] or base['use_hydro_package']:
            raise RuntimeError('Synthetic-only fixture required')
        base['epochs']=3
        v2 = dict(base, experiment_id=key+'_v2')
        if mode!='pinn_causal_forecast': v2['hydrofusion_objective']='log_flow_mse_v2'
        if mode.startswith('ffn'): v2['hydrofusion_ffn_activation']='silu'
        if 'physical' in mode or mode=='pinn_causal_forecast': v2['physics_profile']='causal_two_reservoir_mean_v2'
        for label,cfg in [('v1',base),('v2',v2)]:
            config=out/(key+'_'+label+'.json');config.write_text(json.dumps(cfg,indent=2)+'\n')
            batch=out/(key+'_'+label+'.batch');batch.write_text(mode+' '+config.name+'\n')
            dest=out/(key+'_'+label)
            run(key+'_'+label,[a.batch,batch,dest])
            bundle=dest/cfg['experiment_id']
            run(key+'_'+label+'_reload',[a.reload,bundle,mode])
            saved=json.loads((bundle/'experiment_config.json').read_text())
            for field in ['hydrofusion_objective','hydrofusion_ffn_activation']:
                if saved[field]!=cfg.get(field,{'hydrofusion_objective':'log_mse_v1','hydrofusion_ffn_activation':'tanh'}[field]):
                    raise RuntimeError('Config option did not round-trip: '+field)
            if label=='v1':
                before=out/(key+'_baseline');run(key+'_baseline',[a.baseline,batch,before])
                if (bundle/'predictions.csv').read_bytes()!=(before/cfg['experiment_id']/'predictions.csv').read_bytes():
                    raise RuntimeError(key+' v1 regression')
                evidence['checks'].append(key+'_v1_byte_exact')
    # Semantic mismatch must reject even after refreshing the outer manifest.
    sys.path.insert(0,str(root.parent/'hydrofusion_task4_cli_checks'))
    from check_hydrofusion_bundle_integrity import refresh_manifest
    for field,value,approach,key in [
        ('hydrofusion_objective','log_mse_v1','ffn_causal_forecast','ffn_causal_forecast_direct_log_q_v2'),
        ('hydrofusion_ffn_activation','tanh','ffn_causal_forecast','ffn_causal_forecast_direct_log_q_v2'),
        ('physics_profile','causal_two_reservoir_v1','ffn_physical_forecast','ffn_physical_forecast_log_residual_v2')]:
        source=out/key/key
        altered=out/('tamper_'+field);shutil.copytree(source,altered)
        config=altered/'experiment_config.json';c=json.loads(config.read_text());c[field]=value
        config.write_text(json.dumps(c,indent=2)+'\n');refresh_manifest(altered,config.name)
        run('tamper_'+field,[a.reload,altered,approach],expected=1)
    base=json.loads((root/'ffn_causal_forecast_direct_log_q.json').read_text())
    for name,mode,changes in [
        ('unknown_objective','ffn_causal_forecast',{'hydrofusion_objective':'bad'}),
        ('unknown_activation','ffn_causal_forecast',{'hydrofusion_ffn_activation':'bad'}),
        ('recurrent_silu','lstm_hydrofusion',{'hydrofusion_ffn_activation':'silu'}),
        ('nonphysical_mean','ffn_causal_forecast',{'physics_profile':'causal_two_reservoir_mean_v2'}),
        ('physical_objective','pinn_causal_forecast',{'hydrofusion_objective':'log_flow_mse_v2','hydrofusion_target_mode':'log_residual'}),
        ('legacy_objective','ffn',{'hydrofusion_objective':'log_flow_mse_v2'})]:
        cfg=dict(base,**changes);config=out/(name+'.json');config.write_text(json.dumps(cfg)+'\n')
        batch=out/(name+'.batch');batch.write_text(mode+' '+config.name+'\n')
        run(name,[a.batch,batch,out/(name+'_output')],expected=1)
    evidence['new_research_fits']=0
    evidence['scope']='Tiny fabricated synthetic fixtures, 3 epochs; no real data or performance claims.'
    (out/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('PASS seven v2 variants with fresh reload; seven v1 variants byte-identical to baseline')


if __name__=='__main__': main()
