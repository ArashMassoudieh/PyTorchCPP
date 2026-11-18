#!/usr/bin/env python3
"""Generate a fixed five-configuration annual development comparison."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--seal-guard',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('seal_guard',args.seal_guard)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    protocol=json.loads(args.protocol.read_text())
    template=json.loads(args.template.read_text())
    package=Path(template['hydro_package_path'])
    metadata=json.loads((package/'manifest.json').read_text())
    if metadata['study_end'] >= '2023-10-15T00:00:00Z':
        raise ValueError('Only a development-truncated package may be used')
    args.output.mkdir(parents=True,exist_ok=True)
    models=[('ffn','ffn','water_balance','log_standardize'),
            ('lstm_24h','lstm','water_balance','log_standardize'),
            ('two_reservoir','pinn','pinn_two_reservoir_hybrid','none'),
            ('ffn_residual_hybrid','ffn_pinn','ffn_residual_pinn_hybrid','standardize_input_asinh_residual'),
            ('lstm_residual_hybrid','lstm_pinn','lstm_residual_pinn_hybrid','standardize_input_asinh_residual')]
    jobs=[]
    for fold in protocol['folds']:
        bounds={k:fold[k] for k in ['train_end_x','validation_end_x','test_end_x']}
        for key,value in bounds.items():guard.assert_before_seal(end_x=value,label=f'{fold["name"]}:{key}')
        for model,mode,profile,norm in models:
            # A deterministic physical model needs one run, not three fake replicates.
            for seed in ([42] if mode=='pinn' else protocol['seeds']):
                name=f'annual_{fold["name"]}_{model}_s{seed}'
                config=dict(template,experiment_id=name,epochs=150,batch_size=16,
                    random_seed=seed,physics_profile=profile,normalization=norm,
                    storage_coeff=0.1,lambda_decay=0.04,runoff_coeff=0.3,forcing_gain=0.25,
                    pinn_flow_exponent=0.0,pinn_routing_lag_hours=0.0,pinn_initial_flow=0.0,
                    lstm_warm_start_checkpoint_path='',**bounds)
                path=args.output/(name+'.json')
                if path.exists():raise ValueError(f'Refusing to overwrite frozen config: {path}')
                path.write_text(json.dumps(config,indent=2)+'\n')
                batch=args.output/(name+'.batch');batch.write_text(f'{mode} {path.resolve()}\n')
                jobs.append(dict(id=name,year=fold['name'],model=model,seed=seed,
                                 test_start_x=fold['test_start_x'],
                                 config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 config=str(path.resolve()),batch=str(batch.resolve()),bounds=bounds))
    manifest=dict(protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                  note='Fixed 150-epoch validation-checkpoint runs; common settings, no test tuning. Physical constants are fixed comparison defaults, not claimed calibrated optima. Target transforms differ across configurations.',jobs=jobs)
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Frozen {len(jobs)} runs; deterministic baseline appears once per year.')


if __name__=='__main__':main()
