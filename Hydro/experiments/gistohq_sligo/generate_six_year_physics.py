#!/usr/bin/env python3
"""Freeze six-year paired common-scale physics evaluation after validation calibration."""
from pathlib import Path
import json,hashlib
base=Path(__file__).resolve().parent;repo=base.parents[2]
root=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023')
selection=root/'six_year_physics_calibration/selected_params.json';params=json.loads(selection.read_text())
source=base/'decade_20261022/annual_configs_v2/run_manifest.json';annual=json.loads(source.read_text())
out=base/'improvement_loop_20261023/six_year_physics_configs';out.mkdir(exist_ok=True);jobs=[]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for j in annual['jobs']:
 if j['model']!='lstm_24h' or j['seed']!=42:continue
 template=json.loads(Path(j['config']).read_text());year=j['year'];p=params[year]['managed']
 assert template['test_end_x']<77016
 for arm in ['zero','urban_no_detention']:
  for seed in [42,123,27182]:
   name=f'sixyear_{year}_{arm}_s{seed}';c=dict(template)
   c.update(experiment_id=name,epochs=150,batch_size=16,random_seed=seed,lstm_sequence_length=24,lstm_warm_start_checkpoint_path='',physics_profile='lstm_shared_scale_ablation_zero' if arm=='zero' else 'lstm_shared_scale_ablation_urban',physical_error_weight=0.,high_flow_loss_weight=0.,physics_weight=0.,use_mrms_spatial_cv_feature=False,antecedent_index_halflife_hours=0.,urban_quick_k=p['quick_k'],urban_slow_k=p['slow_k'],urban_detention_k=p['detention_k'],urban_channel_k=p['channel_k'],urban_quick_fraction=p['quick_fraction'],urban_detention_fraction=0.,urban_runoff_coefficient=p['runoff_coefficient'],pinn_routing_lag_hours=float(p['lag_steps']))
   path=(out/(name+'.json')).resolve();assert not path.exists();path.write_text(json.dumps(c,indent=2)+'\n');batch=path.with_suffix('.batch');batch.write_text(f'lstm_pinn {path}\n')
   jobs.append(dict(id=name,year=year,model=arm,variant=arm,seed=seed,test_start_x=j['test_start_x'],config=str(path),config_sha256=sha(path),batch=str(batch),bounds=j['bounds']))
manifest=dict(purpose='Six-year expanding-window common-scale physics evaluation',selection_source=str(selection),selection_sha256=sha(selection),annual_source_sha256=sha(source),binary_sha256=sha(repo/'build-hydrobatch/HydroBatch'),epochs=150,seeds=[42,123,27182],jobs=jobs)
(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Frozen',len(jobs),'jobs',sha(out/'run_manifest.json'))
