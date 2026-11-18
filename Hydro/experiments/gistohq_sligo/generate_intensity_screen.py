"""Freeze prospective matched intensity-response screen after validation-only selection."""
from pathlib import Path
import json,hashlib
base=Path(__file__).resolve().parent;repo=base.parents[2];r=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.load(open(r/'intensity_smoke/verification.json'));assert proof['binary_sha256']==sha(repo/'build-hydrobatch/HydroBatch')
paramsfile=r/'intensity_calibration/selected_params.json';params=json.load(open(paramsfile));src=base/'improvement_loop_20261023/six_year_physics_configs/run_manifest.json';old=json.load(open(src));out=base/'improvement_loop_20261023/intensity_screen_configs';out.mkdir(exist_ok=True);jobs=[]
for j in old['jobs']:
 if j['model']!='urban_no_detention' or j['seed']!=42 or j['year'] not in ['2018','2020','2023']:continue
 for tag in ['0p0','0p5','1p0']:
  p=params[j['year']][tag];strength=float(tag.replace('p','.'));variant='intensity_'+tag;name=f'intensityscreen_{j["year"]}_{variant}_s42';c=json.load(open(j['config']));c.update(experiment_id=name,epochs=30,shared_scale_volume_weight=0.,urban_intensity_quick_strength=strength,urban_intensity_reference_mm_h=p['intensity_reference_mm_h'],urban_quick_k=p['quick_k'],urban_slow_k=p['slow_k'],urban_detention_k=p['detention_k'],urban_channel_k=p['channel_k'],urban_quick_fraction=p['quick_fraction'],urban_detention_fraction=0.,urban_runoff_coefficient=p['runoff_coefficient'],pinn_routing_lag_hours=p['lag_steps'])
  path=(out/(name+'.json')).resolve();assert not path.exists();path.write_text(json.dumps(c,indent=2)+'\n');batch=path.with_suffix('.batch');batch.write_text(f'lstm_pinn {path}\n');jobs.append(dict(j,id=name,model=variant,variant=variant,config=str(path),config_sha256=sha(path),batch=str(batch)))
(out/'run_manifest.json').write_text(json.dumps(dict(purpose='Independent intensity-partition screen with equal validation calibration budgets',binary_sha256=proof['binary_sha256'],selected_params_sha256=sha(paramsfile),selected_params=str(paramsfile),strengths=[0.,.5,1.],reference='Training-positive rainfall q90',jobs=jobs),indent=2)+'\n');print('Frozen',len(jobs),'jobs',sha(out/'run_manifest.json'))
