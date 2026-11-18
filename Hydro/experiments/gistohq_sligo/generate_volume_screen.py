#!/usr/bin/env python3
"""Freeze a bounded shared-scale local-volume objective screen after smoke checks."""
from pathlib import Path
import json,hashlib
base=Path(__file__).resolve().parent;repo=base.parents[2];root=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023')
proof=json.load(open(root/'volume_smoke/verification.json'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert proof['binary_sha256']==sha(repo/'build-hydrobatch/HydroBatch')
assert all(proof[k] for k in ['zero_prediction_identical','weighted_prediction_differs','negative_rejected','exported_weights_verified'])
source=base/'improvement_loop_20261023/six_year_physics_configs/run_manifest.json';manifest=json.load(open(source));out=base/'improvement_loop_20261023/volume_screen_configs';out.mkdir(exist_ok=True);jobs=[]
for j in manifest['jobs']:
 if j['model']!='urban_no_detention' or j['seed']!=42 or j['year'] not in ['2018','2020','2023']:continue
 template=json.load(open(j['config']))
 for w in [0.,.1,1.]:
  variant='volume_'+str(w).replace('.','p');name=f'volumescreen_{j["year"]}_{variant}_s42';c=dict(template,experiment_id=name,epochs=30,shared_scale_volume_weight=w);p=(out/(name+'.json')).resolve();assert not p.exists();p.write_text(json.dumps(c,indent=2)+'\n');b=p.with_suffix('.batch');b.write_text(f'lstm_pinn {p}\n');job=dict(j,id=name,model=variant,variant=variant,config=str(p),batch=str(b),config_sha256=sha(p));jobs.append(job)
(out/'run_manifest.json').write_text(json.dumps(dict(purpose='Independent local-volume-penalty screen on frozen urban physics',binary_sha256=proof['binary_sha256'],source_manifest_sha256=sha(source),weights=[0.,.1,1.],epochs=30,jobs=jobs),indent=2)+'\n');print('Frozen',len(jobs),'jobs')
