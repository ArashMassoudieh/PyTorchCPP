#!/usr/bin/env python3
"""Run validation-only annual calibration; preserve immutable per-year outputs."""
import csv,json,hashlib,subprocess
from pathlib import Path
base=Path(__file__).resolve().parent
configs=base/'improvement_loop_20261023/six_year_physics_calibration_configs'
root=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/six_year_physics_calibration')
root.mkdir(exist_ok=True)
selected={}
for j in json.loads((configs/'manifest.json').read_text())['jobs']:
 output=root/(j['year']+'.csv')
 if output.exists():raise RuntimeError('Existing calibration output requires provenance review')
 assert hashlib.sha256(Path(j['config']).read_bytes()).hexdigest()==j['sha256']
 subprocess.run(['/tmp/urban_validation_only_sweep',j['config'],str(output)],check=True)
 with output.open() as f:rows=list(csv.DictReader(f))
 assert {r['window'] for r in rows}=={'train','validation'}
 candidates=[r for r in rows if r['window']=='validation' and float(r['detention_fraction'])>0]
 best=max(candidates,key=lambda r:float(r['nse']))
 params={k:(int(v) if k=='lag_steps' else float(v)) for k,v in best.items() if k in ['quick_k','slow_k','detention_k','channel_k','quick_fraction','detention_fraction','runoff_coefficient','lag_steps']}
 selected[j['year']]={'managed':params,'validation_nse':float(best['nse']),'source_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
 print(j['year'],'selected on validation only',flush=True)
(root/'selected_params.json').write_text(json.dumps(selected,indent=2)+'\n')
(root/'provenance.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(Path('/tmp/urban_validation_only_sweep').read_bytes()).hexdigest(),'selection':'Maximum validation NSE among managed candidates; no-detention removes only detention fraction'},indent=2)+'\n')
