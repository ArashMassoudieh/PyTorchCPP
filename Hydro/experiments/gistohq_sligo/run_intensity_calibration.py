"""Calibrate the prospectively bounded intensity-response arms on validation only."""
from pathlib import Path
import json,csv,subprocess,hashlib
base=Path(__file__).resolve().parent;root=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023');out=root/'intensity_calibration';out.mkdir(exist_ok=True);configs=base/'improvement_loop_20261023/intensity_calibration_configs';configs.mkdir(exist_ok=True);selected={}
proof=json.load(open(root/'intensity_smoke/verification.json'));assert all(proof[k] for k in ['zero_predictions_identical','response_changes_predictions','invalid_strength_rejected','exports_verified'])
for year in [2018,2020,2023]:
 selected[str(year)]={}
 for strength in [0.,.5,1.]:
  tag=str(strength).replace('.','p');c=json.load(open(base/f'improvement_loop_20261023/six_year_physics_calibration_configs/{year}.json'));c['urban_intensity_quick_strength']=strength;p=configs/f'{year}_{tag}.json';assert not p.exists();p.write_text(json.dumps(c,indent=2));csvpath=out/f'{year}_{tag}.csv';assert not csvpath.exists()
  subprocess.run(['/tmp/urban_intensity_validation_sweep',str(p),str(csvpath)],check=True)
  rows=list(csv.DictReader(csvpath.open()));assert {r['window'] for r in rows}=={'train','validation'}
  candidates=[r for r in rows if r['window']=='validation' and float(r['detention_fraction'])>0];best=max(candidates,key=lambda r:float(r['nse']));selected[str(year)][tag]={k:float(v) for k,v in best.items() if k!='window'};print(year,strength,'selected',flush=True)
(out/'selected_params.json').write_text(json.dumps(selected,indent=2));(out/'provenance.json').write_text(json.dumps(dict(sweep_binary_sha256=hashlib.sha256(Path('/tmp/urban_intensity_validation_sweep').read_bytes()).hexdigest(),reference='90th percentile of positive training rainfall; floor rank',selection='maximum validation NSE, equal physical grid for each strength'),indent=2))
