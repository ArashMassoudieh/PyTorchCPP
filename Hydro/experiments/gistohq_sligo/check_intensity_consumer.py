from pathlib import Path
import subprocess,json,hashlib
base=Path(__file__).resolve().parent;repo=base.parents[2];r=Path('/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/intensity_smoke');r.mkdir(exist_ok=True);b=base/'improvement_loop_20261023/intensity_smoke_configs';b.mkdir(exist_ok=True)
t=json.load(open(base/'improvement_loop_20261023/six_year_physics_configs/sixyear_2018_urban_no_detention_s42.json'))
for arm,w in [('zero',0.),('response',1.),('invalid',1.1)]:
 c=dict(t,experiment_id='intensity_smoke_'+arm,epochs=1,shared_scale_volume_weight=0.,urban_intensity_quick_strength=w,urban_intensity_reference_mm_h=10.);p=b/(arm+'.json');assert not p.exists();p.write_text(json.dumps(c,indent=2));(b/(arm+'.batch')).write_text(f'lstm_pinn {p}\n')
for name,binary,arm in [('old','/tmp/HydroBatch_pre_intensity_2ecd1d','zero'),('new',str(repo/'build-hydrobatch/HydroBatch'),'zero'),('response',str(repo/'build-hydrobatch/HydroBatch'),'response'),('invalid',str(repo/'build-hydrobatch/HydroBatch'),'invalid')]:
 with (r/(name+'.log')).open('w') as log:p=subprocess.run([binary,str(b/(arm+'.batch')),str(r/name)],stdout=log,stderr=subprocess.STDOUT)
 if name=='invalid':assert p.returncode!=0;continue
 assert p.returncode==0,(name,p.returncode);print(name,'complete',flush=True)
old=(r/'old/intensity_smoke_zero/predictions.csv').read_bytes();new=(r/'new/intensity_smoke_zero/predictions.csv').read_bytes();response=(r/'response/intensity_smoke_response/predictions.csv').read_bytes();assert old==new;assert new!=response
for arm,name in [('zero','new'),('response','response')]:
 c=json.load(open(r/name/('intensity_smoke_'+arm)/'experiment_config.json'));assert c['urban_intensity_quick_strength']==(0 if arm=='zero' else 1);assert c['urban_intensity_reference_mm_h']==10
(r/'verification.json').write_text(json.dumps(dict(zero_predictions_identical=True,response_changes_predictions=True,invalid_strength_rejected=True,exports_verified=True,binary_sha256=hashlib.sha256((repo/'build-hydrobatch/HydroBatch').read_bytes()).hexdigest()),indent=2));print('passed')
