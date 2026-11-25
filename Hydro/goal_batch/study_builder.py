#!/usr/bin/env python3
"""Build a bounded calendar-study request from verified native/weather inputs.
Only the existing 2015-origin native time convention is supported. No acquisition,
reserved-data access, fitting, manual JSON or silent dataset replacement.
"""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
from controller import FAMILIES,load,write,sha,goals,integer,validate_request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/gistohq_sligo'))
from prepare_hydrofusion_search_slice import prepare as crop
EPOCH=1420070400

def boundaries(years,train_months):
 integer(years,1,10);integer(train_months,1,11)
 end=int((datetime.datetime(2015+years,1,1,tzinfo=datetime.timezone.utc).timestamp()-EPOCH)/3600)-1
 if end>=77016:raise ValueError('RESERVED_DATA_GATE: requested period reaches reserved observations; separate explicit permission and verified coverage are required. Nothing was read or fitted.')
 train=int((datetime.datetime(2015,train_months+1,1,tzinfo=datetime.timezone.utc).timestamp()-EPOCH)/3600)-1
 return train,end

def build(root,manifest,weather,binary,verifier,probe,years=1,train_months=9,epochs=200,widths=(32,64),families=None,nse=.7,r_squared=.7):
 train,end=boundaries(years,train_months);integer(epochs,1,1000);goals(dict(nse=nse,r_squared=r_squared))
 if end>=17544:raise ValueError('FEATURE_PREFLIGHT_PERIOD_GATE: current verified adapter supports one or two years only')
 families=sorted(FAMILIES if families is None else families)
 if not families or len(set(families))!=len(families) or not set(families)<=FAMILIES:raise ValueError('Select supported methods')
 if not widths or len(set(widths))!=len(widths) or any(w not in (32,64) for w in widths):raise ValueError('Supported finite width inventory is32and64')
 manifest,weather,binary,verifier,probe=[Path(p).resolve(strict=True) for p in (manifest,weather,binary,verifier,probe)]
 meta=load(manifest)
 if meta.get('first_interval_x')!=0 or meta.get('end_interval_x_exclusive',0)<end or meta.get('end_interval_x_exclusive',77016)>=77016:raise ValueError('COVERAGE_GATE: verified presealed native package from2015does not cover selected period')
 if meta.get('schema') not in ('HydroFusionPreparedNative-v2','HydroFusionPreparedNative-v3') or meta.get('catchment_area_m2',0)<=0:raise ValueError('Verified native catchment package required; no legacy-area substitution')
 parent_provenance=weather.parent/'preissue_rain_provenance.json';provenance=load(parent_provenance)
 root=Path(root).absolute();root.mkdir(exist_ok=False);native=crop(manifest,root/'native',end);(root/'weather').mkdir();(root/'configs').mkdir()
 target= root/'weather/preissue_rain.csv';last=0
 with weather.open() as inp,target.open('x') as out:
  out.write(next(inp))
  for line in inp:
   token=line.partition(',')[0]
   if not token.isdecimal():raise ValueError('Weather identity must be integer')
   x=int(token)
   if x!=last+1:raise ValueError('Weather chronology gap; missing remains missing')
   if x>end:break
   out.write(line);last=x
 if last!=end:raise ValueError('COVERAGE_GATE: weather sidecar does not span study')
 weather_meta=root/'weather/preissue_rain_provenance.json';write(weather_meta,dict(schema='GoalStudyWeatherPrefix-v1',prefix_target_end_x=end,parent_csv_sha256=sha(weather),parent_provenance_sha256=sha(parent_provenance),parent_provenance=provenance),True)
 base=load(Path(__file__).with_name('default_forecast.json'));jobs=[]
 # Shared support is derived with the real production feature assembler, never guessed from dates.
 support=root/'support.csv';probe_config=dict(base,epochs=epochs,train_end_x=train,validation_end_x=end,test_end_x=17544,hydrofusion_native_manifest_path=str(native),hydrofusion_native_manifest_sha256=sha(native),hydrofusion_preissue_rain_path=str(target),hydrofusion_preissue_rain_sha256=sha(target),hydrofusion_preissue_rain_provenance_sha256=sha(weather_meta))
 probe_file=root/'feature_preflight_config.json';write(probe_file,probe_config,True)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
 with (root/'feature_preflight.log').open('x') as log:subprocess.run([str(probe),str(probe_file),str(support)],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=300)
 for family in families:
  for width in ((32,) if family=='pinn_causal_forecast' else widths):
   name=f'y{years}_{family}_h{width}_s42';c=dict(probe_config,experiment_id=name,hidden_layers=str(width),shuffle_training=False,hydrofusion_checkpoint_selection='best_validation_v1',hydrofusion_context_recipe='legacy20' if family=='pinn_causal_forecast' else 'anchored25' if 'physical' in family else 'zero25',hydrofusion_history_hours='0,1,3,6,12,23' if family.startswith('ffn') else '0',hydrofusion_objective='log_mse_v1' if family=='pinn_causal_forecast' else 'log_flow_mse_v2')
   cp=root/'configs'/(name+'.json');write(cp,c,True);jobs.append(dict(id=name,candidate=f'width{width}',family=family,fold=f'first{years}years',seed=42,config=str(cp),support=str(support)))
 request=dict(goals=dict(nse=nse,r_squared=r_squared),limits=dict(epochs=epochs,wall_seconds_per_fit=28800,rss_bytes=12*1024**3,threads=2,gpu=False,artifact_bytes_per_fit=250000000,max_fits=len(jobs)),rule=dict(min_nse_gain=.02,max_mae_ratio=1.02,max_high_rmse_ratio=1.02,max_abs_bias_increase=2,required_fraction=1),binary=str(binary),verifier=str(verifier),jobs=jobs,study=dict(years=years,train_end_x=train,validation_end_x=end,calendar_origin='2015-01-01T00:00:00Z',seed_count=1,exploratory=True))
 validate_request(request);write(root/'request.json',request,True);write(root/'study_preflight.json',dict(status='READY_FOR_BOUNDED_SEARCH',request_sha256=sha(root/'request.json'),builder_sha256=sha(__file__),template_sha256=sha(Path(__file__).with_name('default_forecast.json')),probe_sha256=sha(probe),source_manifest_sha256=sha(manifest),source_weather_sha256=sha(weather),new_fits=0,warning='One seed and reused development support; goals are not guarantees. Archive rainfall timestamps are retrospective proxies, not operational delivery proof.'),True)
 print(json.dumps(dict(status='PREPARED',request=str(root/'request.json'),jobs=len(jobs),epochs=epochs,train_end_x=train,validation_end_x=end)),flush=True)
 return root/'request.json'

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
 for k in ('manifest','weather','binary','verifier','probe'):p.add_argument('--'+k,required=True,type=Path)
 p.add_argument('--years',type=int,default=1);p.add_argument('--train-months',type=int,default=9);p.add_argument('--epochs',type=int,default=200);p.add_argument('--widths',default='32,64');p.add_argument('--families',default=','.join(sorted(FAMILIES)));p.add_argument('--nse',type=float,default=.7);p.add_argument('--r-squared',type=float,default=.7);a=p.parse_args()
 try:build(a.root,a.manifest,a.weather,a.binary,a.verifier,a.probe,a.years,a.train_months,a.epochs,tuple(int(v) for v in a.widths.split(',')),a.families.split(','),a.nse,a.r_squared)
 except Exception as e:print('BLOCKED: '+str(e),file=sys.stderr);sys.exit(1)
