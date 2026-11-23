#!/usr/bin/env python3
"""Synthetic SiLU-only preflight, including true activation/reporting and archive binding."""
import argparse,csv,hashlib,json,math,os,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(binary,verifier,native,output):
 binary,verifier,native,output=map(lambda p:Path(p).absolute(),(binary,verifier,native,output));output.mkdir(parents=True,exist_ok=False)
 dispatch=HERE.parents[2]/'build-hydrobatch'/('silu_context_'+hashlib.sha256(str(output).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
 # New trainer changes reporting only; every non-main object and all model
 # headers are byte-identical to the verifier's frozen model/runtime build.
 build=json.loads(Path(str(binary)+'.build/source_manifest.json').read_text());link=json.loads(Path(str(native)+'.link_manifest.json').read_text())
 assert build['exit_code']==0 and build['binary_sha256']==sha(binary)
 for name,h in link['objects'].items():assert build['reused_objects'][name]==h,name
 old=Path(build['reused_from']);assert link['batch_binary_sha256']==sha(old)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');commands=[];bundles={}
 def call(name,args,expected=0,reason=None):
  with (output/(name+'.log')).open('x') as log:p=subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600)
  text=(output/(name+'.log')).read_text()
  if p.returncode!=expected or reason and reason not in text:raise RuntimeError(name+': '+text[-1800:])
  commands.append(dict(name=name,args=list(map(str,args)),exit_code=p.returncode));return text
 for recipe in ['zero25','absolute25','anchored25']:
  name='silu_'+recipe;c=json.loads((HERE/'hydrofusion_five_method_checks/ffn_physical_forecast_log_residual.json').read_text())
  c.update(experiment_id=name,epochs=3,physics_profile='causal_valid_time_dry_v3',hydrofusion_context_recipe=recipe,hydrofusion_history_hours='0,1,3,6,12,23',hydrofusion_development_only=True,hydrofusion_objective='log_flow_mse_v2',hydrofusion_ffn_activation='silu')
  cp=dispatch/(name+'.json');cp.write_text(json.dumps(c,indent=2)+'\n');bp=dispatch/(name+'.batch');bp.write_text('ffn_physical_forecast '+cp.name+'\n')
  text=call(name,[binary,bp,output/name])
  assert 'activation=silu' in text and ' validation_mse=' in text and ' test_mse=' not in text
  summary=next(csv.DictReader((output/name/'batch_summary.csv').open()));assert summary['activation']=='silu' and math.isnan(float(summary['test_mse']))
  bundle=output/name/name;bundles[recipe]=bundle
  call(name+'_reload',[verifier,bundle,'ffn_physical_forecast'])
  # Reporting-only patch must not change TANH numeric outputs.
  c['experiment_id']=name+'_tanh';c['hydrofusion_ffn_activation']='tanh';cp=dispatch/(c['experiment_id']+'.json');cp.write_text(json.dumps(c,indent=2)+'\n');bp=dispatch/(c['experiment_id']+'.batch');bp.write_text('ffn_physical_forecast '+cp.name+'\n')
  call(name+'_new_tanh',[binary,bp,output/(name+'_new_tanh')]);call(name+'_old_tanh',[old,bp,output/(name+'_old_tanh')])
  a=output/(name+'_new_tanh')/c['experiment_id']/'predictions.csv';b=output/(name+'_old_tanh')/c['experiment_id']/'predictions.csv';assert a.read_bytes()==b.read_bytes()
 corrupt=output/'corrupt_activation';shutil.copytree(bundles['anchored25'],corrupt);p=corrupt/'experiment_config.json';c=json.loads(p.read_text());c['hydrofusion_ffn_activation']='tanh';p.write_text(json.dumps(c,indent=2)+'\n')
 manifest=corrupt/'artifact_manifest.csv';lines=manifest.read_text().splitlines();out=lines[:2]
 for line in lines[2:]:
  name,_,_=line.split(',');p=corrupt/name;out.append(f'{name},{p.stat().st_size},{sha(p)}')
 manifest.write_text('\n'.join(out)+'\n');call('activation_binding_rejection',[verifier,corrupt,'ffn_physical_forecast'],1,'recipe/scale mismatch')
 receipt=dict(status='VERIFIED',synthetic_only=True,binary_sha256=sha(binary),native_verifier_sha256=sha(native),verifier_sha256=sha(verifier),commands=commands,
              identical_tanh_predictions=3,verifier_model_objects_identical=True,limitation='Verifier reuses byte-identical model/runtime objects; trainer patch changes CLI/summary reporting only. No native performance claimed.')
 (output/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS SiLU context preflight, activation binding,3identical tanh comparisons')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['binary','verifier','native_verifier','output']:p.add_argument(n,type=Path)
 a=p.parse_args();run(a.binary,a.verifier,a.native_verifier,a.output)
