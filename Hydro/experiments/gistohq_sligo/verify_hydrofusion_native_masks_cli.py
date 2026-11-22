#!/usr/bin/env python3
"""All-family fabricated independent-mask CLI/reload and negative checks."""
import argparse,csv,hashlib,json,os,shutil,subprocess
from pathlib import Path
from verify_hydrofusion_reference_et_cli import NAMES
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
HERE=Path(__file__).resolve().parent

def run(binary,verifier,source,output):
 output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
 folder=output/'masked_input';shutil.copytree(source,folder)
 csvpath=folder/'intervals.csv'
 with csvpath.open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 for i,r in enumerate(rows):r.update(rainfall_valid='0' if i==100 else r['forcing_valid'],common_origin_forcing_available='0' if i==101 else '1')
 with csvpath.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields+['rainfall_valid','common_origin_forcing_available'],lineterminator='\n');w.writeheader();w.writerows(rows)
 manifest=folder/'native_input_manifest.json';m=json.loads(manifest.read_text());m.update(interval_mask_schema='independent_rain_and_common_origin_v1',interval_csv_sha256=sha(csvpath));manifest.write_text(json.dumps(m,indent=2)+'\n')
 dispatch=HERE.parents[2]/'build-hydrobatch'/('native_masks_'+sha(manifest)[:12]);dispatch.mkdir(exist_ok=False)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');receipt=[]
 def command(name,args,code=0,reason=None):
  with (output/(name+'.log')).open('x') as f:p=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600)
  text=(output/(name+'.log')).read_text()
  if p.returncode!=code or reason and reason not in text:raise RuntimeError(name+': '+text[-2000:])
  receipt.append(dict(name=name,exit_code=p.returncode))
 jobs=[]
 for name in NAMES:
  c=json.loads((HERE/'hydrofusion_five_method_checks'/(name+'.json')).read_text());c.update(epochs=3,hydrofusion_synthetic_verification=False,hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=sha(manifest))
  if not name.startswith('pinn_'):c['hydrofusion_objective']='log_flow_mse_v2'
  if 'physical' in name or name.startswith('pinn_'):c['physics_profile']='causal_valid_time_dry_v3'
  path=dispatch/(name+'.json');path.write_text(json.dumps(c));jobs.append(name.removesuffix('_direct_log_q').removesuffix('_log_residual')+' '+path.name)
 batch=dispatch/'jobs.batch';batch.write_text('\n'.join(jobs)+'\n');command('all_families_cli',[binary,batch,output/'bundles'])
 for name in NAMES:command(name+'_reload',[verifier,output/'bundles'/name,name.removesuffix('_direct_log_q').removesuffix('_log_residual')])
 for tag,key,value,reason in [('bad_schema','interval_mask_schema','unknown','Versioned independent native interval masks'),('downgrade','schema','HydroFusionPreparedNative-v2','Versioned independent native interval masks')]:
  p=output/tag;shutil.copytree(folder,p);d=dict(m);d[key]=value;mp=p/'native_input_manifest.json';mp.write_text(json.dumps(d));c=json.loads((dispatch/(NAMES[0]+'.json')).read_text());c.update(experiment_id=tag,hydrofusion_native_manifest_path=str(mp),hydrofusion_native_manifest_sha256=sha(mp));cp=dispatch/(tag+'.json');cp.write_text(json.dumps(c));bp=dispatch/(tag+'.batch');bp.write_text('ffn_causal_forecast '+cp.name+'\n');command(tag,[binary,bp,output/(tag+'_out')],1,reason)
 tamper=output/'tampered';shutil.copytree(output/'bundles'/NAMES[0],tamper);p=tamper/'intervals.csv';p.write_bytes(p.read_bytes()+b'junk\n');command('tampered_interval_rejected',[verifier,tamper,'ffn_causal_forecast'],1,'Artifact bundle integrity check failed: intervals.csv')
 shutil.copytree(dispatch,output/'configs');(output/'acceptance.json').write_text(json.dumps(dict(binary_sha256=sha(binary),verifier_sha256=sha(verifier),commands=receipt,synthetic_only=True),indent=2)+'\n');print('PASS7masked CLI/fresh-reload variants plus3rejections')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['binary','verifier','source','output']:p.add_argument(n,type=Path)
 a=p.parse_args();run(a.binary,a.verifier,a.source,a.output)
