#!/usr/bin/env python3
"""Fabricated-native CLI policy/reload and validation-Q isolation proof.
Synthetic development checks only, no research observations/fits.
"""
import argparse,csv,hashlib,json,os,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
FAMILIES=('ffn_causal_forecast','lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(binary,verifier,comparer,fixture,output):
 binary,verifier,comparer,fixture,output=[Path(p).absolute() for p in (binary,verifier,comparer,fixture,output)];output.mkdir(exist_ok=False)
 native=fixture/'fabricated_development';manifest=json.loads((native/'native_input_manifest.json').read_text())
 if manifest['row_count']!=220 or manifest['target_cutoff_x']!=220:raise ValueError('Known fabricated220-hour fixture required')
 # Mutate ONLY inner validation Q, preserving every training row byte-for-byte.
 cf=output/'counterfactual_native';shutil.copytree(native,cf)
 lines=(cf/'intervals.csv').read_text().splitlines();out=lines[:1];changed=0
 for row,line in enumerate(lines[1:]):
  fields=line.split(',');epoch=int(fields[0]);target=(epoch-1420070400)//3600+1
  if not 1<=target<=220:raise ValueError('Fixture timestamp outside synthetic scope')
  if target>150:fields[1]=format(float(fields[1])*3,'.17g');line=','.join(fields);changed+=1
  out.append(line)
 (cf/'intervals.csv').write_text('\n'.join(out)+'\n');manifest['interval_csv_sha256']=sha(cf/'intervals.csv');manifest['synthetic_validation_Q_counterfactual_factor']=3
 (cf/'native_input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if changed!=70 or lines[1:151]!=(cf/'intervals.csv').read_text().splitlines()[1:151]:raise ValueError('Counterfactual changed training prefix')
 dispatch=HERE.parents[2]/'build-hydrobatch'/('fixed_epoch_checks_'+hashlib.sha256(str(output).encode()).hexdigest()[:12]);dispatch.mkdir(exist_ok=False)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');commands=[]
 def command(name,args,expected=0,reason=None):
  log=output/(name+'.log')
  with log.open('x') as f:r=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600)
  if r.returncode!=expected or reason and reason not in log.read_text():raise ValueError('CLI policy check failed '+name+' '+log.read_text()[-1500:])
  commands.append(dict(name=name,args=list(map(str,args)),exit_code=r.returncode,log_sha256=sha(log)))
 def fit(name,family,c,expected=0,reason=None,raw=None):
  cp=dispatch/(name+'.json');cp.write_text(raw if raw is not None else json.dumps(c,indent=2)+'\n');bp=dispatch/(name+'.batch');bp.write_text(f'{family} {cp.name}\n');command(name,[binary,bp,output/name],expected,reason);return output/name/name
 last=None
 for family in FAMILIES:
  source=fixture/('native_'+family+'_rate_hold_v1')/('native_'+family+'_rate_hold_v1')/'experiment_config.json';base=json.loads(source.read_text());bundles=[]
  for arm,m in [('original',native/'native_input_manifest.json'),('counterfactual',cf/'native_input_manifest.json')]:
   name=family+'_'+arm;c=dict(base,experiment_id=name,epochs=4,hydrofusion_checkpoint_selection='fixed_last_epoch_v1',hydrofusion_native_manifest_path=str(m),hydrofusion_native_manifest_sha256=sha(m));bundle=fit(name,family,c);bundles.append(bundle);last=c
   exported=json.loads((bundle/'experiment_config.json').read_text())
   if exported['hydrofusion_checkpoint_selection']!='fixed_last_epoch_v1':raise ValueError('Policy export mismatch')
   with (bundle/'training_history.csv').open() as f:history=list(csv.DictReader(f))
   if len(history)!=4 or [int(r['epoch']) for r in history if r['selected_checkpoint']=='1']!=[4]:raise ValueError('Final checkpoint not selected')
   command(name+'_reload',[verifier,bundle,output/(name+'_validation.csv')])
  command(family+'_weights_isolation',[comparer,*bundles])
 # Rehash the public bundle manifest after relabelling the policy: rejection
 # must come from the model's recipe binding, not an incidental stale checksum.
 tamper=output/'tamper_policy';shutil.copytree(bundles[0],tamper)
 cp=tamper/'experiment_config.json';tc=json.loads(cp.read_text());tc['hydrofusion_checkpoint_selection']='best_validation_v1';cp.write_text(json.dumps(tc,indent=2)+'\n')
 ap=tamper/'artifact_manifest.csv';lines=ap.read_text().splitlines();rewritten=lines[:2]
 for line in lines[2:]:
  name,_,_=line.split(',');file=tamper/name;rewritten.append(f'{name},{file.stat().st_size},{sha(file)}')
 ap.write_text('\n'.join(rewritten)+'\n')
 command('tamper_policy_reload',[verifier,tamper,output/'never_written_validation.csv'],1,'checkpoint recipe/scale mismatch')
 for name,policy in [('unknown','bad'),('wrong_type',7),('null',None)]:
  c=dict(last,experiment_id=name,hydrofusion_checkpoint_selection=policy,hydrofusion_native_manifest_path='/nonexistent/never_read.json');fit(name,FAMILIES[-1],c,1)
 c=dict(last,experiment_id='duplicate');raw=json.dumps(c)[:-1]+',"hydrofusion_checkpoint_selection":"best_validation_v1"}';fit('duplicate',FAMILIES[-1],c,1,raw=raw)
 c=json.loads((HERE/'hydrofusion_task4_cli_checks/config_regression_ffn.json').read_text());c.update(experiment_id='legacy_reject',hydrofusion_checkpoint_selection='fixed_last_epoch_v1');fit('legacy_reject','ffn',c,1,'require an extended forecast')
 c=dict(last,experiment_id='physical_reject',hydrofusion_objective='log_mse_v1');fit('physical_reject','pinn_causal_forecast',c,1,'Fixed checkpoint selection requires a neural forecast')
 receipt=dict(status='VERIFIED',fabricated_only=True,binary_sha256=sha(binary),verifier_sha256=sha(verifier),comparer_sha256=sha(comparer),changed_validation_Q_rows=changed,training_prefix_byte_identical=True,commands=commands,weight_isolation_families=4,negative_checks=7)
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS4families: fixed epoch4, fresh reload, validation-Q weight/scaler isolation;7negative checks')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['binary','verifier','comparer','fabricated_fixture','new_output']:p.add_argument(n,type=Path)
 a=p.parse_args();run(a.binary,a.verifier,a.comparer,a.fabricated_fixture,a.new_output)
