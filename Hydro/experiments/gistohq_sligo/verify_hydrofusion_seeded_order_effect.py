#!/usr/bin/env python3
"""Reuse completed shuffled synthetic runs; prove the flag changes optimization.
Only additional chronological controls are fitted, on the same fabricated data.
Different predictions are mode-activation evidence, NOT skill improvement.
"""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(binary,verifier,comparer,accepted,output):
 binary,verifier,comparer,accepted,output=[Path(p).absolute() for p in (binary,verifier,comparer,accepted,output)]
 r=json.loads((accepted/'receipt.json').read_text())
 if r['status']!='VERIFIED' or r['binary_sha256']!=sha(binary):raise ValueError('Completed same-build synthetic preflight required')
 output.mkdir(exist_ok=False);dispatch=HERE.parents[2]/'build-hydrobatch'/('seeded_effect_'+sha(accepted/'receipt.json')[:12]);dispatch.mkdir(exist_ok=False)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');done=[]
 def command(name,args,expected=0,reason=None):
  log=output/(name+'.log')
  with log.open('x') as f:p=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600)
  if p.returncode!=expected or reason and reason not in log.read_text():raise ValueError('Synthetic mode-effect check failed '+name)
 for family in ['ffn_causal_forecast','lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast']:
  for objective in ['log_mse_v1','log_flow_mse_v2']:
   oldname=family+'_'+objective+'_original';shuffled=accepted/oldname/oldname;original=json.loads((shuffled/'experiment_config.json').read_text());name=family+'_'+objective+'_chronological'
   c=dict(original,shuffle_training=False,experiment_id=name);cp=dispatch/(name+'.json');bp=dispatch/(name+'.batch');cp.write_text(json.dumps(c,indent=2)+'\n');bp.write_text(f'{family} {cp.name}\n')
   command(name,[binary,bp,output/name]);bundle=output/name/name
   command(name+'_reload',[verifier,bundle,output/(name+'_validation.csv')])
   if (bundle/'scalers.csv').read_bytes()!=(shuffled/'scalers.csv').read_bytes():raise ValueError('Training order changed canonical scalers')
   if (bundle/'predictions.csv').read_bytes()==(shuffled/'predictions.csv').read_bytes():raise ValueError('Synthetic mode produced no prediction change')
   command(name+'_different_weights',[comparer,bundle,shuffled],1,'Validation changed weights')
   done.append(dict(family=family,objective=objective,canonical_scalers_identical=True,weights_differ=True,predictions_differ=True,chronological_predictions_sha256=sha(bundle/'predictions.csv'),shuffled_predictions_sha256=sha(shuffled/'predictions.csv')))
 receipt=dict(status='VERIFIED_MODE_EFFECT_NOT_SKILL',cases=done,fabricated_only=True,additional_chronological_controls=8,reused_shuffled_cases=8,source_receipt_sha256=sha(accepted/'receipt.json'),binary_sha256=sha(binary))
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS8mode-effect controls, canonical scalers identical; no skill claim')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['binary','verifier','comparer','completed_synthetic_preflight','new_output']:p.add_argument(n,type=Path)
 a=p.parse_args();verify(a.binary,a.verifier,a.comparer,a.completed_synthetic_preflight,a.new_output)
