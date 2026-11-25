#!/usr/bin/env python3
"""Construct a tiny all-five controller smoke request from FABRICATED native data.
Never a research benchmark. Does not download or inspect research observations.
"""
import argparse,json,os,subprocess
from pathlib import Path
from controller import FAMILIES,sha,write,load

def prepare(root,template_dir,binary,verifier,probe):
 root.mkdir(exist_ok=False);configs=root/'configs';configs.mkdir();jobs=[]
 for family in sorted(FAMILIES):
  source=template_dir/((family if family!='pinn_causal_forecast' else 'ffn_causal_forecast')+'_log_flow_mse_v2_original.json');c=load(source)
  # Explicit fixture identity guard prevents accidentally running this test on research data.
  manifest=load(c['hydrofusion_native_manifest_path'])
  if manifest['row_count']!=220 or 'fabricated' not in c['hydrofusion_native_manifest_path']:raise ValueError('Only220hour fabricated fixture accepted')
  c.update(epochs=2,test_end_x=17544,shuffle_training=False,hydrofusion_checkpoint_selection='best_validation_v1',hydrofusion_history_hours='0',hidden_layers='8')
  if family=='pinn_causal_forecast':c['hydrofusion_objective']='log_mse_v1'
  config=configs/(family+'.json');write(config,c,True)
  if not jobs:
   env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
   with (root/'support_preflight.log').open('x') as log:subprocess.run([str(probe),str(config),str(root/'support.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=120)
  jobs.append(dict(id='smoke_'+family,candidate='baseline',family=family,fold='fabricated',seed=42,config=str(config),support=str(root/'support.csv')))
 request=dict(goals=dict(nse=.99,r_squared=.99),limits=dict(epochs=2,wall_seconds_per_fit=120,rss_bytes=12*1024**3,threads=2,gpu=False,artifact_bytes_per_fit=250000000,max_fits=5),rule=dict(min_nse_gain=.02,max_mae_ratio=1.02,max_high_rmse_ratio=1.02,max_abs_bias_increase=2,required_fraction=1),binary=str(binary),verifier=str(verifier),jobs=jobs)
 write(root/'request.json',request,True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('template_dir',type=Path);p.add_argument('binary',type=Path);p.add_argument('verifier',type=Path);p.add_argument('probe',type=Path);a=p.parse_args();prepare(a.root.absolute(),a.template_dir.absolute(),a.binary.absolute(),a.verifier.absolute(),a.probe.absolute())
