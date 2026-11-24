#!/usr/bin/env python3
"""Verify the closed component screen using KKT certificates, not its optimizer."""
import json,sys
from pathlib import Path
import numpy as np
from run_preissue_physical_calibration import load,sha
from verify_preissue_physical_calibration import metrics,check

def verify(root):
 p=json.loads((root/'plan.json').read_text());s=json.loads((root/'training_selection.json').read_text());r=json.loads((root/'completion.json').read_text())
 for name,h in {**p['input_sha256'],**p['source_sha256']}.items():
  saved=root/'provenance'/Path(name).name
  if sha(saved if saved.exists() else name)!=h:raise ValueError('Hash mismatch')
 if len(s['candidates'])!=54 or sha(root/'training_selection.json')!=r['selection_sha256']:raise ValueError('Selection changed')
 old=Path(next(name for name in p['input_sha256'] if name.endswith('training_selection.json'))).parent
 x,y,w,d=load(old/'train_default.csv');baseline=metrics(y,d);check(s['training_default_dry'],baseline)
 for c in s['candidates']:
  xx,yy,wet,dry=load(old/('train_default.csv' if c['params']==[1.,.1,.7] else f"train_{c['shape_index']:02d}.csv"))
  if not np.array_equal(x,xx) or not np.array_equal(y,yy) or x.max()>17543:raise ValueError('Training drift')
  columns=[dry,wet-dry];bounds=[1,1]
  if c['background']:columns.append(np.ones_like(dry));bounds.append(s['background_upper_mm_h'])
  a=np.column_stack(columns);coef=np.array(c['coefficients']);u=np.array(bounds)
  if not np.isfinite(coef).all() or (coef < -1e-12).any() or (coef>u+1e-12).any():raise ValueError('Invalid coefficient domain')
  prediction=a@coef;gradient=a.T@(prediction-y)
  # First-order KKT conditions are sufficient for this convex box problem.
  tolerance=1e-7*max(1.,float(np.max(np.abs(a.T@y))))
  for i,g in enumerate(gradient):
   if coef[i]<=1e-10:
    if g < -tolerance:raise ValueError('Lower-bound KKT failed')
   elif coef[i]>=u[i]-1e-10:
    if g > tolerance:raise ValueError('Upper-bound KKT failed')
   elif abs(g)>tolerance:raise ValueError('Free-variable KKT failed')
  m=metrics(y,prediction);check(c['metrics'],m)
  if (m['mae']<=baseline['mae']*1.02 and abs(m['pbias_percent'])<=20 and m['nse']>=baseline['nse']+.005)!=c['eligible']:raise ValueError('Training gate failed')
 best=max((c for c in s['candidates'] if c['eligible']),key=lambda c:(c['metrics']['nse'],-int(c['background']),-c['shape_index']))
 if best!=s['selected']:raise ValueError('Wrong selection')
 if sha(root/'validation_components.csv')!=r['validation_components_sha256'] or sha(root/'validation_predictions.csv')!=r['validation_predictions_sha256']:raise ValueError('Validation changed')
 vx,vy,vw,vd=load(root/'validation_components.csv');bx,by,bw,bd=load(old/'validation_default.csv')
 if not np.array_equal(vx,bx) or not np.array_equal(vy,by):raise ValueError('Validation drift')
 columns=[vd,vw-vd]
 if best['background']:columns.append(np.ones_like(vd))
 prediction=np.column_stack(columns)@np.array(best['coefficients']);b=metrics(vy,bd);c=metrics(vy,prediction)
 high=vy>s['training_q90']
 for m,values in [(b,bd),(c,prediction)]:m['high_flow_rmse']=float(np.sqrt(np.mean((values[high]-vy[high])**2)))
 check(r['validation_baseline'],b);check(r['validation_candidate'],c)
 gates=dict(nse=c['nse']>=b['nse']+.02,r_squared=c['r_squared']>=b['r_squared']+.01,mae=c['mae']<=b['mae']*1.02,high_flow=c['high_flow_rmse']<=b['high_flow_rmse']*1.02,bias=abs(c['pbias_percent'])<=abs(b['pbias_percent'])+2)
 if gates!=r['gates'] or all(gates.values())!=r['advanced']:raise ValueError('Final gate mismatch')
 return dict(status='INDEPENDENTLY_VERIFIED_NOT_ADVANCED',candidates=54,kkt_certificates=54,training_rows=len(x),validation_rows=len(vx),test_rows=0,validation_candidate=c,gates=gates)
if __name__=='__main__':
 root=Path(sys.argv[1]);report=verify(root)
 with (root/'independent_review.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print('PASS54 KKT certificates, hashes, selection,metrics,gates')
