#!/usr/bin/env python3
"""Independent scalar-math check of the closed train-only calibration receipts."""
import csv, hashlib, json, math, sys
from pathlib import Path

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,p):
 n=len(y);ym=math.fsum(y)/n;pm=math.fsum(p)/n
 yy=math.fsum((v-ym)**2 for v in y);pp=math.fsum((v-pm)**2 for v in p)
 return dict(n=n,nse=1-math.fsum((a-b)**2 for a,b in zip(y,p))/yy,
  r_squared=math.fsum((a-ym)*(b-pm) for a,b in zip(y,p))**2/(yy*pp),
  mae=math.fsum(abs(a-b) for a,b in zip(y,p))/n,pbias_percent=100*math.fsum(b-a for a,b in zip(y,p))/math.fsum(y))
def read(p):
 with p.open() as f:r=list(csv.DictReader(f))
 x=[int(v['target_x']) for v in r]
 if len(set(x))!=len(x) or sorted(x)!=x:raise ValueError('Identity drift')
 return x,*[[float(v[k]) for v in r] for k in ['observed','wet_unit_gain','dry_unit_gain']]
def check(expected,actual):
 for k,v in expected.items():
  if k in actual and not math.isclose(v,actual[k],rel_tol=1e-10,abs_tol=1e-11):raise ValueError('Metric mismatch '+k)
def verify(root):
 plan=json.loads((root/'plan.json').read_text());sel=json.loads((root/'training_selection.json').read_text());result=json.loads((root/'result.json').read_text())
 for p,h in plan['hashes'].items():
  # Saved source snapshot remains authoritative if later development changes live code.
  saved=root/'provenance'/Path(p).name
  if digest(saved if saved.exists() else p)!=h:raise ValueError('Pinned input/source mismatch '+p)
 for name,h in result['artifacts'].items():
  if digest(root/name)!=h:raise ValueError('Artifact changed')
 if digest(root/'training_selection.json')!=result['selection_sha256']:raise ValueError('Selection changed')
 x,y,p,d=read(root/'train_default.csv');check(sel['default'],metrics(y,p))
 for c in sel['candidates']:
  xx,yy,pp,dd=read(root/('train_default.csv' if c['params']==[1.,.1,.7] else f"train_{c['index']:02d}.csv"))
  if xx!=x or yy!=y or max(xx)>17543:raise ValueError('Training support drift')
  g=max(0.,min(1.,math.fsum(a*b for a,b in zip(y,pp))/math.fsum(a*a for a in pp)))
  if abs(g-c['gain'])>1e-12:raise ValueError('Gain mismatch')
  m=metrics(y,[g*a for a in pp]);check(c['metrics'],m)
  eligible=m['mae']<=sel['default']['mae']*1.02 and abs(m['pbias_percent'])<=abs(sel['default']['pbias_percent'])+5
  if eligible!=c['eligible']:raise ValueError('Eligibility mismatch')
 best=max((c for c in sel['candidates'] if c['eligible']),key=lambda c:(c['metrics']['nse'],-c['index']))
 if best!=sel['selected'] or best!=result['selected']:raise ValueError('Selection mismatch')
 vx,vy,vp,vd=read(root/'validation_selected.csv');bx,by,bp,bd=read(root/'validation_default.csv')
 if vx!=bx or vy!=by or min(vx)<=17543 or max(vx)>26303:raise ValueError('Validation support drift')
 for name,pred in [('validation_default_wet',bp),('validation_default_dry',bd),('validation_selected',[best['gain']*v for v in vp])]:check(result[name],metrics(vy,pred))
 dry=result['validation_default_dry']; candidate=result['validation_selected']
 gate=candidate['nse']>=dry['nse']+.005 and candidate['mae']<=dry['mae']*1.02 and abs(candidate['pbias_percent'])<=abs(dry['pbias_percent'])+5
 if gate!=result['gate_pass']:raise ValueError('Gate mismatch')
 return dict(status='INDEPENDENTLY_VERIFIED',train_rows=len(x),validation_rows=len(vx),test_rows=0,candidates=27,selected=best,gate_pass=gate)
if __name__=='__main__':
 root=Path(sys.argv[1]);v=verify(root)
 with (root/'independent_review.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
 print('PASS independent hashes/scalar metrics/train-only selection/support')
