#!/usr/bin/env python3
"""Select fixed-grid physics on validation and apply a frozen stop gate; no fitting."""
import argparse,csv,hashlib,json,math
from pathlib import Path
FIELDS=['index','train_nse','train_mae','train_pbias_percent','validation_nse','validation_mae','validation_pbias_percent']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def select(proposal_path,scores_path):
 p=json.loads(Path(proposal_path).read_text())
 if p['schema']!='CalibratedTimingV3BatchProposal-v1' or len(p['candidates'])!=24:raise ValueError('Wrong proposal scope')
 for path,h in p['pins'].items():
  if sha(path)!=h:raise ValueError('Changed proposal input')
 for c in p['candidates']:
  if sha(c['config'])!=c['sha256']:raise ValueError('Candidate config changed')
 with Path(scores_path).open() as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=FIELDS:raise ValueError('Only declared train/validation scores allowed')
  rows=list(reader)
 if len(rows)!=24:raise ValueError('Exactly24candidate scores required')
 normalized=[]
 for i,row in enumerate(rows):
  if row['index']!=str(i):raise ValueError('Candidate identities/order changed')
  v={k:float(row[k]) for k in FIELDS[1:]}
  if not all(math.isfinite(x) for x in v.values()) or v['train_mae']<0 or v['validation_mae']<0:raise ValueError('Invalid candidate metric')
  normalized.append(v)
 selected=max(range(24),key=lambda i:normalized[i]['validation_nse'])
 m=normalized[selected];g=p['stage_a_gate']
 passed=dict(nse=m['validation_nse']>=g['minimum_validation_nse'],bias=abs(m['validation_pbias_percent'])<=g['maximum_absolute_validation_pbias_percent'],mae=m['validation_mae']<=g['reference_validation_mae'])
 return dict(status='GATE_PASSED_REQUIRES_FROZEN_EXECUTION_PLAN' if all(passed.values()) else 'STOP_NO_NEURAL_FITS',selected_candidate_index=selected,selected_config=p['candidates'][selected]['config'],metrics=m,checks=passed,proposal_sha256=sha(proposal_path),scores_sha256=sha(scores_path),selection_conditioned_validation=True,training_launched=False)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('proposal','scores','output'):parser.add_argument(name,type=Path)
 args=parser.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 result=select(args.proposal,args.scores)
 with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(result['status'])
