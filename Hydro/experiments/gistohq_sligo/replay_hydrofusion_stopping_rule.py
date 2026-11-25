#!/usr/bin/env python3
"""One frozen causal stopping-rule replay; oracle loss is reporting only."""
import argparse,csv,hashlib,json,math,itertools
from pathlib import Path
RULE=dict(min_epochs=10,patience=5,max_epochs=50,min_delta=0.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def select(losses):
 best=float('inf');best_epoch=0;stopped=0;wait=0;triggered=False
 for epoch,value in enumerate(itertools.islice(losses,RULE['max_epochs']),1):
  value=float(value)
  if not math.isfinite(value) or value<0:raise ValueError('Nonfinite/negative observed objective')
  if value<best-RULE['min_delta']:best=value;best_epoch=epoch;wait=0
  else:wait+=1
  stopped=epoch
  if epoch>=RULE['min_epochs'] and wait>=RULE['patience']:triggered=True;break
 if stopped<RULE['min_epochs']:raise ValueError('Insufficient observed epochs')
 return dict(stop_epoch=stopped,selected_epoch=best_epoch,selected_loss=best,patience_triggered=triggered)
def replay(root):
 root=Path(root);proposal=json.loads((root/'replay_plan.json').read_text());source=Path(proposal['source_campaign']);plan=json.loads((source/'screen_plan.json').read_text());review=json.loads((source/'independent_final_check.json').read_text())
 if any(proposal['rule'][k]!=v for k,v in RULE.items()) or proposal['evaluation']!=dict(max_relative_objective_regret=.02,all_six_required=True):raise ValueError('Frozen replay rule changed')
 if review['status']!='COMPLETE_INDEPENDENTLY_VERIFIED_TRAIN_ONLY' or sha(source/'screen_plan.json')!=proposal['source_plan_sha256'] or len(plan['jobs'])!=6:raise ValueError('Closed TRAIN-only source required')
 for p,h in plan['pins'].items():
  if sha(p)!=h:raise ValueError('Source pin changed '+p)
 result={};pins={}
 for j in plan['jobs']:
  fit=source/'fits'/j['name'];rec=json.loads((fit/'completion.json').read_text());c=json.loads(Path(j['config']).read_text())
  if not 0<c['train_end_x']<c['validation_end_x']<=17543 or not c['hydrofusion_development_only']:raise ValueError('Outer chronology forbidden')
  history=fit/'bundles'/j['name']/'training_history.csv'
  if sha(history)!=rec['sha256'][str(history.relative_to(fit))]:raise ValueError('History changed')
  with history.open() as f:rows=list(csv.DictReader(f))
  if [int(z['epoch']) for z in rows]!=list(range(1,51)):raise ValueError('Epoch identities')
  losses=[float(z['validation_loss']) for z in rows]
  # Selection consumes only the prefix. The following full-history oracle is
  # a separate retrospective diagnostic, never fed back into select().
  chosen=select(iter(losses))
  if any(not math.isfinite(v) or v<0 for v in losses):raise ValueError('Invalid oracle history')
  oracle_epoch=min(range(50),key=lambda i:losses[i])+1;oracle=losses[oracle_epoch-1]
  absolute=chosen['selected_loss']-oracle;relative=absolute/oracle if oracle else 0. if absolute==0 else None
  within=absolute<=.02*oracle
  result[j['name']]=dict(chosen,oracle_epoch=oracle_epoch,oracle_loss=oracle,absolute_objective_regret=absolute,relative_objective_regret=relative,within2percent=within,saved_epochs=50-chosen['stop_epoch'],family=j['family'],fold=j['fold'])
  pins[str(history)]=sha(history);pins[str(fit/'completion.json')]=sha(fit/'completion.json')
 passed=all(v['within2percent'] for v in result.values())
 return dict(status='REPLAY_DIAGNOSTIC_PASS' if passed else 'REPLAY_DIAGNOSTIC_FAIL',models=result,rule=proposal['rule'],rule_plan_sha256=sha(root/'replay_plan.json'),source_pins=pins,all_six_pass=passed,total_replayed_epochs=sum(v['stop_epoch'] for v in result.values()),maximum_epochs=300,no_real_compute_saving_claim=True,new_fits=0,test_rows=0,outer2017_rows=0,limitations=proposal['limitations'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('replay_root',type=Path);p.add_argument('new_output',type=Path);a=p.parse_args();r=replay(a.replay_root)
 with a.new_output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(r,indent=2))
