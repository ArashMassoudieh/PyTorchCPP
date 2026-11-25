#!/usr/bin/env python3
"""Import exactly one completed fit stopped by the old exact-metric parity guard.
Preserves the failed root; creates a new frozen continuation charging the imported
fit. No retraining, retry, new candidate, goal or budget change is permitted.
"""
import argparse,csv,json,shutil,math
from pathlib import Path
from controller import load,write,sha,prepare,compare_predictions,support,evaluate

def recover(parent,root,old_controller):
 parent=parent.resolve(strict=True);p=load(parent/'plan.json');state=load(parent/'state.json')
 if state.get('status')!='FAILED' or state.get('error')!='Fresh reload/exported metrics differ' or state['charged_fits']!=1 or sha(parent/'plan.json')!=state['plan_sha256']:raise ValueError('Not the specific first-fit roundoff stop')
 controller=Path(__file__).with_name('controller.py').resolve()
 for path,h in p['pins'].items():
  if sha(path)!=h and not (Path(path)==controller and sha(old_controller)==h):raise ValueError('Original pin changed: '+path)
 claims=list((parent/'jobs').glob('*/claim.json'))
 if len(claims)!=1 or claims[0].parent.name!=p['jobs'][0]['id']:raise ValueError('Unexpected consumed inventory')
 if load(claims[0])['plan_sha256']!=state['plan_sha256']:raise ValueError('Original claim/plan binding mismatch')
 j=p['jobs'][0];src=claims[0].parent;bundle=src/'bundles'/j['id'];c=load(j['config'])
 with (bundle/'training_history.csv').open() as f:history=list(csv.DictReader(f))
 if [int(z['epoch']) for z in history]!=list(range(1,p['limits']['epochs']+1)):raise ValueError('Incomplete training history')
 losses=[float(z['validation_loss']) for z in history]
 if any(not math.isfinite(v) or v<0 for v in losses) or [i for i,z in enumerate(history) if z['selected_checkpoint']=='1']!=[min(range(len(losses)),key=lambda i:losses[i])]:raise ValueError('Invalid loss/checkpoint history')
 if 'production_target_free_checks=12' not in (src/'reload.log').read_text():raise ValueError('Missing native verification')
 parity=compare_predictions(bundle/'predictions.csv',src/'validation_predictions.csv');s=support(j['support'],c);scores=evaluate(src/'validation_predictions.csv',j['family'],c,s);evaluate(bundle/'predictions.csv',j['family'],c,s)
 request={k:v for k,v in p.items() if k not in ('pins','schema','no_retries','no_expansion','goal_revision')};request['recovery']=dict(parent=str(parent),parent_plan_sha256=sha(parent/'plan.json'),already_charged_fits=1,remaining_unstarted_jobs=len(p['jobs'])-1,reason='Roundoff-only verification repair; never retrain imported fit')
 request_path=root.with_suffix('.request.json');write(request_path,request,True);prepare(request_path,root)
 dest=root/'jobs'/j['id'];shutil.copytree(src,dest);hashes={str(q.relative_to(dest)):sha(q) for q in dest.rglob('*') if q.is_file()};write(dest/'verified.json',dict(scores=scores,hashes=hashes,reload_parity=parity,imported_from=str(src)),True)
 state=load(root/'state.json');state['charged_fits']=1;write(root/'state.json',state)
 shutil.copy2(old_controller,root/'original_controller.py');write(root/'recovery_review.json',dict(status='VERIFIED_COMPLETED_FIT_IMPORTED_WITHOUT_RETRAINING',parity=parity,original_charged=1,new_total_fit_cap=p['limits']['max_fits'],remaining_unstarted=len(p['jobs'])-1,parent_plan_sha256=sha(parent/'plan.json'),copied_artifacts_byte_identical=all(sha(src/rel)==h for rel,h in hashes.items()),original_failure_preserved=True),True)
 print('Recovered completed fit;8unstarted jobs may proceed under unchanged total9job cap.')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('new_root',type=Path);ap.add_argument('original_controller',type=Path);a=ap.parse_args();recover(a.parent,a.new_root,a.original_controller)
