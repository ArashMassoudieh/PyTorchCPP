#!/usr/bin/env python3
"""Nondeployable timing-error diagnostic. Positive offsets use FUTURE predictions.
No shifted series is an admissible forecast or evidence of achieved skill.
"""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(root,output):
 root=Path(root);output=Path(output)
 if output.exists():raise FileExistsError(output)
 plan=json.loads((root/'screen_plan.json').read_text());claim=json.loads((root/'launch_claim.json').read_text())
 assert claim['plan_sha256']==sha(root/'screen_plan.json')
 result={}
 for j in plan['jobs']:
  fit=root/'fits'/j['name'];receipt=fit/'completion.json'
  if not receipt.exists():continue
  rec=json.loads(receipt.read_text())
  for f,h in rec['sha256'].items():assert sha(fit/f)==h
  data={}
  with (fit/'validation_predictions.csv').open() as handle:
   for r in csv.DictReader(handle):
    t=float(r['x']);assert math.isfinite(t) and t.is_integer() and 17543<t<=26303 and r['split']=='validation'
    x=int(t);assert x not in data
    o,p=float(r['observed']),float(r['predicted']);assert all(math.isfinite(v) and v>=0 for v in [o,p]);data[x]=(o,p)
  # Identical complete support for EVERY offset; forbid bridging missing hours.
  support=[x for x in sorted(data) if all(x+k in data for k in range(-3,4))]
  obs=[data[x][0] for x in support];mu=statistics.fmean(obs);sst=math.fsum((v-mu)**2 for v in obs)
  scores={}
  for k in range(-3,4):
   pred=[data[x+k][1] for x in support];sse=math.fsum((a-b)**2 for a,b in zip(obs,pred))
   scores[str(k)]=dict(nse=1-sse/sst,r_squared=statistics.correlation(obs,pred)**2,uses_future_predictions=k>0)
  # Pointwise truth-selected offset is an even less attainable diagnostic bound.
  oracle=1-math.fsum(min((data[x+k][1]-data[x][0])**2 for k in range(-3,4)) for x in support)/sst
  assert oracle+1e-12>=max(v['nse'] for v in scores.values())
  result[j['name']]=dict(n=len(support),offset_scores=scores,pointwise_truth_selected_offset_nse=oracle,receipt_sha256=sha(receipt))
 report=dict(models=result,new_fits=0,test_scores=0,limitation='NONDEPLOYABLE timing diagnostic using future predictions and truth-selected offsets. Not achieved forecast skill. Fixed common contiguous support for offsets -3 through +3.')
 with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 for n,v in result.items():print(n,'n=',v['n'],'unshifted=',round(v['offset_scores']['0']['nse'],4),'plus1future=',round(v['offset_scores']['1']['nse'],4),'pointwise_oracle=',round(v['pointwise_truth_selected_offset_nse'],4))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign');p.add_argument('new_output');a=p.parse_args();run(a.campaign,a.new_output)
