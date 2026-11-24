import tempfile,unittest
from pathlib import Path
from verify_hydrofusion_joint_inner_screen import read_predictions,gated,metric
class InnerVerifierTests(unittest.TestCase):
 def test_rejects_outer_before_values(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.csv';p.write_text('approach,split,x,observed,predicted\nffn_causal_forecast,validation,17544,nan,nan\n')
   with self.assertRaisesRegex(ValueError,'identity'):read_predictions(p,'ffn_causal_forecast',{})
 def test_rejects_duplicate(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.csv';p.write_text('approach,split,x,observed,predicted\n'+'ffn_causal_forecast,validation,8800,1,1\n'*2)
   with self.assertRaisesRegex(ValueError,'identity'):read_predictions(p,'ffn_causal_forecast',{8800:(1.,1.)})
 def test_rejects_incomplete(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.csv';p.write_text('approach,split,x,observed,predicted\nffn_causal_forecast,validation,8800,1,1\n')
   with self.assertRaises(ValueError):read_predictions(p,'ffn_causal_forecast',{8800:(1.,1.),8801:(2.,2.)})
 def test_high_flow_partition(self):
  m=metric([1.,2.,4.],[1.,2.,3.],2.);self.assertEqual(m['high_flow_rmse'],1.)
 def test_bias_cannot_be_hidden_by_improved_nse(self):
  a=dict(nse=.5,r_squared=.5,mae=1.,high_flow_rmse=1.,pbias_percent=0.)
  b=dict(nse=.8,r_squared=.8,mae=.5,high_flow_rmse=.5,pbias_percent=2.1)
  self.assertFalse(all(gated(a,b).values()))

class CompleteVerifierFixtureTests(unittest.TestCase):
 def fixture(self,root):
  import csv,hashlib,itertools,json
  from run_hydrofusion_joint_inner_screen import NEURAL,physical
  from verify_hydrofusion_joint_inner_screen import verify
  common={'train':{100:(1.,.9),101:(2.,1.8)},'validation':{8800:(1.,1.),8801:(2.,2.)}}
  (root/'support.csv').write_text('split,target_x,observed,persistence\ntrain,100,1,.9\ntrain,101,2,1.8\nvalidation,8800,1,1\nvalidation,8801,2,2\n')
  (root/'physical_train_features.csv').write_text('target_x,observed,origin_q,available_effective_rain,predicted\n100,1,.9,4,0\n101,2,1.8,8,0\n8800,1,1,4,0\n8801,2,2,8,0\n')
  jobs=[dict(name=f'inner_{f}_h{w}_s42',family=f,width=w) for f in NEURAL for w in [32,64]]
  plan=dict(jobs=jobs,no_retries=True,no_automatic_expansion=True,pins={},support_counts={'train':2,'validation':2},physical_shapes=[list(p) for p in itertools.product([.1,.4,1.],[.005,.025,.05],[.3,.7,.9])])
  (root/'screen_plan.json').write_text(json.dumps(plan));h=hashlib.sha256((root/'screen_plan.json').read_bytes()).hexdigest();(root/'launch_claim.json').write_text(json.dumps(dict(plan_sha256=h)))
  m=dict(n=2,nse=1.,r_squared=1.,mae=0.,pbias_percent=0.,high_flow_rmse=0.);models={}
  for j in jobs:
   fit=root/'fits'/j['name'];bundle=fit/'bundles'/j['name'];bundle.mkdir(parents=True)
   data='approach,split,x,observed,predicted\n'+f"{j['family']},validation,8800,1,1\n{j['family']},validation,8801,2,2\n"
   (bundle/'predictions.csv').write_text(data);(fit/'validation_predictions.csv').write_text(data)
   (fit/'fresh_reload.log').write_text('PASS all_validation_rows=2 production_target_free_checks=12\n')
   with (bundle/'training_history.csv').open('w') as f:
    f.write('epoch,validation_loss,training_loss,selected_checkpoint\n')
    for i in range(1,51):f.write(f'{i},{i},1,{int(i==1)}\n')
   pins={str(p.relative_to(fit)):hashlib.sha256(p.read_bytes()).hexdigest() for p in fit.rglob('*') if p.is_file()}
   (fit/'completion.json').write_text(json.dumps(dict(sha256=pins,selected_epoch=1,validation_metrics=m)))
   models[j['name']]=dict(metrics=m,selected_epoch=1)
  checks=dict(nse=False,r_squared=False,mae=True,high_flow=True,bias=True)
  completion=dict(plan_sha256=h,outer_refits_launched=0,outer_validation_rows=0,test_rows=0,models=models,training_q90=1.9,decisions={f:dict(width=32,selected_epoch=1,width64_checks=checks) for f in NEURAL},physical=physical(root,plan,common,1.9))
  (root/'screen_completion.json').write_text(json.dumps(completion))
  return verify,completion
 def test_complete_scope_and_physical_scalar_checks(self):
  with tempfile.TemporaryDirectory() as t:
   verify,_=self.fixture(Path(t));r=verify(Path(t));self.assertEqual(r['completed_jobs'],8);self.assertEqual(r['outer_refits_launched'],0)
 def test_rejects_fabricated_physical_gain(self):
  import json
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);verify,c=self.fixture(root);candidate=next(v for v in c['physical']['candidates'] if 'gain' in v);candidate['gain']+=.1;(root/'screen_completion.json').write_text(json.dumps(c))
   with self.assertRaisesRegex(ValueError,'gain mismatch'):verify(root)

if __name__=='__main__':unittest.main()
