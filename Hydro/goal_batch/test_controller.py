import copy,csv,importlib.util,json,math,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('goal_controller',Path(__file__).with_name('controller.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ControllerTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.d=Path(self.t.name);self.root=self.d/'run'
  def save(name,obj):
   p=self.d/name;m.write(p,obj,True);return str(p)
  self.save=save
  intervals=self.d/'intervals.csv';intervals.write_text('synthetic fixture only\n')
  manifest=save('manifest.json',dict(execution_scope='train_validation_only',target_cutoff_x=9,end_interval_x_exclusive=9,interval_csv='intervals.csv',interval_csv_sha256=m.sha(intervals)))
  weather=self.d/'rain.csv';weather.write_text('fixture\n');save('preissue_rain_provenance.json',dict(synthetic=True))
  s=self.d/'support.csv'
  with s.open('w') as f:
   w=csv.writer(f);w.writerow(['split','target_x','observed','persistence'])
   for i,y,p in [(1,.1,.2),(2,.2,.3),(3,.3,.1),(7,1,2),(8,2,1),(9,3,2)]:w.writerow(['train' if i<=3 else 'validation',i,y,p])
  self.s=s
  self.limits=dict(epochs=2,wall_seconds_per_fit=5,rss_bytes=256*1024**2,threads=1,gpu=False,artifact_bytes_per_fit=1000000,max_fits=5)
  self.request=dict(goals=dict(nse=.8,r_squared=.9),limits=self.limits,rule=dict(min_nse_gain=.01,max_mae_ratio=1.02,max_high_rmse_ratio=1.02,max_abs_bias_increase=2,required_fraction=1),jobs=[])
  for i,family in enumerate(sorted(m.FAMILIES)):
   c=dict(hydrofusion_development_only=True,hydrofusion_synthetic_verification=False,train_end_x=3,validation_end_x=9,test_end_x=10,epochs=2,random_seed=42,experiment_id='unused',hydrofusion_native_manifest_path=manifest,hydrofusion_native_manifest_sha256=m.sha(manifest),hydrofusion_preissue_rain_path=str(weather),hydrofusion_preissue_rain_sha256=m.sha(weather))
   cp=save(f'c{i}.json',c);self.request['jobs'].append(dict(id=f'job{i}',candidate='base',fold='f1',seed=42,family=family,config=cp,support=str(s)))
  for key in ('binary','verifier'):
   p=self.d/key;p.write_text('#!/bin/sh\nexit 0\n');p.chmod(0o755);self.request[key]=str(p)
 def prepare(self):
  req=Path(self.save('request.json',self.request));m.prepare(req,self.root);self.addCleanup(__import__('shutil').rmtree,Path(m.load(self.root/'plan.json')['jobs'][0]['config']).parent)
 def fake_execute(self,command,output,env,limits,cancel):
  c=m.load(Path(command[1]).with_suffix('.json'));name=c['experiment_id'];b=Path(command[2])/name;b.mkdir(parents=True);(output/'training.log').write_text('fabricated orchestration test; not engine proof\n');(b/'model.pt').write_text('fixture')
  j=next(j for j in m.load(self.root/'plan.json')['jobs'] if j['id']==name);self.write_predictions(b/'predictions.csv',j['family'])
 def write_predictions(self,p,family):
  with p.open('w') as f:
   w=csv.writer(f);w.writerow(['x','split','approach','observed','predicted'])
   for x,y in [(7,1),(8,2),(9,3)]:w.writerow([x,'validation',family,y,y])
 def fake_reload(self,args,stdout,**kwargs):
  import shutil
  shutil.copy2(Path(args[1])/'predictions.csv',args[2]);stdout.write('PASS production_target_free_checks=12\n')
 def run_fake(self):
  with patch.object(m,'execute',side_effect=self.fake_execute),patch.object(m.subprocess,'run',side_effect=self.fake_reload):return m.run(self.root)
 def test_all_five_run_resume_and_scores(self):
  self.prepare();out=self.run_fake();self.assertTrue(out['goal_met']);self.assertEqual(out['completed'],5);self.assertEqual(m.load(self.root/'state.json')['charged_fits'],5)
  with patch.object(m,'execute',side_effect=AssertionError('must not rerun')):self.assertTrue(m.run(self.root)['goal_met'])
 def test_nonfinite_and_unequal_goals(self):
  for v in [float('nan'),float('inf'),-.1,1.1,True]:
   with self.assertRaises(ValueError):m.goals(dict(nse=v,r_squared=.9))
  self.assertEqual(m.goals(dict(nse=.7,r_squared=.9))['r_squared'],.9)
 def test_duplicate_job(self):
  self.request['jobs'][1]['id']=self.request['jobs'][0]['id']
  with self.assertRaises(ValueError):m.validate_request(self.request)
 def test_budget(self):
  self.request['limits']['max_fits']=4
  with self.assertRaises(ValueError):m.validate_request(self.request)
 def test_test_scope(self):
  c=m.load(self.request['jobs'][0]['config']);c['hydrofusion_development_only']=False;m.write(self.request['jobs'][0]['config'],c)
  with self.assertRaises(ValueError):m.validate_request(self.request)
 def test_uncropped_manifest(self):
  c=m.load(self.request['jobs'][0]['config']);p=Path(c['hydrofusion_native_manifest_path']);obj=m.load(p);obj['target_cutoff_x']=77016;m.write(p,obj)
  with self.assertRaises(ValueError):m.validate_request(self.request)
 def test_reserved_identity_before_values(self):
  self.s.write_text('split,target_x,observed,persistence\nvalidation,77016,unreadable,unreadable\n')
  with self.assertRaisesRegex(ValueError,'Reserved'):m.support(self.s,dict(train_end_x=3,validation_end_x=9))
 def test_interruption_never_retries(self):
  self.prepare();d=self.root/'jobs/job0';d.mkdir();m.write(d/'claim.json',{},True)
  with patch.object(m,'execute',side_effect=AssertionError('retry')):
   with self.assertRaisesRegex(ValueError,'prior claim'):m.run(self.root)
  self.assertEqual(m.load(self.root/'state.json')['status'],'INTERRUPTED')
 def test_cancellation_before_fit(self):
  self.prepare();(self.root/'CANCEL').touch();self.assertEqual(self.run_fake()['completed'],0)
 def test_failure_stops_inventory(self):
  self.prepare()
  with patch.object(m,'execute',side_effect=RuntimeError('fixture failure')):
   with self.assertRaises(RuntimeError):m.run(self.root)
  self.assertEqual(m.load(self.root/'state.json')['charged_fits'],1)
 def test_corrupt_verified_artifact(self):
  self.prepare();self.run_fake();(self.root/'jobs/job0/bundles/job0/model.pt').write_text('changed')
  with self.assertRaisesRegex(ValueError,'result changed'):m.run(self.root)
 def test_changed_support(self):
  self.prepare();self.s.write_text('changed')
  with self.assertRaisesRegex(ValueError,'Pinned'):m.run(self.root)
 def test_predictions_reject_test_and_nan(self):
  c=m.load(self.request['jobs'][0]['config']);s=m.support(self.s,c);p=self.d/'pred.csv';p.write_text('x,split,approach,observed,predicted\n7,test,ffn_causal_forecast,unreadable,unreadable\n')
  with self.assertRaisesRegex(ValueError,'Non-validation'):m.evaluate(p,'ffn_causal_forecast',c,s)
  p.write_text('x,split,approach,observed,predicted\n7,validation,ffn_causal_forecast,1,nan\n')
  with self.assertRaisesRegex(ValueError,'invalid'):m.evaluate(p,'ffn_causal_forecast',c,s)
 def test_lock(self):
  self.prepare()
  with (self.root/'controller.lock').open('a') as f:
   m.fcntl.flock(f,m.fcntl.LOCK_EX)
   with self.assertRaises(BlockingIOError):m.run(self.root)
 def test_undefined_metrics(self):
  with self.assertRaises(ValueError):m.scores({1:(1,2),2:(1,2)},.5)
 def test_real_watchdog_cancel(self):
  d=self.d/'watch';d.mkdir();cancel=self.d/'cancel';cancel.touch()
  with self.assertRaisesRegex(RuntimeError,'CANCELLED'):m.execute([sys.executable,'-c','import time;time.sleep(30)'],d,os.environ,self.limits,cancel)
 def test_duplicate_json(self):
  p=self.d/'duplicate.json';p.write_text('{"nse":0.7,"nse":0.9}')
  with self.assertRaises(ValueError):m.load(p)
 def test_goal_revision_preserves_budget_and_fits(self):
  self.prepare();self.run_fake();before=m.load(self.root/'state.json')['charged_fits']
  with patch.object(sys,'argv',['controller','revise-goals',str(self.root),'--nse','0.9','--r-squared','0.7']):m.main()
  p,state=m.verify_plan(self.root);self.assertEqual(p['goals'],dict(nse=.9,r_squared=.7));self.assertEqual(state['charged_fits'],before)
  with patch.object(m,'execute',side_effect=AssertionError('repeat')):self.assertTrue(m.run(self.root)['goal_met'])
 def test_corrupt_goal_revision_chain(self):
  self.prepare();d=self.root/'goal_revisions';d.mkdir();m.write(d/'000002.json',dict(goals=dict(nse=.9,r_squared=.9),previous_sha256='bad'),True)
  with self.assertRaisesRegex(ValueError,'revision chain'):m.verify_plan(self.root)
 def test_partial_candidate_cannot_meet_goal(self):
  self.request['jobs']=[self.request['jobs'][0],dict(self.request['jobs'][0],id='other',seed=43)]
  c=m.load(self.request['jobs'][0]['config']);c['random_seed']=43;self.request['jobs'][1]['config']=self.save('other_config.json',c)
  self.prepare();p,_=m.verify_plan(self.root);j=p['jobs'][0];d=self.root/'jobs'/j['id'];d.mkdir();self.fake_execute([p['binary'],j['batch'],str(d/'bundles')],d,{},self.limits,None)
  self.write_predictions(d/'validation_predictions.csv',j['family']);value=m.evaluate(d/'validation_predictions.csv',j['family'],m.load(j['config']),m.support(j['support'],m.load(j['config'])));m.write(d/'verified.json',dict(scores=value,hashes={str(q.relative_to(d)):m.sha(q) for q in d.rglob('*') if q.is_file()}),True)
  self.assertFalse(m.report(self.root,p)['goal_met'])
 def test_real_watchdog_timeout(self):
  d=self.d/'timeout';d.mkdir();limits=dict(self.limits,wall_seconds_per_fit=1)
  with self.assertRaisesRegex(RuntimeError,'BUDGET'):m.execute([sys.executable,'-c','import time;time.sleep(10)'],d,os.environ,limits,self.d/'cancel')
if __name__=='__main__':unittest.main()
