import csv,hashlib,json,tempfile,unittest
from pathlib import Path
from diagnose_hydrofusion_timing_bound import run,sha
class TimingTests(unittest.TestCase):
 def setUp(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.r=Path(t.name);self.fit=self.r/'fits/model';self.fit.mkdir(parents=True)
  p=self.fit/'validation_predictions.csv'
  with p.open('w') as f:
   w=csv.writer(f);w.writerow(['x','split','observed','predicted'])
   for x in range(17550,17570):w.writerow([x,'validation',(x-17550)**2,(x-17551)**2])
  (self.fit/'completion.json').write_text(json.dumps({'sha256':{'validation_predictions.csv':sha(p)}}))
  (self.r/'screen_plan.json').write_text(json.dumps({'jobs':[{'name':'model'}]}));(self.r/'launch_claim.json').write_text(json.dumps({'plan_sha256':sha(self.r/'screen_plan.json')}))
 def test_delayed_signal_identified_without_claiming_forecast(self):
  out=self.r/'result';run(self.r,out);r=json.loads(out.read_text());m=r['models']['model']
  self.assertEqual(m['n'],14);self.assertEqual(m['offset_scores']['1']['nse'],1);self.assertTrue(m['offset_scores']['1']['uses_future_predictions']);self.assertLess(m['offset_scores']['0']['nse'],1);self.assertIn('NONDEPLOYABLE',r['limitation'])
 def test_mutated_predictions_rejected(self):
  with (self.fit/'validation_predictions.csv').open('a') as f:f.write('bad\n')
  with self.assertRaises(AssertionError):run(self.r,self.r/'result')
if __name__=='__main__':unittest.main()
