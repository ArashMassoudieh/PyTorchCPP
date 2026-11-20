import csv,json,tempfile,unittest
from pathlib import Path
from select_calibrated_physics import select,sha,FIELDS
class SelectorTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);c=self.root/'config.json';c.write_text('{}')
  self.p=self.root/'proposal.json';self.data=dict(schema='CalibratedTimingV3BatchProposal-v1',pins={},candidates=[dict(config=str(c),sha256=sha(c)) for i in range(24)],stage_a_gate=dict(minimum_validation_nse=.1,maximum_absolute_validation_pbias_percent=30,reference_validation_mae=.04))
  self.s=self.root/'scores.csv';self.rows=[[i,.1,.02,0,.2,.03,0] for i in range(24)]
 def run_select(self,fields=FIELDS):
  self.p.write_text(json.dumps(self.data))
  with self.s.open('w') as f:w=csv.writer(f);w.writerow(fields);w.writerows(self.rows)
  return select(self.p,self.s)
 def test_pass_and_tie_first_index(self):
  r=self.run_select();self.assertEqual(r['selected_candidate_index'],0);self.assertTrue(all(r['checks'].values()));self.assertFalse(r['training_launched'])
 def test_stop_on_bias(self):
  self.rows[0][4]=.3;self.rows[0][6]=31;r=self.run_select();self.assertEqual(r['status'],'STOP_NO_NEURAL_FITS');self.assertFalse(r['checks']['bias'])
 def test_stop_on_mae(self):
  for r in self.rows:r[5]=.05
  self.assertFalse(self.run_select()['checks']['mae'])
 def test_reject_test_score_column(self):
  with self.assertRaisesRegex(ValueError,'train/validation'):self.run_select(FIELDS+['test_nse'])
 def test_reject_nonfinite(self):
  self.rows[3][4]='nan'
  with self.assertRaisesRegex(ValueError,'metric'):self.run_select()
 def test_reject_reordered_indices(self):
  self.rows[0][0]=1
  with self.assertRaisesRegex(ValueError,'identities'):self.run_select()
 def test_reject_changed_config(self):
  Path(self.data['candidates'][0]['config']).write_text('{"changed":true}')
  with self.assertRaisesRegex(ValueError,'config changed'):self.run_select()
if __name__=='__main__':unittest.main()
