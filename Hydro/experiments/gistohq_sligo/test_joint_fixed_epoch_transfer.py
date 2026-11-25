import tempfile,unittest
from pathlib import Path
from diagnose_joint_fixed_epoch_transfer import rows,event_windows,partitions
class TransferTests(unittest.TestCase):
 def test_reserved_identity_rejected_before_value(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p.csv';p.write_text('x,split,observed,predicted\n77016,validation,not_a_number,not_a_number\n')
   with self.assertRaisesRegex(ValueError,'identity'):rows(p,{})
 def test_missing_hour_excludes_event(self):
  truth={x:0. for x in range(17544,17570)};truth[17555]=2.;del truth[17556]
  good,bad=event_windows(truth,1.);self.assertEqual(good,[]);self.assertEqual(bad,[[17549,17561]])
 def test_overlapping_windows_merge(self):
  truth={x:0. for x in range(17544,17580)};truth[17555]=2.;truth[17560]=2.
  self.assertEqual(event_windows(truth,1.),([[17549,17566]],[]))
 def test_high_and_remaining_partition(self):
  r=partitions({1:1.,2:10.},{1:1.,2:9.},{1:2.,2:7.},5.)
  self.assertEqual(r['delta_sse'],9.);self.assertEqual(r['high_flow_delta_sse'],8.);self.assertEqual(r['remaining_delta_sse'],1.)
if __name__=='__main__':unittest.main()
