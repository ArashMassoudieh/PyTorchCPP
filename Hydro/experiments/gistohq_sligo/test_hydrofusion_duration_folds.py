import unittest,tempfile
from pathlib import Path
from run_hydrofusion_duration_folds import stability,read_support,FOLDS
class DurationTests(unittest.TestCase):
 def test_no_forced_epoch_consensus(self):
  h=[]
  for i in [4,19,39]:
   v=[2.]*50;v[i]=1.;h.append(v)
  self.assertEqual(stability(h)['status'],'NO_STABLE_EPOCH')
 def test_stable_earliest_near_optimum(self):
  v=[2.]*50;v[9]=1.;v[10]=1.01
  self.assertEqual(stability([v,v,v])['earliest_common_epoch'],10)
 def test_late_optimum_is_flagged(self):
  v=[2.]*49+[1.];self.assertEqual(stability([v,v,v])['status'],'PROVISIONAL_LATE_OPTIMUM')
 def test_nonfinite_loss_rejected(self):
  with self.assertRaises(ValueError):stability([[float('nan')]*50]*3)
 def test_validation_blocks_do_not_overlap(self):
  self.assertEqual([f[2] for f in FOLDS],[8759,13127,17543]);self.assertTrue(all(FOLDS[i][2]==FOLDS[i+1][1] for i in range(2)))
 def test_sealed_identity_before_values(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'s.csv';p.write_text('split,target_x,observed,persistence\nvalidation,77016,bad,bad\n')
   with self.assertRaisesRegex(ValueError,'identity'):read_support(p,8759,17543)
if __name__=='__main__':unittest.main()
