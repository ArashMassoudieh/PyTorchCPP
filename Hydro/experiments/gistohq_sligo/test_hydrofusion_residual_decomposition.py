import tempfile,unittest
from pathlib import Path
from analyze_hydrofusion_residual_decomposition import read,sums
class DecompositionTests(unittest.TestCase):
 def test_reserved_identity_precedes_values(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'v.csv';p.write_text('target_x,observed\n77016,bad\n')
   with self.assertRaisesRegex(ValueError,'identity'):read(p)
 def test_correction_sse_cross_term(self):
  r=dict(reference=1.,observed=3.,predicted=2.,physical_correction=1.,required_physical_correction=2.,learned_log_delta=0.,required_log_delta=0.,clipped=0)
  v=sums([r]);self.assertEqual(v['physical_reference_sse'],4.);self.assertEqual(v['twice_error_correction_cross_term'],-4.);self.assertEqual(v['total_sse'],1.)
 def test_inconsistent_arithmetic_rejected(self):
  r=dict(reference=1.,observed=3.,predicted=2.,physical_correction=10.,required_physical_correction=2.,learned_log_delta=0.,required_log_delta=0.,clipped=0)
  with self.assertRaisesRegex(ValueError,'identity'):sums([r])
 def check_bad_csv(self,line,message):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'v.csv';p.write_text('target_x,observed,reference,predicted,learned_log_delta,required_log_delta,physical_correction,required_physical_correction,clipped\n'+line+'\n')
   with self.assertRaisesRegex(ValueError,message):read(p)
 def test_nonfinite_latent_rejected(self):self.check_bad_csv('18000,3,1,2,nan,1,1,2,0','Nonfinite')
 def test_saved_correction_must_match_prediction(self):self.check_bad_csv('18000,3,1,2,0,1,7,2,0','arithmetic')
 def test_unknown_clip_flag_rejected(self):self.check_bad_csv('18000,3,1,2,0,1,1,2,2','arithmetic')
if __name__=='__main__':unittest.main()
