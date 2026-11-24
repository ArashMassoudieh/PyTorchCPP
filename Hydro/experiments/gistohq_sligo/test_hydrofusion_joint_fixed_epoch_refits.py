import tempfile,unittest
from pathlib import Path
from run_hydrofusion_joint_fixed_epoch_refits import read,validate,LIMITS
class FixedRefitTests(unittest.TestCase):
 def test_no_extra_job_inventory(self):
  with self.assertRaises(ValueError):validate(dict(jobs=[],limits=LIMITS))
 def test_sealed_identity_precedes_value_read(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.csv';p.write_text('approach,split,x,observed,predicted\nffn_causal_forecast,validation,77016,nan,nan\n')
   with self.assertRaisesRegex(ValueError,'identity'):read(p,'ffn_causal_forecast',{})
 def test_native_support_exact(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.csv';p.write_text('approach,split,x,observed,predicted\nffn_causal_forecast,validation,18000,1,2\n')
   self.assertEqual(read(p,'ffn_causal_forecast',{18000:(1.,1.)}),{18000:(1.,2.)})
   with self.assertRaises(ValueError):read(p,'ffn_causal_forecast',{18000:(1.,1.),18001:(2.,2.)})
 def test_nonfinite_prediction_fails(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.csv';p.write_text('approach,split,x,observed,predicted\nffn_causal_forecast,validation,18000,1,nan\n')
   with self.assertRaises(ValueError):read(p,'ffn_causal_forecast',{18000:(1.,1.)})
if __name__=='__main__':unittest.main()
