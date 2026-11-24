import csv,json,tempfile,unittest
from pathlib import Path
from run_hydrofusion_joint_inner_screen import check,raw_predictions,gates,NEURAL,LIMITS
class InnerScreenTests(unittest.TestCase):
 def test_gates_require_all_tradeoffs(self):
  a=dict(nse=.5,r_squared=.6,mae=1.,high_flow_rmse=2.,pbias_percent=0.)
  b=dict(nse=.6,r_squared=.7,mae=.9,high_flow_rmse=1.9,pbias_percent=3.)
  g=gates(a,b);self.assertFalse(g['bias']);self.assertFalse(all(g.values()))
 def test_exact_eight_job_inventory(self):
  plan={'limits':LIMITS,'jobs':[],'pins':{}}
  with self.assertRaises(ValueError):check(plan)
 def test_sealed_and_outer_identity_rejected_before_nan(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.csv';p.write_text('approach,index,split,x,observed,predicted,residual\nffn_causal_forecast,0,validation,77016,nan,nan,nan\n')
   with self.assertRaisesRegex(ValueError,'identity'):raw_predictions(p,'ffn_causal_forecast',{})
   p.write_text('approach,index,split,x,observed,predicted,residual\nffn_causal_forecast,0,validation,17544,nan,nan,nan\n')
   with self.assertRaisesRegex(ValueError,'identity'):raw_predictions(p,'ffn_causal_forecast',{})
 def test_exact_support_and_nonfinite_predictions(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.csv';header='approach,index,split,x,observed,predicted,residual\n';p.write_text(header+'ffn_causal_forecast,0,validation,8800,1,2,1\n')
   self.assertEqual(raw_predictions(p,'ffn_causal_forecast',{8800:(1.,1.)}),{8800:(1.,2.)})
   p.write_text(header+'ffn_causal_forecast,0,validation,8800,1,nan,nan\n')
   with self.assertRaises(ValueError):raw_predictions(p,'ffn_causal_forecast',{8800:(1.,1.)})
 def test_family_inventory_includes_four_neural(self):
  self.assertEqual(len(NEURAL),4);self.assertNotIn('pinn_causal_forecast',NEURAL)
if __name__=='__main__':unittest.main()
