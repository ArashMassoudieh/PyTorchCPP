import copy,tempfile,unittest
from pathlib import Path
from run_hydrofusion_radar_batch import verify,LIMITS,FAMILIES,ARMS,digest
class FixedCampaignGuards(unittest.TestCase):
 def plan(self):
  return dict(limits=dict(LIMITS),jobs=[dict(family=f,seed=s,arm=a) for f in FAMILIES for s in ([42] if f=='pinn_causal_forecast' else [42,43,44]) for a in ARMS],input_sha256={})
 def test_fixed_inventory(self):verify(Path('/tmp'),self.plan())
 def test_duplicate_cannot_replace_missing_job(self):
  p=self.plan();p['jobs'][-1]=dict(p['jobs'][0])
  with self.assertRaises(ValueError):verify(Path('/tmp'),p)
 def test_budget_cannot_expand(self):
  p=self.plan();p['limits']['epochs']=51
  with self.assertRaises(ValueError):verify(Path('/tmp'),p)
 def test_changed_pinned_file_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'input';f.write_text('original');p=self.plan();p['input_sha256'][str(f)]=digest(f);verify(Path(d),p);f.write_text('changed')
   with self.assertRaises(ValueError):verify(Path(d),p)
if __name__=='__main__':unittest.main()
