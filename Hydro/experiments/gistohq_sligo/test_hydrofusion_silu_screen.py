import copy,json,tempfile,unittest
from pathlib import Path
import run_hydrofusion_silu_screen as s
class IsolationTests(unittest.TestCase):
 def setUp(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);r=Path(t.name);m=r/'manifest';m.write_text('{}');jobs=[]
  for recipe in s.RECIPES:
   c=dict(experiment_id='old',hydrofusion_ffn_activation='tanh',hydrofusion_native_manifest_path='old',hydrofusion_native_manifest_sha256=s.digest(m),epochs=50,random_seed=42,hidden_layers='32',hydrofusion_development_only=True,hydrofusion_context_recipe=recipe)
   ref=r/(recipe+'_old');ref.write_text(json.dumps(c));c.update(experiment_id='new',hydrofusion_ffn_activation='silu',hydrofusion_native_manifest_path=str(m));f=r/recipe;f.write_text(json.dumps(c));jobs.append(dict(config=str(f),reference_config=str(ref),family=s.FAMILY,recipe=recipe,seed=42))
  self.p=dict(limits=copy.deepcopy(s.LIMITS),jobs=jobs,pins={str(m):s.digest(m)},manifest=str(m))
 def test_allowed(self):s.verify(self.p)
 def test_legacy_default_tanh(self):
  for j in self.p['jobs']:
   f=Path(j['reference_config']);c=json.loads(f.read_text());del c['hydrofusion_ffn_activation'];f.write_text(json.dumps(c))
  s.verify(self.p)
 def test_loss_change(self):
  f=Path(self.p['jobs'][0]['config']);c=json.loads(f.read_text());c['objective']='changed';f.write_text(json.dumps(c))
  with self.assertRaisesRegex(ValueError,'isolation'):s.verify(self.p)
 def test_family(self):
  self.p['jobs'][0]['family']='lstm'
  with self.assertRaisesRegex(ValueError,'budget'):s.verify(self.p)
 def test_extra_fit(self):
  self.p['jobs'].append(self.p['jobs'][0])
  with self.assertRaisesRegex(ValueError,'budget'):s.verify(self.p)
 def test_resources(self):
  self.p['limits']['epochs']=51
  with self.assertRaisesRegex(ValueError,'budget'):s.verify(self.p)
 def test_input(self):
  Path(self.p['manifest']).write_text('changed')
  with self.assertRaisesRegex(ValueError,'Pinned input'):s.verify(self.p)
 def test_recipe(self):
  for k in ['config','reference_config']:
   f=Path(self.p['jobs'][0][k]);c=json.loads(f.read_text());c['hydrofusion_context_recipe']='absolute25';f.write_text(json.dumps(c))
  with self.assertRaisesRegex(ValueError,'identity'):s.verify(self.p)
if __name__=='__main__':unittest.main()
