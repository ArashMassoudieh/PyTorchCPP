import copy
import json
from pathlib import Path
import tempfile
import unittest
from run_hydrofusion_remaining_methods import specifications,expected_config,validate,LIMITS,digest

class ExtensionGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'prepared_input').mkdir();(self.root/'prepared_input/native_input_manifest.json').write_text('{}')
        jobs=[];pins={}
        for family,mode,seed in specifications():
            c=expected_config(self.root,self.root,family,mode,seed);config=self.root/(c['experiment_id']+'.json');config.write_text(json.dumps(c))
            batch=config.with_suffix('.batch');batch.write_text(f'{family} {config.name}\n')
            jobs.append(dict(name=c['experiment_id'],family=family,mode=mode,seed=seed,config=str(config),batch=str(batch),neural=family!='pinn_causal_forecast'))
            pins.update({str(p):digest(p) for p in (config,batch)})
        self.plan=dict(schema='HydroFusionRemainingMethods-v1',limits=copy.deepcopy(LIMITS),jobs=jobs,input_sha256=pins,source_campaign=str(self.root))
    def test_exact_extension(self):
        validate(self.plan,self.root)
        self.assertEqual(sum(j['neural'] for j in self.plan['jobs']),12)
    def test_no_extra_seed(self):
        self.plan['jobs'][-1]['seed']=43
        with self.assertRaisesRegex(ValueError,'Exactly twelve'):validate(self.plan,self.root)
    def test_no_expanded_limit(self):
        self.plan['limits']['epochs']=51
        with self.assertRaisesRegex(ValueError,'Budget'):validate(self.plan,self.root)
    def test_rehashed_recipe_change_rejected(self):
        p=Path(self.plan['jobs'][0]['config']);c=json.loads(p.read_text());c['hidden_layers']='64';p.write_text(json.dumps(c));self.plan['input_sha256'][str(p)]=digest(p)
        with self.assertRaisesRegex(ValueError,'recipe changed'):validate(self.plan,self.root)
    def test_unbound_batch_change_rejected(self):
        p=Path(self.plan['jobs'][0]['batch']);p.write_text('ffn_causal_forecast wrong.json\n');self.plan['input_sha256'][str(p)]=digest(p)
        with self.assertRaisesRegex(ValueError,'recipe changed'):validate(self.plan,self.root)
    def test_physical_is_not_a_fit(self):
        self.plan['jobs'][-1]['neural']=True
        with self.assertRaisesRegex(ValueError,'type changed'):validate(self.plan,self.root)

if __name__=='__main__':unittest.main()
