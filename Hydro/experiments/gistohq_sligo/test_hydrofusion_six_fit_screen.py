import json
import tempfile
import unittest
from pathlib import Path
from run_hydrofusion_six_fit_screen import validate_plan, digest

class SixFitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);jobs=[];pins={}
        for seed in (42,43,44):
            for mode in ('direct_log_q','log_residual'):
                name=f'{mode}_{seed}';p=root/(name+'.json')
                p.write_text(json.dumps(dict(epochs=50,hidden_layers='32',batch_size=64,learning_rate=.001,
                    train_end_x=17543,validation_end_x=26303,test_end_x=35063,random_seed=seed,
                    hydrofusion_target_mode=mode,hydrofusion_synthetic_verification=False,use_csv_data=False,use_hydro_package=False)))
                jobs.append(dict(name=name,seed=seed,mode=mode,config=str(p)));pins[str(p)]=digest(p)
        self.plan=dict(jobs=jobs,input_sha256=pins,limits=dict(epochs=50,wall_seconds_per_fit=28800,
            rss_bytes=12*1024**3,threads=2,gpu=False,artifact_bytes_per_fit=250_000_000))
    def test_exact_plan(self):validate_plan(self.plan)
    def test_duplicate_fit(self):
        self.plan['jobs'][-1]=self.plan['jobs'][0]
        with self.assertRaisesRegex(ValueError,'Exactly six'):validate_plan(self.plan)
    def test_expanded_budget(self):
        self.plan['limits']['epochs']=51
        with self.assertRaisesRegex(ValueError,'limits changed'):validate_plan(self.plan)
    def test_changed_pin(self):
        Path(self.plan['jobs'][0]['config']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'Pinned input changed'):validate_plan(self.plan)
    def test_rehashed_config_still_constrained(self):
        p=Path(self.plan['jobs'][0]['config']);c=json.loads(p.read_text());c['epochs']=51
        p.write_text(json.dumps(c));self.plan['input_sha256'][str(p)]=digest(p)
        with self.assertRaisesRegex(ValueError,'Prospective config changed'):validate_plan(self.plan)

if __name__=='__main__':unittest.main()
