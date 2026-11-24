import copy
import json
from pathlib import Path
import tempfile
import unittest
import run_hydrofusion_preissue_screen as s
class FrozenPlan(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        m=self.root/'manifest';m.write_text('native')
        w=self.root/'weather';w.write_text('weather');p=self.root/'preissue_rain_provenance.json';p.write_text('provenance')
        jobs=[]
        for f in s.FAMILIES:
            for r in s.RECIPES:
                c=dict(epochs=50,hidden_layers='32',batch_size=64,learning_rate=.001,random_seed=42,train_end_x=17543,validation_end_x=26303,test_end_x=35063,lstm_sequence_length=24,hydrofusion_synthetic_verification=False,hydrofusion_development_only=True,hydrofusion_context_recipe=s.context(f),hydrofusion_history_hours='0,1,3,6,12,23' if f.startswith('ffn') else '0',hydrofusion_preissue_rain_recipe=r,hydrofusion_target_mode='log_residual',hydrofusion_objective='log_mse_v1' if f=='pinn_causal_forecast' else 'log_flow_mse_v2',physics_profile='causal_valid_time_dry_v3',hydrofusion_ffn_activation='tanh',storage_coeff=1.,lambda_decay=.1,runoff_coeff=.7,forcing_gain=1.,use_hydro_package=False,use_csv_data=False,hydrofusion_native_manifest_path=str(m),hydrofusion_native_manifest_sha256=s.digest(m),hydrofusion_preissue_rain_path=str(w),hydrofusion_preissue_rain_sha256=s.digest(w),hydrofusion_preissue_rain_provenance_sha256=s.digest(p))
                cp=self.root/(f+r+'.json');cp.write_text(json.dumps(c));jobs.append(dict(family=f,recipe=r,config=str(cp)))
        self.plan=dict(limits=copy.deepcopy(s.LIMITS),jobs=jobs,manifest=str(m),weather=str(w),pins={str(x):s.digest(x) for x in (m,w,p)})
    def test_prepare_interface(self):
        import inspect
        args=s.argument_parser().parse_args(['new_root','--prepare','manifest','weather','support','binary','verifier','preflight'])
        self.assertEqual(len(args.prepare)+1,len(inspect.signature(s.prepare).parameters))
    def test_valid(self):s.verify(self.plan)
    def test_duplicate_job(self):
        self.plan['jobs'][-1]=self.plan['jobs'][0]
        with self.assertRaises(ValueError):s.verify(self.plan)
    def test_limits(self):
        self.plan['limits']['epochs']=100
        with self.assertRaises(ValueError):s.verify(self.plan)
    def test_changed_weather(self):
        Path(self.plan['weather']).write_text('changed')
        with self.assertRaises(ValueError):s.verify(self.plan)
    def test_forbid_test_execution(self):
        p=Path(self.plan['jobs'][0]['config']);c=json.loads(p.read_text());c['hydrofusion_development_only']=False;p.write_text(json.dumps(c))
        with self.assertRaises(ValueError):s.verify(self.plan)
    def test_old_activation_cannot_change(self):
        p=Path(self.plan['jobs'][0]['config']);c=json.loads(p.read_text());c['hydrofusion_ffn_activation']='silu';p.write_text(json.dumps(c))
        with self.assertRaises(ValueError):s.verify(self.plan)
if __name__=='__main__':unittest.main()
