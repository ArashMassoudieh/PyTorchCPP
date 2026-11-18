import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from analyze_hydrofusion_six_fit_screen import analyze, sha

class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        prepared=self.root/'prepared_input';prepared.mkdir();(prepared/'native_input_manifest.json').write_text('{"catchment_area_m2":3600000}')
        support=self.root/'frozen_support';support.mkdir()
        (support/'train_observations.csv').write_text('x,observed\n1,1\n2,2\n3,3\n')
        jobs=[]
        for split,start in [('validation',20),('test',30)]:
            (support/(split+'_support.csv')).write_text('x\n'+''.join(f'{x}\n' for x in range(start,start+3)))
            self.predictions(support/(split+'_persistence.csv'),'persistence',split,start,.3)
        for seed in (42,43,44):
            for mode,error in [('direct_log_q',.4),('log_residual',.05)]:
                name=f'ffn_{mode}_s{seed}';jobs.append(dict(name=name))
                fit=self.root/'fits'/name;bundle=fit/'bundles'/name;bundle.mkdir(parents=True)
                self.predictions(bundle/'predictions.csv','ffn_causal_forecast','test',30,error)
                self.predictions(fit/'validation_predictions.csv','ffn_causal_forecast','validation',20,error)
                (bundle/'scalers.csv').write_text('approach,kind,index,method,shape,offset,scale\na,input,0,standardize,1,0,1\n')
                (bundle/'hydrofusion_metadata.txt').write_text('q_ref=2\n')
                record=dict(wall_seconds=1,peak_sampled_rss_bytes=100,retained_artifact_bytes=1000,selected_epoch=3,
                    file_sha256={str(p.relative_to(fit)):sha(p) for p in fit.rglob('*') if p.is_file()})
                (fit/'completion.json').write_text(json.dumps(record))
        plan=dict(jobs=jobs,input_sha256={},study_type='fabricated-analysis-test',advancement=dict(positive_validation_seeds=2,mean_validation_delta_nse=.02,maximum_relative_validation_mae_increase=.02))
        (self.root/'screen_plan.json').write_text(json.dumps(plan))
        (self.root/'screen_completion.json').write_text(json.dumps(dict(completed=[j['name'] for j in jobs],plan_sha256=sha(self.root/'screen_plan.json'))))
    @staticmethod
    def predictions(path,approach,split,start,error):
        with path.open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['approach','index','split','x','observed','predicted','residual'])
            for i in range(3):w.writerow([approach,i,split,start+i,i+1,i+1+error,error])
    def test_complete_and_gate(self):
        with contextlib.redirect_stdout(io.StringIO()):result=analyze(self.root,self.root/'report.json')
        self.assertTrue(result['advancement_passed'])
        self.assertEqual(result['splits']['validation']['paired']['residual_minus_direct']['positive_seeds'],3)
        self.assertTrue(result['input_scalers_identical_all_arms_seeds'])
    def test_incomplete_refused(self):
        (self.root/'screen_completion.json').write_text('{"completed":[],"plan_sha256":"bad"}')
        with self.assertRaisesRegex(ValueError,'Incomplete'):analyze(self.root,self.root/'report.json')
    def test_changed_completed_artifact_refused(self):
        path=self.root/'fits/ffn_direct_log_q_s42/validation_predictions.csv';path.write_text('tampered')
        with self.assertRaisesRegex(ValueError,'Changed completed artifact'):analyze(self.root,self.root/'report.json')

if __name__=='__main__':unittest.main()
