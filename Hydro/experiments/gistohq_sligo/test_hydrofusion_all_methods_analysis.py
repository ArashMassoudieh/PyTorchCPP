import contextlib
import io
import json
import shutil
import unittest
from pathlib import Path
from test_hydrofusion_six_fit_analysis import AnalysisTests
from analyze_hydrofusion_all_methods import analyze,kge_components,sha

class DiagnosticsTests(AnalysisTests):
    def setUp(self):
        super().setUp()
        p=self.root/'screen_plan.json';plan=json.loads(p.read_text())
        for j in plan['jobs']:j['mode']=j['name'][4:].rsplit('_s',1)[0]
        p.write_text(json.dumps(plan))
        done=self.root/'screen_completion.json';c=json.loads(done.read_text());c['plan_sha256']=sha(p);done.write_text(json.dumps(c))
    def test_complete_diagnostics(self):
        with contextlib.redirect_stdout(io.StringIO()):r=analyze(self.root,self.root/'diagnostics.json')
        self.assertEqual(r['splits']['test']['n'],3)
        self.assertEqual(r['splits']['test']['paired']['ffn_residual_minus_direct']['positive_seeds'],3)
        for m in r['splits']['test']['methods'].values():
            self.assertAlmostEqual(m['metrics']['kge'],m['kge_components']['kge'])
    def make_extension(self):
        from run_hydrofusion_remaining_methods import specifications
        ext=self.root/'extension';ext.mkdir()
        shutil.copytree(self.root/'frozen_support',ext/'frozen_support')
        shutil.copytree(self.root/'prepared_input',ext/'prepared_input')
        (self.root/'prepared_input/intervals.csv').write_text('fabricated presealed fixture')
        (ext/'prepared_input/intervals.csv').write_text('fabricated presealed fixture')
        jobs=[]
        for family,mode,seed in specifications():
            name=f'{family}_{mode}_s{seed}';jobs.append(dict(name=name,family=family,mode=mode,seed=seed))
            fit=ext/'fits'/name;bundle=fit/'bundles'/name;bundle.mkdir(parents=True)
            self.predictions(bundle/'predictions.csv',family,'test',30,.1)
            self.predictions(fit/'validation_predictions.csv',family,'validation',20,.1)
            (bundle/'scalers.csv').write_text('approach,kind,index,method,shape,offset,scale\na,input,0,standardize,1,0,1\n')
            (bundle/'hydrofusion_metadata.txt').write_text('q_ref=2\n')
            record=dict(wall_seconds=1,peak_sampled_rss_bytes=100,retained_artifact_bytes=1000,selected_epoch=3,
                        file_sha256={str(p.relative_to(fit)):sha(p) for p in fit.rglob('*') if p.is_file()})
            (fit/'completion.json').write_text(json.dumps(record))
        plan=dict(jobs=jobs,input_sha256={},source_plan_sha256=sha(self.root/'screen_plan.json'))
        (ext/'screen_plan.json').write_text(json.dumps(plan))
        (ext/'screen_completion.json').write_text(json.dumps(dict(completed=[j['name'] for j in jobs],plan_sha256=sha(ext/'screen_plan.json'))))
        return ext
    def test_full_extension_matched_analysis(self):
        ext=self.make_extension()
        with contextlib.redirect_stdout(io.StringIO()):r=analyze(self.root,self.root/'all.json',ext)
        self.assertEqual(len(r['splits']['test']['methods']),20)
        self.assertEqual(len(r['splits']['test']['paired']),6)
    def test_different_support_rejected(self):
        ext=self.make_extension();(ext/'frozen_support/test_support.csv').write_text('x\n31\n')
        with self.assertRaisesRegex(ValueError,'Support changed'):analyze(self.root,self.root/'all.json',ext)
    def test_different_native_input_rejected(self):
        ext=self.make_extension();(ext/'prepared_input/intervals.csv').write_text('different')
        with self.assertRaisesRegex(ValueError,'Native source changed'):analyze(self.root,self.root/'all.json',ext)
    def test_scale_compression(self):
        r=kge_components([1.,2.,3.],[.5,1.,1.5])
        self.assertAlmostEqual(r['correlation'],1)
        self.assertAlmostEqual(r['variability_ratio'],.5)
        self.assertAlmostEqual(r['mean_ratio'],.5)
        self.assertAlmostEqual(r['kge'],1-2**.5/2)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):kge_components([1.,2.],[1.,float('nan')])
    def test_constant_prediction_undefined(self):
        self.assertIsNone(kge_components([1.,2.,3.],[1.,1.,1.])['kge'])
    def test_tamper_refused(self):
        (self.root/'fits/ffn_log_residual_s42/validation_predictions.csv').write_text('corrupt')
        with self.assertRaisesRegex(ValueError,'Changed completed artifact'):analyze(self.root,self.root/'diagnostics.json')
    def test_incomplete_refused_diagnostics(self):
        (self.root/'screen_completion.json').write_text('{"completed":[],"plan_sha256":"bad"}')
        with self.assertRaisesRegex(ValueError,'Incomplete'):analyze(self.root,self.root/'diagnostics.json')

if __name__=='__main__':unittest.main()
