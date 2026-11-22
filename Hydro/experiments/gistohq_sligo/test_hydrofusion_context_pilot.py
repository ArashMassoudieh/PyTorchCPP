import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_hydrofusion_context_pilot as pilot

class ContextPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.common={'train':dict(enumerate([1.,2.,3.,4.,10.])), 'validation':dict(zip(range(17544,17549),[1.,2.,4.,8.,16.]))}
        self.plan=dict(jobs=[],gates=dict(nse_gain=.02,r_squared_gain=.01,maximum_mae_relative_increase=.02,maximum_high_flow_rmse_relative_increase=.02,maximum_absolute_pbias_increase_percentage_points=2))
        for f in pilot.FAMILIES:
            for recipe in pilot.RECIPES:
                name=f+'_'+recipe+'_s42';self.plan['jobs'].append(dict(name=name,family=f,recipe=recipe))
                d=self.root/'fits'/name;d.mkdir(parents=True)
                predicted=list(self.common['validation'].values()) if recipe=='anchored25' else [2.,1.,5.,5.,10.]
                with (d/'validation_predictions.csv').open('w') as out:
                    w=csv.writer(out);w.writerow(['approach','index','split','x','observed','predicted','residual'])
                    for i,((x,q),v) in enumerate(zip(self.common['validation'].items(),predicted)):w.writerow([f,i,'validation',x,q,v,v-q])
    def tearDown(self):self.tmp.cleanup()
    def test_validation_selects_better_variant_without_launch(self):
        with patch.object(pilot,'support',return_value=self.common):d=pilot.analyze(self.root,self.plan)
        for f in pilot.FAMILIES:
            self.assertEqual(d[f]['selected_recipe'],'anchored25');self.assertFalse(d[f]['confirmation_launched'])
            self.assertTrue(d[f]['variants']['anchored25']['goal_met'])
    def test_exact_tie_prefers_simpler_absolute_reference(self):
        for f in pilot.FAMILIES:
            a=self.root/'fits'/(f+'_absolute25_s42')/'validation_predictions.csv'
            b=self.root/'fits'/(f+'_anchored25_s42')/'validation_predictions.csv'
            a.write_bytes(b.read_bytes())
        with patch.object(pilot,'support',return_value=self.common):d=pilot.analyze(self.root,self.plan)
        self.assertTrue(all(d[f]['selected_recipe']=='absolute25' for f in pilot.FAMILIES))
    def test_mismatch_rejected(self):
        p=self.root/'fits'/self.plan['jobs'][0]['name']/'validation_predictions.csv'
        p.write_text(p.read_text().replace(',17544,',',17550,'))
        with patch.object(pilot,'support',return_value=self.common),self.assertRaises(ValueError):pilot.analyze(self.root,self.plan)
    def test_changed_budget_rejected(self):
        with self.assertRaises(ValueError):pilot.verify(dict(limits={},jobs=[]))
    def test_test_support_rejected_before_observation(self):
        (self.root/'support.csv').write_text('split,target_x,observed,persistence\ntest,26304,UNREADABLE,UNREADABLE\n')
        with self.assertRaisesRegex(ValueError,'Test rows forbidden'):pilot.support(self.root)

if __name__=='__main__':unittest.main()
