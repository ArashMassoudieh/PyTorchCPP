import csv
from pathlib import Path
import tempfile
import unittest
import verify_hydrofusion_preissue_results as v
class IndependentResults(unittest.TestCase):
    def test_correlation_is_not_agreement(self):
        m=v.independent_metrics([1,2,3],[2,4,6],3)
        self.assertEqual(m['r_squared'],1);self.assertEqual(m['nse'],-6);self.assertEqual(m['pbias_percent'],100);self.assertEqual(m['high_flow_rmse'],3)
    def test_perfect(self):
        m=v.independent_metrics([1,2,3],[1,2,3],2)
        self.assertEqual(m['nse'],1);self.assertEqual(m['r_squared'],1);self.assertEqual(m['mae'],0);self.assertEqual(m['kge'],1)
    def test_constant_prediction_is_undefined_correlation(self):
        m=v.independent_metrics([1,2,3],[2,2,2],2)
        self.assertEqual(m['nse'],0);self.assertIsNone(m['r_squared']);self.assertIsNone(m['kge'])
    def fixture(self,rows):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p=Path(t.name)/'predictions.csv'
        with p.open('w') as f:
            w=csv.writer(f);w.writerow(['approach','index','split','x','observed','predicted','residual']);w.writerows(rows)
        return p
    def test_seal_rejected_before_value(self):
        p=self.fixture([['family',0,'validation',77016,'nan','nan','nan']])
        with self.assertRaisesRegex(ValueError,'Forbidden prediction identity'):v.predictions(p,'family',{17544:1})
    def test_duplicate_rejected(self):
        p=self.fixture([['family',0,'validation',17544,1,1,0],['family',1,'validation',17544,1,1,0]])
        with self.assertRaisesRegex(ValueError,'identity/index/support'):v.predictions(p,'family',{17544:1})
    def test_nonfinite_rejected(self):
        p=self.fixture([['family',0,'validation',17544,1,'nan','nan']])
        with self.assertRaisesRegex(ValueError,'Nonfinite'):v.predictions(p,'family',{17544:1})
    def test_missing_support_rejected(self):
        p=self.fixture([['family',0,'validation',17544,1,1,0]])
        with self.assertRaisesRegex(ValueError,'Missing prediction'):v.predictions(p,'family',{17544:1,17545:2})
if __name__=='__main__':unittest.main()
