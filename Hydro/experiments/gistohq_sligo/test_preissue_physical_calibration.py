import unittest
import numpy as np
from run_preissue_physical_calibration import fit_gain, scores

class CalibrationTests(unittest.TestCase):
    def test_exact_gain(self):
        p=np.array([1.,2.,4.]);self.assertAlmostEqual(fit_gain(.3*p,p),.3)
    def test_bounds(self):
        p=np.array([1.,2.]);self.assertEqual(fit_gain(2*p,p),1.)
    def test_bad_inputs(self):
        for p in [np.array([0.,0.]),np.array([float('nan'),1.]),np.array([-1.,1.])]:
            with self.assertRaises(ValueError):fit_gain(np.array([1.,2.]),p)
    def test_perfect_metrics(self):
        y=np.array([1.,2.,4.]);s=scores(y,y);self.assertEqual(s['nse'],1);self.assertAlmostEqual(s['r_squared'],1);self.assertEqual(s['pbias_percent'],0)
    def test_metric_nan_rejected(self):
        with self.assertRaises(ValueError):scores(np.array([1.,2.]),np.array([np.nan,2.]))
    def test_scaling_does_not_change_correlation(self):
        y=np.array([1.,2.,4.]);p=np.array([3.,2.,5.]);self.assertAlmostEqual(scores(y,p)['r_squared'],scores(y,.3*p)['r_squared'])

if __name__=='__main__':unittest.main()
