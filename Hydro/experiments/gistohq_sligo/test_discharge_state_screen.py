import unittest
import numpy as np
from run_discharge_state_screen import predict,fit_rain_gain
class StateTests(unittest.TestCase):
    def test_dry_closed_form(self):
        q=np.array([1.,2.]);p,base,r=predict(q,np.zeros(2),(.4,.025,.7),1)
        np.testing.assert_allclose(p,q*(.7*np.exp(-.4)+.3*np.exp(-.025)));np.testing.assert_array_equal(r,np.zeros(2))
    def test_fits_known_gain(self):
        b=np.array([1.,2.,3.]);r=np.array([1.,0.,2.]);self.assertAlmostEqual(fit_rain_gain(b+.2*r,b,r),.2)
    def test_infeasible_gain(self):
        self.assertIsNone(fit_rain_gain(np.ones(2),10*np.ones(2),np.ones(2)))
    def test_zero_response(self):
        self.assertEqual(fit_rain_gain(np.ones(2),np.ones(2),np.zeros(2)),0)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):fit_rain_gain(np.ones(2),np.array([np.nan,1]),np.ones(2))
if __name__=='__main__':unittest.main()
