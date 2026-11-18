import unittest
import tempfile
from pathlib import Path
import evaluate_causal_forecast_methods as e

class EvaluatorTest(unittest.TestCase):
    def test_identity_and_constant(self):
        self.assertEqual(e.metrics([1,2,3],[1,2,3],2)['nse'],1)
        self.assertAlmostEqual(e.metrics([1,2,3],[1,2,3],2)['kge'],1)
        self.assertIsNone(e.metrics([1,1],[1,1],1)['nse'])
    def test_units_bias_tail(self):
        m=e.metrics([1,2,3],[2,3,4],2)
        self.assertEqual(m['rmse'],1)
        self.assertEqual(m['high_flow_n'],2)
        self.assertEqual(m['pbias_percent'],50)
    def test_support_and_observed_disagreement(self):
        with self.assertRaises(ValueError):e.evaluate({'a':{1:(1,1)}},[1,2],1,[])
        with self.assertRaises(ValueError):e.evaluate({'a':{1:(1,1)},'b':{1:(2,1)}},[1],1,[])
    def test_incomplete_event(self):
        with self.assertRaises(ValueError):e.evaluate({'a':{1:(1,1),3:(3,3)}},[1,3],1,[{'name':'gap','start_x':1,'end_x':3}])
    def test_event_volume_peak(self):
        event={'name':'frozen','start_x':1,'end_x':3}
        out=e.evaluate({'a':{1:(1,1),2:(3,2),3:(1,4)}},[1,2,3],2,[event])['a']['events']['frozen']
        self.assertEqual(out['peak_timing_error_hours'],1)
        self.assertEqual(out['predicted_volume_mm'],7)
    def test_training_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'train.csv';p.write_text('x,observed\n1,1\n2,3\n')
            self.assertAlmostEqual(e.training_threshold(p,2),2.8)
            with self.assertRaises(ValueError):e.training_threshold(p,1)
    def test_paired_seed_summary(self):
        methods={'control':{'pooled':{'nse':0.5}},'a':{'pooled':{'nse':0.7}},'b':{'pooled':{'nse':0.3}}}
        pairs=[{'name':'paired','seed':1,'control':'control','candidate':'a'},{'name':'paired','seed':2,'control':'control','candidate':'b'}]
        result=e.paired_seed_summary(methods,pairs)['paired']
        self.assertAlmostEqual(result['mean_delta_nse'],0)
        self.assertAlmostEqual(result['population_std_delta_nse'],0.2)
        with self.assertRaises(ValueError):e.paired_seed_summary(methods,pairs+pairs)
    def test_seal_and_nan(self):
        for value in ['nan','77016','-1','1.5']:
            with self.assertRaises(ValueError):e.timestamp(value)
        with self.assertRaises(ValueError):e.finite_nonnegative('nan')

if __name__=='__main__':unittest.main()
