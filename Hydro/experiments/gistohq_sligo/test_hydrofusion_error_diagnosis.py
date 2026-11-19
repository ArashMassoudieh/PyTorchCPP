import math
import unittest
from analyze_hydrofusion_error_diagnosis import native_split,summarize_neural

class DiagnosticStatsTests(unittest.TestCase):
    def test_split_boundaries(self):
        self.assertEqual(native_split('17543'),'train')
        self.assertEqual(native_split('17544'),'validation')
        self.assertEqual(native_split('26304'),'test')
        for x in ['35064','77016','nan','1.5','-1']:
            with self.assertRaises(ValueError):native_split(x)
    def test_target_space_and_physical_errors_distinguished(self):
        rows=[]
        for q in [1.,2.,4.]:
            p=math.expm1(math.log1p(q)-.2)
            rows.append(dict(observed=q,predicted=p,transformed_target=math.log1p(q),transformed_prediction=math.log1p(q)-.2,clipped=0))
        result=summarize_neural(rows,3)
        self.assertAlmostEqual(result['target_space']['mean_error'],-.2)
        self.assertAlmostEqual(result['target_space']['mse'],.04)
        self.assertLess(result['physical']['pbias_percent'],0)
        self.assertEqual(result['physical']['high_flow_n'],1)
    def test_zero_clipping_count(self):
        rows=[dict(observed=1,predicted=0,transformed_target=.5,transformed_prediction=-.1,clipped=1),
              dict(observed=2,predicted=2,transformed_target=.8,transformed_prediction=.8,clipped=0)]
        self.assertEqual(summarize_neural(rows,1)['clipped_fraction'],.5)
    def test_empty_tail(self):self.assertEqual(summarize_neural([],1),dict(n=0))
    def test_constant_target_undefined_variance_explained(self):
        row=dict(observed=1,predicted=1,transformed_target=.5,transformed_prediction=.5,clipped=0)
        self.assertIsNone(summarize_neural([row,row],1)['target_space']['variance_explained'])

if __name__=='__main__':unittest.main()
