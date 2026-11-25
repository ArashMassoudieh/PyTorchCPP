import unittest
from analyze_hydrofusion_order_errors import partition
class Tests(unittest.TestCase):
 def test_partition(self):
  r=partition({1:(1,2),2:(3,1)},2);self.assertEqual(r['sse'],5);self.assertEqual(r['high_sse'],4);self.assertEqual(r['remaining_sse'],1)
 def test_threshold_strict(self):self.assertEqual(partition({1:(2,3)},2)['high_rows'],0)
 def test_perfect(self):self.assertEqual(partition({1:(2,2)},1)['high_sse_fraction'],0)
if __name__=='__main__':unittest.main()
