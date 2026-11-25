import unittest
from replay_hydrofusion_stopping_rule import select
class StopTests(unittest.TestCase):
 def test_monotonic_improvement_reaches_cap(self):
  r=select(range(50,0,-1));self.assertEqual((r['stop_epoch'],r['selected_epoch']),(50,50))
 def test_warmup_and_ties(self):
  r=select([1.]*50);self.assertEqual((r['stop_epoch'],r['selected_epoch']),(10,1))
 def test_patience_reset(self):
  values=[10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,2.,2.,2.,2.,.5]+[1.]*35
  r=select(values);self.assertEqual((r['stop_epoch'],r['selected_epoch']),(20,15))
 def test_later_optimum_cannot_change_selection(self):
  a=select([1.]*10+[0.]*40);b=select([1.]*10+[100.]*40);self.assertEqual(a,b)
 def test_future_values_are_not_consumed(self):
  def values():
   yield from [1.]*10
   raise AssertionError('Lookahead consumed')
  self.assertEqual(select(values())['stop_epoch'],10)
 def test_observed_nan_rejected(self):
  with self.assertRaises(ValueError):select([float('nan')]*50)
 def test_epoch_cap_does_not_consume_future(self):
  def values():
   yield from range(50,0,-1)
   raise AssertionError('Epoch51 consumed')
  self.assertEqual(select(values())['stop_epoch'],50)
 def test_insufficient_history_rejected(self):
  with self.assertRaises(ValueError):select([1.]*9)
if __name__=='__main__':unittest.main()
