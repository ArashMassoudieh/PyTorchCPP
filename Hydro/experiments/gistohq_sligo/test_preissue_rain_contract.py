import unittest
from preissue_rain_contract import select_rate,EPOCH_SECONDS
class RainContractTests(unittest.TestCase):
 def setUp(self):self.issue=EPOCH_SECONDS+3600;self.r=dict(valid_seconds=self.issue-600,release_proxy_seconds=self.issue-500,release_basis='archive_http_last_modified',coverage=1.,rate_mm_h=5.)
 def test_fixed_identity(self):
  r=select_rate([self.r],self.issue);self.assertEqual(r['target_x'],2);self.assertEqual(r['rate_mm_h'],5.);self.assertFalse(r['historical_delivery_proven'])
 def test_missing_is_not_dry(self):self.assertIsNone(select_rate([],self.issue))
 def test_late_release_is_not_dry(self):
  self.r['release_proxy_seconds']=self.issue+1;self.assertIsNone(select_rate([self.r],self.issue))
 def test_future_value_not_read(self):
  self.r.update(valid_seconds=self.issue+10,rate_mm_h=object());self.assertIsNone(select_rate([self.r],self.issue))
 def test_scope_guard_precedes_weather_access(self):
  with self.assertRaises(ValueError):select_rate([{'poison':object()}],EPOCH_SECONDS+26303*3600)
 def test_nan_rejected(self):
  self.r['rate_mm_h']=float('nan')
  with self.assertRaises(ValueError):select_rate([self.r],self.issue)
 def test_duplicate_rejected(self):
  with self.assertRaises(ValueError):select_rate([self.r,self.r],self.issue)
 def test_low_coverage_is_missing(self):
  self.r['coverage']=.74;self.assertIsNone(select_rate([self.r],self.issue))
if __name__=='__main__':unittest.main()
