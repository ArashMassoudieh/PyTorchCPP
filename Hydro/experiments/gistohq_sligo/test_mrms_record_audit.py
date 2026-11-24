import unittest
from verify_mrms_weather_records import check_record
class RecordTests(unittest.TestCase):
 def setUp(self):self.r=dict(target_x=2,url='https://example/PrecipRate_00.00_20150101-005000.grib2.gz',valid_seconds=1420073400,issue_seconds=1420074000,usable=True,historical_delivery_proven=False,status='WEATHER_VALUE_VERIFIED',unit='mm/h',rate_mm_h=2.,coverage=1.,release_proxy_seconds=1420073460,release_basis='archive_http_last_modified')
 def test_valid(self):check_record(self.r)
 def test_late(self):
  self.r['release_proxy_seconds']=1420074001
  with self.assertRaises(ValueError):check_record(self.r)
 def test_nonfinite(self):
  self.r['rate_mm_h']=float('nan')
  with self.assertRaises(ValueError):check_record(self.r)
 def test_no_zero_fill(self):
  self.r['usable']=False;self.r['rate_mm_h']=0
  with self.assertRaises(ValueError):check_record(self.r)
if __name__=='__main__':unittest.main()
