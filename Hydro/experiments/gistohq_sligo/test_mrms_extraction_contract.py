import datetime as d,unittest
from extract_mrms_preissue_development import identity,timestamp_metadata
class ExtractionTests(unittest.TestCase):
 def test_identity(self):
  a,b=identity(dict(target_x=2,url='https://example/PrecipRate_00.00_20150101-005000.grib2.gz'));self.assertEqual(a.minute,50);self.assertEqual(b.hour,1)
 def test_boundary(self):
  with self.assertRaises(ValueError):identity(dict(target_x=0,url='bad'))
 def test_wrong_filename(self):
  with self.assertRaises(ValueError):identity(dict(target_x=2,url='https://example/PrecipRate_00.00_20150101-015000.grib2.gz'))
 def test_release_known_before_issue(self):
  a,b=identity(dict(target_x=2,url='https://example/PrecipRate_00.00_20150101-005000.grib2.gz'));r=timestamp_metadata({'Last-Modified':'Thu, 01 Jan 2015 00:51:00 GMT'},a,b);self.assertLess(r,b)
 def test_late_archive_timestamp_rejected(self):
  a,b=identity(dict(target_x=2,url='https://example/PrecipRate_00.00_20150101-005000.grib2.gz'))
  with self.assertRaises(ValueError):timestamp_metadata({'Last-Modified':'Thu, 01 Jan 2015 01:01:00 GMT'},a,b)
if __name__=='__main__':unittest.main()
