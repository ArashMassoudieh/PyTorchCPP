import datetime as d,unittest
from inventory_mrms_preissue_archive import hourly_candidates
class InventoryTests(unittest.TestCase):
 def test_availability_identity(self):
  a=hourly_candidates('<a href="PrecipRate_00.00_20150215-115000.grib2.gz">x</a>',d.date(2015,2,15))[0]
  self.assertEqual(a['origin_utc'],'2015-02-15T12:00:00+00:00');self.assertEqual(a['target_x'],1093)
 def test_future_development_labels_excluded(self):self.assertEqual(hourly_candidates('<a href="PrecipRate_00.00_20171231-235000.grib2.gz">x</a>',d.date(2017,12,31)),[])
 def test_wrong_date_rejected(self):
  with self.assertRaises(ValueError):hourly_candidates('<a href="PrecipRate_00.00_20150216-115000.grib2.gz">x</a>',d.date(2015,2,15))
 def test_other_minutes_excluded(self):self.assertEqual(hourly_candidates('<a href="PrecipRate_00.00_20150215-115200.grib2.gz">x</a>',d.date(2015,2,15)),[])
if __name__=='__main__':unittest.main()
