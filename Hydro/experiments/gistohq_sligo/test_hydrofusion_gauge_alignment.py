import unittest
from audit_hydrofusion_gauge_alignment import audit


class GaugeAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.spec={'geometry':{'outlet':{'latitude':38.,'longitude':-77.}},'catchment':{'area_m2':1609.344**2}}
        self.rdb='site_no\tdrain_area_va\tdec_lat_va\tdec_long_va\n15s\t8n\t10n\t10n\n01650800\t1\t38\t-77\n'

    def test_matching_metadata_is_not_training_permission(self):
        result=audit(self.spec,self.rdb,'01650800')
        self.assertFalse(result['metadata_discrepancy'])
        self.assertFalse(result['new_training_ready'])
        self.assertEqual(result['outlet_distance_m'],0)

    def test_wrong_area(self):
        self.spec['catchment']['area_m2']*=2
        self.assertAlmostEqual(audit(self.spec,self.rdb,'01650800')['package_area_excess_percent'],100)

    def test_wrong_outlet(self):
        self.spec['geometry']['outlet']['latitude']+=.1
        self.assertTrue(audit(self.spec,self.rdb,'01650800')['metadata_discrepancy'])

    def test_missing_duplicate_invalid(self):
        for text in ['',self.rdb+self.rdb.splitlines()[-1]+'\n',self.rdb.replace('\t1\t','\tnan\t')]:
            with self.assertRaises(ValueError):audit(self.spec,text,'01650800')


if __name__=='__main__':unittest.main()
