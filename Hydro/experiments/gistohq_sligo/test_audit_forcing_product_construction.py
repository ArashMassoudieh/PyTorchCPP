"""Synthetic-only input guard, missingness and coverage regressions."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from audit_forcing_product_construction import audit, main


class ForcingAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        (self.p/'observations').mkdir()
        self.meta = dict(study_start='2020-01-01T00:00:00Z', study_end='2020-01-01T02:00:00Z', catchment_area_m2=19089300)
        self.manifest()
        self.rain('temporal_4.csv', 'PRECIP_MRMS_BASIN_AVG', [(0,0),(1,2)])
        self.rain('temporal_1.csv', 'PRECTOTCORR', [(0,7),(1,3),(2,4)])

    def manifest(self):
        (self.p/'manifest.json').write_text(json.dumps(self.meta))

    def rain(self, name, variable, values, unit='mm/hour'):
        with (self.p/'observations'/name).open('w', newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp_utc','variable','value','native_unit'])
            for x,v in values:
                stamp=f'2020-01-01T{x:02d}:00:00Z' if isinstance(x,int) else x
                w.writerow([stamp,variable,v,unit])

    def test_zero_primary_is_not_missing(self):
        r=audit(self.p)
        self.assertEqual(r['hours_using_nasa_power_fallback'],1)
        self.assertEqual(r['mrms_present_hours'],2)
        self.assertEqual(r['hours_with_both_present'],2)
        self.assertEqual(r['product_disagreement_mm_per_hour']['mean_abs_diff'],4)
        self.assertEqual(r['mrms_zero_value_fraction_when_present'],.5)
        self.assertEqual(len(r['input_sha256']),3)

    def test_missing_is_not_zero(self):
        self.rain('temporal_4.csv','PRECIP_MRMS_BASIN_AVG',[])
        self.rain('temporal_1.csv','PRECTOTCORR',[])
        r=audit(self.p)
        self.assertEqual(r['hours_with_both_missing'],3)
        self.assertIsNone(r['mrms_zero_value_fraction_when_present'])
        self.assertIsNone(r['product_disagreement_mm_per_hour']['mean_abs_diff'])

    def test_duplicate_rejected(self):
        self.rain('temporal_4.csv','PRECIP_MRMS_BASIN_AVG',[(0,1),(0,2)])
        with self.assertRaisesRegex(ValueError,'duplicate'):audit(self.p)

    def test_bad_values_and_daily_units_rejected(self):
        for v in ['nan','inf',-1]:
            with self.subTest(value=v):
                self.rain('temporal_4.csv','PRECIP_MRMS_BASIN_AVG',[(0,v)])
                with self.assertRaisesRegex(ValueError,'finite and nonnegative'):audit(self.p)
        self.rain('temporal_4.csv','PRECIP_MRMS_BASIN_AVG',[(0,1)],'mm/day')
        with self.assertRaisesRegex(ValueError,'hourly rainfall'):audit(self.p)

    def test_reserved_manifest_refused_before_observation_open(self):
        self.meta['study_end']='2023-10-15T00:00:00Z';self.manifest()
        with patch('audit_forcing_product_construction.load_rain',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'reserved-period'):audit(self.p)

    def test_reserved_row_refused_before_value_even_other_variable(self):
        self.rain('temporal_1.csv','00060',[('2023-10-15T00:00:00Z','bad-numeric')])
        with self.assertRaisesRegex(ValueError,'reserved-period'):audit(self.p)

    def test_timestamp_quality_before_value(self):
        for stamp,msg in [('2020-01-01T00:00:00','timezone'),('2020-01-01T00:30:00Z','whole hour'),('2020-01-02T00:00:00Z','outside')]:
            with self.subTest(stamp=stamp):
                self.rain('temporal_4.csv','PRECIP_MRMS_BASIN_AVG',[(stamp,'bad-numeric')])
                with self.assertRaisesRegex(ValueError,msg):audit(self.p)

    def test_invalid_area_and_reversed_window(self):
        for v in [0,-1,float('nan')]:
            self.meta['catchment_area_m2']=v;self.manifest()
            with self.assertRaisesRegex(ValueError,'area'):audit(self.p)
        self.meta.update(catchment_area_m2=1,study_end='2019-12-31T23:00:00Z');self.manifest()
        with self.assertRaisesRegex(ValueError,'precedes'):audit(self.p)

    def test_cli_and_no_overwrite(self):
        out=self.p/'out/report.json';args=['--package',str(self.p),'--output',str(out)]
        self.assertEqual(main(args),0);original=out.read_bytes()
        self.assertEqual(json.loads(original)['total_hours_in_study_window'],3)
        with self.assertRaises(FileExistsError):main(args)
        self.assertEqual(out.read_bytes(),original)


if __name__=='__main__':unittest.main()
