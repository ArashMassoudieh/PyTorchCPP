import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audit_hydrofusion_producer_package as audit


class ProducerAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'observations').mkdir()
        self.rows = [
            ['2015-01-01T00:00:00Z', '00060', '1', 'cfs', 'A'],
            ['2015-01-01T00:15:00Z', '00060', '3', 'cfs', 'A'],
            ['2015-01-01T00:30:00Z', '00060', '999', 'cfs', 'e'],
            ['2015-01-01T00:45:00Z', '00060', '5', 'cfs', 'A'],
            ['2015-01-01T01:00:00Z', '00060', '', 'cfs', 'A'],
            ['2015-01-01T00:00:00Z', 'PRECTOTCORR', '2', 'mm/hour', ''],
        ]
        self.manifest = dict(schema_name='HydroPINNExport', schema_version='1.2',
                             profile='water-balance-v1', study_start='2015-01-01T00:00:00Z',
                             study_end='2015-01-01T01:00:00Z', catchment_area_m2=1e6,
                             catchment_area_source='fabricated', source_package_id='fabricated-only',
                             assets=[dict(asset_id='fake', path='observations/temporal_1.csv')])
        self.policy = dict(source_package_id='fabricated-only', qualifier_policy_evidence='synthetic test rule only',
                           accepted_discharge_qualifier_sets=['A'], minimum_native_hour_coverage=.75,
                           discharge_sample_support_seconds={'fake': 900})
        self.save()

    def save(self):
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(audit.HEADER)
        writer.writerows(self.rows)
        data = stream.getvalue().encode()
        (self.root / 'observations/temporal_1.csv').write_bytes(data)
        self.manifest['assets'][0]['sha256'] = audit.sha(data)
        data = json.dumps(self.manifest).encode()
        (self.root / 'manifest.json').write_bytes(data)
        self.policy['manifest_sha256'] = audit.sha(data)

    def run_audit(self):
        return audit.audit(self.root, self.policy)

    def test_native_coverage_and_units_no_imputation(self):
        result = self.run_audit()
        first, second = result['native_discharge_hours']
        self.assertEqual(first['native_coverage'], .75)
        self.assertAlmostEqual(first['q_m3_s'], 3 * .028316846592)
        self.assertAlmostEqual(first['q_mm_h'], first['q_m3_s'] * 3.6)
        self.assertFalse(second['eligible_discharge'])
        self.assertIsNone(second['q_mm_h'])
        self.assertFalse(result['research_ready'])
        self.assertEqual(result['qualifier_counts'], {'A': 4, 'e': 1})

    def test_missing_record_terminates_support(self):
        self.rows[1][2] = ''
        self.save()
        self.assertEqual(self.run_audit()['native_discharge_hours'][0]['native_coverage'], .5)

    def test_cadence_is_not_guessed(self):
        self.policy['discharge_sample_support_seconds'] = {}
        with self.assertRaisesRegex(ValueError, 'explicit discharge sample support'):
            self.run_audit()

    def test_qualifiers_are_exact_sets_not_substrings(self):
        self.rows[0][4] = 'A;e'
        self.save()
        self.assertEqual(self.run_audit()['native_discharge_hours'][0]['native_coverage'], .5)

    def test_manifest_gate_precedes_any_asset_open(self):
        self.manifest['study_end'] = '2024-01-01'
        self.save()
        original = Path.read_bytes
        def guarded(path):
            if path.name != 'manifest.json':
                raise AssertionError('observation asset opened')
            return original(path)
        with patch.object(Path, 'read_bytes', guarded):
            with self.assertRaisesRegex(ValueError, 'wholly presealed'):
                self.run_audit()

    def test_bad_timestamp_precedes_numeric_conversion(self):
        self.rows[0][2] = 'not-a-number'
        self.rows[-1][0] = '2024-01-01T00:00:00Z'
        self.save()
        with self.assertRaisesRegex(ValueError, 'outside presealed'):
            self.run_audit()

    def test_hash_binding(self):
        with (self.root / 'observations/temporal_1.csv').open('a') as stream:
            stream.write('tampered')
        with self.assertRaisesRegex(ValueError, 'asset hash'):
            self.run_audit()

    def test_duplicate_and_bad_values(self):
        self.rows.append(list(self.rows[0]))
        self.save()
        with self.assertRaisesRegex(ValueError, 'duplicate variable'):
            self.run_audit()
        self.rows.pop()
        for value in ('nan', 'inf', '-1', '1junk'):
            with self.subTest(value=value):
                self.rows[0][2] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.run_audit()

    def test_traversal_and_symlink(self):
        self.manifest['assets'][0]['path'] = 'observations/../manifest.json'
        self.save()
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            self.run_audit()
        self.manifest['assets'][0]['path'] = 'observations/temporal_1.csv'
        self.save()
        target = self.root / 'observations/temporal_1.csv'
        other = self.root / 'other.csv'
        target.rename(other)
        target.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_audit()

    def test_strict_json_and_dates(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                audit.read_json(value)
        for value in ('2015-02-30T00:00:00Z', '2015-1-1T00:00:00Z'):
            with self.assertRaises(ValueError):
                audit.timestamp(value)

    def test_unknown_units_and_overflow(self):
        self.rows[0][3] = 'mm/hr'
        self.save()
        with self.assertRaisesRegex(ValueError, 'unknown discharge unit'):
            self.run_audit()
        self.rows[0][3] = 'm3/s'
        self.rows[0][2] = '1e308'
        self.save()
        with self.assertRaisesRegex(ValueError, 'overflow'):
            self.run_audit()

    def test_hourly_record_not_extended_across_gap(self):
        self.rows = [self.rows[0]]
        self.policy['discharge_sample_support_seconds']['fake'] = 300
        self.save()
        self.assertAlmostEqual(self.run_audit()['native_discharge_hours'][0]['native_coverage'], 1/12)


if __name__ == '__main__':
    unittest.main()
