import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from audit_hydrofusion_producer_package import EPOCH, HEADER, sha
from prepare_hydrofusion_causal_package import prepare


def fabricated_package(root):
    source = root / 'producer'
    (source / 'observations').mkdir(parents=True)
    values = []
    units = {'00060': 'cfs', 'PRECTOTCORR': 'mm/hour', 'T2M': 'C', 'RH2M': '%',
             'WS2M': 'm/s', 'ALLSKY_SFC_SW_DWN': 'MJ/hr', 'EVPTRNS': 'MJ/m^2/day'}
    for h in range(300):
        t = EPOCH + h * 3600
        met = {'PRECTOTCORR': (h % 17) * .01, 'T2M': 15, 'RH2M': 50,
               'WS2M': 1, 'ALLSKY_SFC_SW_DWN': .1}
        for name, value in met.items():
            values.append((t, name, value, units[name], ''))
        for step in range(12):
            values.append((t + step * 300, '00060', 10 + h*.01 + (h%13)*.1, 'cfs', 'A'))
        if h % 24 == 0:
            values.append((t, 'EVPTRNS', 2.45, 'MJ/m^2/day', ''))
    target = source / 'observations/temporal_1.csv'
    with target.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(HEADER)
        for t, *rest in sorted(values):
            writer.writerow([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')] + rest)
    manifest = dict(schema_name='HydroPINNExport', schema_version='1.2', profile='water-balance-v1',
                    study_start='2015-01-01T00:00:00Z', study_end='2015-01-13T11:00:00Z',
                    catchment_area_m2=3600000, catchment_area_source='fabricated-only', source_package_id='fabricated-only',
                    source_package_qc_status='pass', source_failed_qc_rule_ids=[],
                    assets=[dict(asset_id='fake', path='observations/temporal_1.csv', sha256=sha(target.read_bytes()))])
    p = source / 'manifest.json'
    p.write_text(json.dumps(manifest))
    variables = {'variables': [dict(name=k, units=[v]) for k,v in sorted(units.items())]}
    (source/'variables.json').write_text(json.dumps(variables))
    policy = dict(manifest_sha256=sha(p.read_bytes()), source_package_id='fabricated-only',
                  qualifier_policy_evidence='fabricated test policy', accepted_discharge_qualifier_sets=['A'],
                  minimum_native_hour_coverage=.75, discharge_sample_support_seconds={'fake': 300})
    return source, policy


class PreparedTests(unittest.TestCase):
    def test_pinned_intervals_not_premature_endpoint_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, policy = fabricated_package(root)
            output = root/'prepared'
            prepare(source, policy, output)
            manifest = json.loads((output/'native_input_manifest.json').read_text())
            self.assertEqual(manifest['row_count'], 300)
            self.assertEqual(manifest['time_label'], 'interval_end')
            self.assertEqual(manifest['daily_pet_delay_hours'], 24)
            self.assertEqual(manifest['interval_csv_sha256'], sha((output/'intervals.csv').read_bytes()))
            self.assertEqual(manifest['study_type'], 'idealized_retrospective_causal_hindcast')
            with self.assertRaises(FileExistsError):
                prepare(source, policy, output)

    def test_variables_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, policy = fabricated_package(root)
            (source/'variables.json').write_text('{"variables":[]}')
            with self.assertRaisesRegex(ValueError, 'variables.json mismatch'):
                prepare(source, policy, root/'prepared')

    def test_qc_failure_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, policy = fabricated_package(root)
            p = source/'manifest.json'
            m = json.loads(p.read_text())
            m['source_package_qc_status'] = 'fail'
            p.write_text(json.dumps(m))
            policy['manifest_sha256'] = sha(p.read_bytes())
            with self.assertRaisesRegex(ValueError, 'QC'):
                prepare(source, policy, root/'prepared')


if __name__ == '__main__':
    unittest.main()
