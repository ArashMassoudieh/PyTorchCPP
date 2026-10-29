import csv
import tempfile
import unittest
from pathlib import Path

from audit_development_predictions import audit, metrics


class AuditTests(unittest.TestCase):
    def write_rows(self, root, rows):
        path = Path(root)/'predictions.csv'
        with path.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['split','x','observed','predicted'])
            writer.writerows(rows)
        return path

    def test_metrics_and_excluded_tail(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [('train',0,1,1),('validation',1,1,1),
                    ('test',2,1,1),('test',4,3,3),
                    ('beyond_test_end',5,1,999)]
            report, _ = audit(self.write_rows(root,rows))
            self.assertEqual(report['test_metrics']['nse'],1)
            self.assertEqual(report['test']['missing_hours'],1)
            self.assertEqual(report['test_metrics']['n'],2)

    def test_invalid_boundaries_and_duplicates(self):
        for final in [('test',77016,1,1),('test',1,1,1),('test',0.5,1,1)]:
            with self.subTest(final=final), tempfile.TemporaryDirectory() as root:
                path=self.write_rows(root,[('train',0,1,1),('validation',1,1,1),final])
                with self.assertRaises(ValueError):
                    audit(path)

    def test_correlation_is_not_nse(self):
        result=metrics([(0,1,2),(1,2,4),(2,3,6)])
        self.assertAlmostEqual(result['pearson_r2'],1)
        self.assertLess(result['nse'],0)


if __name__ == '__main__':
    unittest.main()
