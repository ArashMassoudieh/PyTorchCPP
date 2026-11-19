import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_gauge_alignment_review as review


class GaugeReviewTest(unittest.TestCase):
    def test_identity_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.csv'
            path.write_text('epoch_seconds,q\n1420070400,1\n1420074000,2\n')
            review.validate_identity_csv(path,0,2)

    def test_bad_or_incomplete_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.csv'
            for rows in ['1420070400,nan\n', '1420070400junk,nan\n1420074000,1\n',
                         '1420070400,1\n1697328000,nan\n',
                         '1420070400,1\n1420070400,2\n']:
                path.write_text('epoch_seconds,q\n'+rows)
                with self.assertRaises(ValueError):review.validate_identity_csv(path,0,2)

    def test_seal_precedes_csv_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)
            (source/'native_input_manifest.json').write_text(json.dumps({
                'schema':'HydroFusionPreparedNative-v1','first_interval_x':0,
                'end_interval_x_exclusive':77016,'row_count':77016,'interval_csv':'intervals.csv'}))
            with patch.object(review,'digest',side_effect=AssertionError('CSV touched')):
                with self.assertRaises(ValueError):review.prepare(source,source,source,source/'new')
            self.assertFalse((source/'new').exists())

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);marker=output/'keep';marker.write_text('original')
            with self.assertRaises(FileExistsError):review.prepare('missing','missing','missing',output)
            self.assertEqual(marker.read_text(),'original')


if __name__=='__main__':unittest.main()
