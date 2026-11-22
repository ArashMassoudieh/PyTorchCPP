import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from prepare_hydrofusion_search_slice import prepare, EPOCH

class SearchSliceTests(unittest.TestCase):
    def source(self, root):
        csv = root/'intervals.csv'
        csv.write_text('epoch_seconds,q_m3_s\n'+''.join(f'{EPOCH+x*3600},'+('2' if x<3 else 'UNREADABLE_TEST_Q')+'\n' for x in range(6)))
        m=dict(interval_csv='intervals.csv',interval_csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
               end_interval_x_exclusive=6,first_interval_x=0,daily_pet=[])
        p=root/'native_input_manifest.json';p.write_text(json.dumps(m));return p
    def test_crop_before_observation_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=self.source(r);out=prepare(p,r/'new',3)
            m=json.loads(out.read_text());text=(out.parent/'intervals.csv').read_text()
            self.assertNotIn('UNREADABLE',text);self.assertEqual(m['row_count'],3)
            self.assertEqual(m['target_cutoff_x'],3);self.assertEqual(m['execution_scope'],'train_validation_only')
            self.assertIn('UNREADABLE', (r/'intervals.csv').read_text())
            with self.assertRaises(FileExistsError):prepare(p,r/'new',3)
    def test_seal_hash_and_cutoff(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=self.source(r)
            for n in [True,0,77016,1.5]:
                with self.assertRaises(ValueError):prepare(p,r/'bad',n)
            (r/'intervals.csv').write_text('changed')
            with self.assertRaises(ValueError):prepare(p,r/'bad',3)

if __name__ == '__main__':unittest.main()
