import json,tempfile,unittest
from pathlib import Path
from finish_mrms_unvisited_tail import prepare
class TailTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.r=Path(self.t.name);self.old=self.r/'old';self.old.mkdir();(self.old/'records').mkdir();(self.old/'provenance').mkdir()
  for n in ['extract_mrms_preissue_verified.py','mrms_preissue_geometry.py']:(self.old/'provenance'/n).write_text('# fixture\n')
  for x in [1,2]:(self.old/'records'/f'{x:05d}.json').write_text('{}')
  (self.old/'failure.json').write_text(json.dumps(dict(error='Extraction wall ceiling',completed_hours=2)))
  (self.old/'progress.json').write_text(json.dumps(dict(completed_hours=1,compressed_bytes=100)))
  (self.old/'plan.json').write_text(json.dumps(dict(jobs=[{}, {}, {}],pins={},limits=dict(compressed_bytes=30000000000,per_response_bytes=5000000))))
 def test_only_unvisited_tail_and_conservative_bytes(self):
  prepare(self.r/'new',self.old);p=json.loads((self.r/'new/plan.json').read_text());self.assertEqual(set(p['prior_records']),{'1','2'});self.assertEqual(p['continuation']['unvisited_hours'],1);self.assertEqual(p['prior_bytes'],5032868);self.assertEqual(p['continuation']['parent_wall_seconds'],3600)
 def test_non_wall_failure_rejected(self):
  (self.old/'failure.json').write_text(json.dumps(dict(error='different',completed_hours=2)))
  with self.assertRaises(ValueError):prepare(self.r/'new',self.old)
 def test_missing_previous_identity_rejected(self):
  (self.old/'records/00001.json').unlink()
  with self.assertRaises(ValueError):prepare(self.r/'new',self.old)
if __name__=='__main__':unittest.main()
