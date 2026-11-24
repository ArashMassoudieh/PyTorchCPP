import json,tempfile,unittest
from pathlib import Path
from audit_joint_study_readiness import inspect_metadata
class ReadinessTests(unittest.TestCase):
 def test_metadata_only_and_blocks_ten_year(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);values=[{'catchment_area_m2':23754600,'source_failed_qc_rule_ids':['temporal.study_period_coverage'],'study_start':'2015-01-01','study_end':'2024-12-31'}, {'catchment_area_m2':16737500,'row_count':26303,'end_interval_x_exclusive':26303,'target_cutoff_x':26303,'execution_scope':'train_validation_only'},{}]
   paths=[]
   for i,v in enumerate(values):
    p=root/f'{i}.json';p.write_text(json.dumps(v));paths.append(p)
   # No CSV even exists: audit must succeed with only metadata.
   r=inspect_metadata(*paths);self.assertEqual(r['ten_year_expected_hours'],87672);self.assertEqual(r['observation_csv_reads'],0);self.assertFalse(r['training_launched']);self.assertGreaterEqual(len(r['blockers']),5)
 def test_non_json_rejected_before_read(self):
  with self.assertRaises(ValueError):inspect_metadata(Path('/nonexistent/observations.csv'),None,None)
if __name__=='__main__':unittest.main()
