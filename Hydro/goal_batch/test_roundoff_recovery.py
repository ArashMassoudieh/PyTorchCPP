import tempfile,unittest
from pathlib import Path
from controller import write
from recover_roundoff_stop import recover
class RecoveryGuards(unittest.TestCase):
 def reject(self,state):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'parent';p.mkdir();write(p/'plan.json',{},True);write(p/'state.json',state,True)
   with self.assertRaisesRegex(ValueError,'specific first-fit'):recover(p,Path(d)/'new',Path(d)/'absent_source')
   self.assertFalse((Path(d)/'new').exists())
 def test_active_job_not_recovered(self):self.reject(dict(status='RUNNING'))
 def test_real_training_failure_not_recovered(self):self.reject(dict(status='FAILED',error='ENGINE_FAILED',charged_fits=1))
 def test_multiple_charged_fits_not_recovered(self):self.reject(dict(status='FAILED',error='Fresh reload/exported metrics differ',charged_fits=2))
if __name__=='__main__':unittest.main()
