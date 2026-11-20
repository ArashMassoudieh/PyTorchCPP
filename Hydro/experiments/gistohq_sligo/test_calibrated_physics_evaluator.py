"""Synthetic-only executable checks; no native calibration or research fits."""
import csv,itertools,subprocess,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class CalibrationEvaluator(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.build=tempfile.TemporaryDirectory();cls.binary=Path(cls.build.name)/'evaluator'
  subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(HERE/'hydrofusion_calibrate_valid_time_physics.cpp'),'-o',str(cls.binary)],check=True)
 @classmethod
 def tearDownClass(cls):cls.build.cleanup()
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.rows=[[x,'never_parse_Q',1+x%7,0,50,1,0,.1,1,0] for x in list(range(90,145))+list(range(17543,17611))+[26304]]
  self.train=[[x,.2+x%3] for x in range(101,111)];self.val=[[x,.1+x%5] for x in range(17570,17581)]
  self.grid=[[i,*p] for i,p in enumerate(itertools.product((1.,2.),(.04,.06,.1),(.4,.7),(.5,1.)))]
 def run_case(self,ok=True):
  def write(name,header,rows):
   p=self.root/name
   with p.open('w') as f:w=csv.writer(f,lineterminator='\n');w.writerow(header.split(','));w.writerows(rows)
   return p
  canonical=write('rows.csv','x,q,p,t,rh,wind,solar,pet,forcing_valid,discharge_valid',self.rows)
  train=write('train.csv','x,observed',self.train);val=write('validation.csv','x,observed',self.val);grid=write('grid.csv','index,fast,slow,fraction,gain',self.grid)
  p=subprocess.run([str(self.binary),str(canonical),str(train),str(val),str(grid),str(self.root/'output.csv')],capture_output=True,text=True)
  self.assertEqual(p.returncode==0,ok,p.stdout+p.stderr);return p
 def test_all24_and_no_canonical_Q_or_future_forcing_read(self):
  self.rows[-1][2]='nan';p=self.run_case();self.assertIn('test_scores=0 canonical_Q_parsed=0',p.stdout)
  with (self.root/'output.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),24)
 def test_zero_rain_scores_match_independent_equations(self):
  for r in self.rows:r[2]=0;r[7]=0
  self.run_case()
  with (self.root/'output.csv').open() as f:scores=list(csv.DictReader(f))
  obs=[r[1] for r in self.val];m=sum(obs)/len(obs);nse=1-sum(v*v for v in obs)/sum((v-m)**2 for v in obs)
  for row in scores:
   self.assertAlmostEqual(float(row['validation_nse']),nse,places=12);self.assertAlmostEqual(float(row['validation_mae']),m,places=12);self.assertEqual(float(row['validation_pbias_percent']),-100.)
 def test_sealed_timestamp_rejected_before_payload(self):
  self.rows[-1]=[77016]+['not_numeric']*9;self.assertIn('timestamp',self.run_case(False).stderr)
 def test_test_truth_rejected_before_observed_parse(self):
  self.val.append([26304,'not_numeric']);self.assertIn('truth split',self.run_case(False).stderr)
 def test_duplicate_canonical_identity(self):
  self.rows.insert(1,self.rows[0]);self.assertIn('Unsorted',self.run_case(False).stderr)
 def test_trailing_numeric_junk(self):
  self.rows[0][0]='90junk';self.assertIn('Invalid number',self.run_case(False).stderr)
 def test_candidate_grid_expansion_rejected(self):
  self.grid.append(self.grid[0]);self.assertIn('Exactly24',self.run_case(False).stderr)
 def test_changed_candidate_rejected(self):
  self.grid[0][-1]=2.;self.assertIn('grid changed',self.run_case(False).stderr)
 def test_negative_truth_rejected(self):
  self.train[0][1]=-1.;self.assertIn('Negative truth',self.run_case(False).stderr)
 def test_refuses_existing_output(self):
  (self.root/'output.csv').write_text('keep');self.run_case(False);self.assertEqual((self.root/'output.csv').read_text(),'keep')
if __name__=='__main__':unittest.main()
