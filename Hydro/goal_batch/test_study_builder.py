import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import study_builder as b
class Tests(unittest.TestCase):
 def test_one_year(self):self.assertEqual(b.boundaries(1,9),(6551,8759))
 def test_leap_year(self):self.assertEqual(b.boundaries(2,9),(6551,17543))
 def test_ten_year_before_any_input_access(self):
  with tempfile.TemporaryDirectory() as d,patch.object(b,'load',side_effect=AssertionError('must not read')):
   with self.assertRaisesRegex(ValueError,'RESERVED_DATA_GATE'):b.build(Path(d)/'new',*([Path('/nonexistent')]*5),years=10)
   self.assertFalse((Path(d)/'new').exists())
 def test_unsupported_adapter_before_inputs(self):
  with patch.object(b,'load',side_effect=AssertionError('must not read')):
   with self.assertRaisesRegex(ValueError,'FEATURE_PREFLIGHT_PERIOD_GATE'):b.build(Path('/tmp/unused'),*([Path('/nonexistent')]*5),years=3)
 def test_invalid_periods(self):
  for years,months in [(0,9),(11,9),(True,9),(1,0),(1,12),(1,True)]:
   with self.assertRaises(ValueError):b.boundaries(years,months)
 def test_source_ending_at_seal_rejected_before_crop(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'input';f.touch()
   with patch.object(b,'load',return_value=dict(first_interval_x=0,end_interval_x_exclusive=77016)),patch.object(b,'crop',side_effect=AssertionError('must not read intervals')):
    with self.assertRaisesRegex(ValueError,'COVERAGE_GATE'):b.build(Path(d)/'output',f,f,f,f,f)
 def test_invalid_goals_before_inputs(self):
  with self.assertRaises(ValueError):b.build(Path('/tmp/unused'),*([Path('/nonexistent')]*5),nse=float('nan'))
 def test_unbounded_widths_before_inputs(self):
  with self.assertRaisesRegex(ValueError,'width inventory'):b.build(Path('/tmp/unused'),*([Path('/nonexistent')]*5),widths=(32,128))
 def test_unknown_methods(self):
  with self.assertRaisesRegex(ValueError,'supported methods'):b.build(Path('/tmp/unused'),*([Path('/nonexistent')]*5),families=['invented'])
if __name__=='__main__':unittest.main()
