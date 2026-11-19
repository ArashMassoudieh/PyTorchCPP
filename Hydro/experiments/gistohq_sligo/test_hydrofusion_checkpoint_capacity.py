import json
import math
from pathlib import Path
import tempfile
import unittest
from analyze_hydrofusion_checkpoint_capacity import analyze

class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        folder=self.root/'frozen_support';folder.mkdir()
        for split,start in [('train',1),('validation',17544),('test',26304)]:
            (folder/f'{split}_support.csv').write_text('x\n'+''.join(f'{start+i}\n' for i in range(3)))
            (folder/f'{split}_observations.csv').write_text('x,observed\n'+''.join(f'{start+i},{q}\n' for i,q in enumerate([1,2,10])))
        self.capacity=self.root/'capacity.csv'
        self.header='bundle,approach,target_mode,q_ref,minimum_transformed_output,maximum_transformed_output,parameters\n'
        self.row=f'ffn_direct,ffn_causal_forecast,direct_log_q,1,-1,{math.log1p(2)},705\n'
        self.capacity.write_text(self.header+self.row)
    def test_bound_and_unavoidable_error(self):
        result=analyze(self.root,self.capacity,self.root/'report.json');m=result['models'][0]
        self.assertAlmostEqual(m['ceiling_mm_h'],2)
        self.assertEqual(m['splits']['test']['count_above_ceiling'],1)
        self.assertAlmostEqual(m['splits']['test']['minimum_sse_from_ceiling_alone'],64)
    def test_duplicate_rejected(self):
        self.capacity.write_text(self.header+self.row+self.row)
        with self.assertRaisesRegex(ValueError,'Duplicate'):analyze(self.root,self.capacity,self.root/'report.json')
    def test_nonfinite_rejected(self):
        self.capacity.write_text(self.header+'ffn_direct,ffn_causal_forecast,direct_log_q,1,-1,nan,705\n')
        with self.assertRaisesRegex(ValueError,'Invalid'):analyze(self.root,self.capacity,self.root/'report.json')
    def test_observation_support_mismatch(self):
        (self.root/'frozen_support/test_support.csv').write_text('x\n26304\n')
        with self.assertRaisesRegex(ValueError,'support mismatch'):analyze(self.root,self.capacity,self.root/'report.json')
    def test_existing_output_preserved(self):
        output=self.root/'report.json';output.write_text('preserve')
        with self.assertRaises(FileExistsError):analyze(self.root,self.capacity,output)
        self.assertEqual(output.read_text(),'preserve')

if __name__=='__main__':unittest.main()
