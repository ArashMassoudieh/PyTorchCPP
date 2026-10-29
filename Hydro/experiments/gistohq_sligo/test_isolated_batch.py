import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_isolated_batch import run, validate_generated_wrapper_build


class BatchTests(unittest.TestCase):
    def test_stale_generated_wrapper_and_binary_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); hydro=root/'Hydro'
            (hydro/'models').mkdir(parents=True); (hydro/'generated').mkdir()
            generator=hydro/'generate_reduced_reservoir_supervised_wrappers.py'
            generator.write_text('generator')
            sources=[]; generated=[]
            for name, out in (('ffn_wrapper.cpp','ffn_wrapper_reduced_reservoir.cpp'),
                              ('lstmnetworkwrapper.cpp','lstmnetworkwrapper_reduced_reservoir.cpp')):
                source=hydro/'models'/name; source.write_text('source'); sources.append(source)
                target=hydro/'generated'/out; target.write_text('generated'); generated.append(target)
            binary=root/'HydroBatch'; binary.write_text('binary')
            now=1_800_000_000_000_000_000
            import os
            for path in [generator,*sources]: os.utime(path,ns=(now,now))
            for path in generated: os.utime(path,ns=(now+1,now+1))
            os.utime(binary,ns=(now+2,now+2))
            validate_generated_wrapper_build(binary,root)
            os.utime(sources[1],ns=(now+3,now+3))
            with self.assertRaisesRegex(ValueError,'Stale generated wrapper'):
                validate_generated_wrapper_build(binary,root)
            os.utime(generated[1],ns=(now+4,now+4))
            with self.assertRaisesRegex(ValueError,'Stale HydroBatch binary'):
                validate_generated_wrapper_build(binary,root)

    def test_verified_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'config.json';config.write_text('{}')
            binary=root/'binary';binary.write_text('fake executable for test')
            job=dict(id='case',config=str(config),batch='unused',
                     bounds=dict(train_end_x=0,validation_end_x=1,test_end_x=2),
                     config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
            def fake_run(*args,**kwargs):
                out=root/'case'/'case';out.mkdir()
                (out/'predictions.csv').write_text('split,x,observed,predicted\ntrain,0,1,1\nvalidation,1,2,2\ntest,2,3,3\n')
            with patch('run_isolated_batch.subprocess.run',side_effect=fake_run) as execute:
                result=run(job,binary,root)
                self.assertEqual(result['status'],'complete')
                self.assertEqual(run(job,binary,root)['status'],'verified_resume')
                self.assertEqual(execute.call_count,1)
                config.write_text('{"changed":true}')
                with self.assertRaises(ValueError):run(job,binary,root)


if __name__=='__main__':unittest.main()
