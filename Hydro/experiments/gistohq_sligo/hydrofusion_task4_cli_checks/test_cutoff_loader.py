"""Exact integer configuration parsing through the real loader, without Torch."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


class ExactCutoffLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='hydro_cutoff_loader_')
        cls.root = Path(cls.temp.name)
        source = cls.root/'probe.cpp'
        source.write_text('#include "Hydro/evaluation/experiment_loader.h"\n'
                          '#include <iostream>\n'
                          'int main(int n,char**a){if(n!=2)return 2;try{auto e=HydroExperimentLoader().loadConfig(a[1]);'
                          'std::cout<<e.config.train_end_x;}catch(const std::exception&e){std::cerr<<e.what();return 1;}}\n')
        cls.exe = cls.root/'probe'
        subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=undefined,float-cast-overflow',
                        '-fno-sanitize-recover=all','-I'+str(REPO),str(source),
                        str(REPO/'Hydro/evaluation/experiment_loader.cpp'),'-o',str(cls.exe)],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe(self, value, field='train_end_x', remove=False):
        config = json.loads((HERE/'config.json').read_text())
        if remove:
            config.pop(field,None)
        else:
            config[field] = value
        p = self.root/'config.json'
        p.write_text(json.dumps(config))
        return subprocess.run([str(self.exe),str(p)],text=True,capture_output=True)

    def test_integer_and_integral_float_remain_compatible(self):
        for value in [150,150.0,9223372036854775807]:
            with self.subTest(value=value):
                result = self.probe(value)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(int(result.stdout),int(value))

    def test_missing_cutoff_retains_unset_sentinel(self):
        result = self.probe(None,remove=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'-1')

    def test_invalid_indices_rejected_before_cast(self):
        for value in [150.5,1e40,9223372036854775808,-2,float('nan'),float('inf'),'150junk',None]:
            with self.subTest(value=value):
                result = self.probe(value)
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertIn('exact int64',result.stderr)
                self.assertNotIn('runtime error:',result.stderr)

    def test_other_int64_counts_cannot_overflow_either(self):
        for field in ['multiscale_daily_history_days','multiscale_daily_hidden_size','regional_per_basin_sequence_count']:
            result = self.probe(1e40,field)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertIn('exact int64',result.stderr)


if __name__ == '__main__':
    unittest.main()
