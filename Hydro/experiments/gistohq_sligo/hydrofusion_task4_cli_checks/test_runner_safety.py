"""Runner filesystem/provenance checks using fake executables; no model fitting."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
COMPARE = HERE / 'compare_regression_against_baseline_build.sh'
CLI = HERE / 'run_hydrofusion_task4_cli_checks.sh'


class RunnerSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hydrofusion_runner_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def binary(self, name='fake binary', empty=False, mismatch=False):
        p = self.root / name
        row = '' if empty else ('ffn,0,test,1,1,2,-1\\n' if mismatch else 'ffn,0,test,1,1,1,0\\n')
        p.write_text('''#!/usr/bin/env python3
import pathlib,sys
batch=pathlib.Path(sys.argv[1]).name
if 'guard' in batch:
 print('this mode is synthetic-fixture-only for now');sys.exit(1)
out=pathlib.Path(sys.argv[2])
for mode in ['ffn','lstm','pinn','ffn_pinn','lstm_pinn']:
 p=out/('hydrofusion_regression_'+mode)/'predictions.csv'
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text('approach,index,split,x,observed,predicted,residual\\n'+''' + repr(row.replace('\\n', '\n')) + ''')
''')
        p.chmod(0o755)
        return p

    def run_script(self, script, *args, env=None):
        return subprocess.run(['bash', str(script), *map(str, args)], capture_output=True, text=True, env=env)

    def test_matching_rows_and_hashes(self):
        a, b = self.binary(), self.binary('baseline')
        out = self.root / 'new output'
        result = self.run_script(COMPARE, a, b, out)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('5 passed, 0 failed', result.stdout)
        saved = (out/'binary_hashes.txt').read_text()
        for p in [a, b]:
            self.assertIn(hashlib.sha256(p.read_bytes()).hexdigest(), saved)

    def test_existing_directory_preserved_by_both_runners(self):
        a = self.binary()
        out = self.root / 'existing'
        out.mkdir()
        sentinel = out/'preserve.txt'
        sentinel.write_text('untouched')
        for script, args in [(COMPARE, [a, a, out]), (CLI, [a, out])]:
            result = self.run_script(script, *args)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'untouched')

    def test_input_binary_inside_output_preserved(self):
        a = self.binary()
        original = a.read_bytes()
        result = self.run_script(COMPARE, a, a, self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(a.read_bytes(), original)

    def test_identical_header_only_files_fail(self):
        a = self.binary(empty=True)
        result = self.run_script(COMPARE, a, a, self.root/'empty')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('5 failed', result.stdout)

    def test_prediction_difference_fails(self):
        result = self.run_script(COMPARE, self.binary(), self.binary('different', mismatch=True), self.root/'diff')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('5 failed', result.stdout)

    def test_cli_guard_and_success_orchestration(self):
        result = self.run_script(CLI, self.binary(), self.root/'cli')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('5 passed, 0 failed', result.stdout)
        self.assertTrue((self.root/'cli'/'binary_hashes.txt').is_file())

    def test_hash_failure_prevents_launch(self):
        a = self.binary()
        fake_bin = self.root/'bin'
        fake_bin.mkdir()
        bad = fake_bin/'sha256sum'
        bad.write_text('#!/bin/sh\nexit 9\n')
        bad.chmod(0o755)
        import os
        env = dict(os.environ, PATH=str(fake_bin)+os.pathsep+os.environ['PATH'])
        for script, args in [(COMPARE, [a, a, self.root/'hash_fail']), (CLI, [a, self.root/'cli_hash_fail'])]:
            result = self.run_script(script, *args, env=env)
            self.assertEqual(result.returncode, 9)
            self.assertFalse(Path(args[-1]).exists())


if __name__ == '__main__':
    unittest.main()
