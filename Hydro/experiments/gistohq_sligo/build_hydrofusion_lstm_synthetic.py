#!/usr/bin/env python3
"""Build only the fabricated LSTM check, from hashed immutable source copies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.output.absolute()
    work = Path(str(output)+'.build')
    if output.exists() or work.exists():
        parser.error('Choose new binary/build paths; existing evidence is never overwritten.')
    source = Path('Hydro/experiments/gistohq_sligo/test_hydrofusion_lstm_synthetic.cpp')
    # This check has no out-of-line project dependencies. Snapshot its complete
    # transitive quoted-include closure before starting the compiler.
    inventory = {}
    def visit(path):
        name = path.relative_to(repo)
        if str(name) in inventory:
            return
        inventory[str(name)] = digest(path)
        for include in re.findall(r'^#include\s+"([^"]+)"', path.read_text(), re.M):
            target = (path.parent/include).resolve(strict=True)
            target.relative_to(repo)
            visit(target)
    visit(repo/source)
    work.mkdir(parents=True, exist_ok=False)
    frozen = work/'source'
    for name, expected in inventory.items():
        target = frozen/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo/name).read_bytes())
        if digest(target) != expected:
            raise RuntimeError('Source changed during copy: '+name)
    if any(digest(repo/name) != expected for name, expected in inventory.items()):
        raise RuntimeError('Source changed before compilation; no build launched.')
    torch = Path(os.environ.get('LIBTORCH_PATH', '/mnt/3rd900/Projects/libtorch')).resolve()
    command = [os.environ.get('CXX','g++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
               '-isystem', str(torch/'include'), '-isystem', str(torch/'include/torch/csrc/api/include'),
               '-D_GLIBCXX_USE_CXX11_ABI=1', str(frozen/source), '-o', str(output),
               '-L'+str(torch/'lib'), '-Wl,-rpath,'+str(torch/'lib'), '-ltorch', '-ltorch_cpu', '-lc10', '-lpthread']
    with (work/'build.log').open('w') as log:
        status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    manifest = {'head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
                'sources':inventory, 'command':command, 'exit_code':status}
    if status == 0:
        manifest['binary_sha256'] = digest(output)
    (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if status:
        print((work/'build.log').read_text())
    raise SystemExit(status)


if __name__ == '__main__':
    main()
