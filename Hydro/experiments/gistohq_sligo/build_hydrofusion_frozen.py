#!/usr/bin/env python3
"""Build from a stable source-only copy; never compile against live edited headers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_SUFFIXES = {'.h', '.hpp', '.cpp', '.c', '.cc', '.cxx', '.inc', '.ipp', '.tpp', '.py', '.sh', '.pro', '.pri'}


def inventory(repo):
    paths = set()
    for base in [repo, repo / 'Utilities']:
        raw = subprocess.check_output(['git', '-C', str(base), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
        for name in raw.decode().split('\0'):
            p = base / name
            if name and p.is_file() and p.suffix in SOURCE_SUFFIXES:
                paths.add(p.relative_to(repo))
    return {str(p): hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}


def compile_identity(makefile):
    names = {'CXX','DEFINES','CXXFLAGS','INCPATH','LINK','LFLAGS','LIBS'}
    result = {}
    for line in Path(makefile).read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            if key.strip() in names:
                result[key.strip()] = value.strip()
    if set(result) != names:
        raise RuntimeError('Incomplete qmake compile/link identity.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('script', choices=['build_hydrofusion_ffn_engine_integration_test.sh',
                                         'build_hydrofusion_batch_bundle_reload_check.sh', 'HydroBatch'])
    parser.add_argument('output')
    parser.add_argument('--reuse-batch', help='Reuse verified unchanged objects from a prior frozen build')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = Path(args.output).absolute()
    work = Path(str(output) + '.build')
    if output.exists() or work.exists():
        parser.error('Output binary/build evidence must not already exist; select a new path.')
    lock = Path('/tmp') / ('hydrofusion_source_build_' + hashlib.sha256(str(repo).encode()).hexdigest()[:16] + '.lock')
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        before = inventory(repo)
        work.mkdir(parents=True, exist_ok=False)
        frozen = work / 'source'
        for name, digest in before.items():
            target = frozen / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo/name, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Source changed during copy: ' + name)
        if before != inventory(repo):
            raise RuntimeError('Source inventory changed during snapshot; no build launched.')
        manifest = {'head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                    'utilities_head': subprocess.check_output(['git', '-C', str(repo/'Utilities'), 'rev-parse', 'HEAD'], text=True).strip(),
                    'sources': before, 'script': args.script, 'output': str(output)}
        env = dict(os.environ, HYDROFUSION_FROZEN_SOURCE_ROOT=str(frozen))
        with (work/'build.log').open('w') as log:
            if args.script == 'HydroBatch':
                build = work / 'objects'
                build.mkdir()
                command = ['qmake', str(frozen/'HydroBatch.pro'), 'CONFIG+=release',
                           'LIBTORCH_PATH='+os.environ.get('LIBTORCH_PATH', '/mnt/3rd900/Projects/libtorch')]
                status = subprocess.run(command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if status == 0 and args.reuse_batch:
                    prior_binary = Path(args.reuse_batch).absolute()
                    prior = Path(str(prior_binary)+'.build')
                    old = json.loads((prior/'source_manifest.json').read_text())
                    if old['exit_code'] != 0 or hashlib.sha256(prior_binary.read_bytes()).hexdigest() != old['binary_sha256']:
                        raise RuntimeError('Prior build binary changed.')
                    if compile_identity(prior/'objects'/'Makefile') != compile_identity(build/'Makefile'):
                        raise RuntimeError('Cannot reuse objects across changed compiler/defines/includes/link flags.')
                    critical_suffixes = {'.h','.hpp','.inc','.ipp','.tpp','.pro','.pri'}
                    critical = {k:v for k,v in before.items() if Path(k).suffix in critical_suffixes}
                    old_critical = {k:v for k,v in old['sources'].items() if Path(k).suffix in critical_suffixes}
                    generator = 'Hydro/generate_reduced_reservoir_supervised_wrappers.py'
                    if critical != old_critical or before[generator] != old['sources'][generator]:
                        raise RuntimeError('Cannot reuse objects across changed headers/project/generator inputs.')
                    changed = {Path(k).stem for k,v in before.items() if Path(k).suffix in {'.cpp','.c','.cc','.cxx'} and old['sources'].get(k) != v}
                    if 'ffn_wrapper' in changed: changed.add('ffn_wrapper_reduced_reservoir')
                    if 'lstmnetworkwrapper' in changed: changed.add('lstmnetworkwrapper_reduced_reservoir')
                    reused = {}
                    for obj in (prior/'objects').glob('*.o'):
                        if obj.stem not in changed:
                            shutil.copy2(obj, build/obj.name)
                            reused[obj.name] = hashlib.sha256(obj.read_bytes()).hexdigest()
                    manifest['reused_from'] = str(prior_binary)
                    manifest['reused_objects'] = reused
                if status == 0:
                    status = subprocess.run(['make', '-j2'], cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if status == 0:
                    shutil.copy2(build/'HydroBatch', output)
            else:
                script = frozen / 'Hydro/experiments/gistohq_sligo' / args.script
                status = subprocess.run(['bash', str(script), str(output)], cwd=frozen, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        manifest['generated_sources'] = {str(x.relative_to(frozen)): hashlib.sha256(x.read_bytes()).hexdigest() for x in (frozen/'Hydro/generated').glob('*.cpp')}
        if args.script == 'HydroBatch':
            manifest['compile_identity'] = compile_identity(work/'objects'/'Makefile')
        manifest['exit_code'] = status
        if status == 0:
            manifest['binary_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        (work/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('Build evidence:', work, flush=True)
        if status:
            print((work/'build.log').read_text()[-6000:], file=sys.stderr)
        return status


if __name__ == '__main__':
    sys.exit(main())
