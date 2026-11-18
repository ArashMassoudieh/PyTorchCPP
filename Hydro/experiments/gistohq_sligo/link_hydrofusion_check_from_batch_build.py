#!/usr/bin/env python3
"""Link the reload verifier against one completed, frozen HydroBatch build.

Reuses that build's exact objects/defines/includes instead of recompiling live
sources or running another generator. The resulting verifier still runs in a
separate process. Build evidence and object hashes are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def digest(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
        return h.hexdigest()


def variables(text):
    values = {}
    pending = ''
    for line in text.splitlines():
        pending += line.rstrip('\\').strip() + ' '
        if line.endswith('\\'):
            continue
        if '=' in pending and not pending.lstrip().startswith('#'):
            name, value = pending.split('=', 1)
            if name.strip() in {'CXX', 'CXXFLAGS', 'DEFINES', 'INCPATH', 'LINK', 'LFLAGS', 'LIBS', 'OBJECTS'}:
                values[name.strip()] = shlex.split(value.strip())
        pending = ''
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('batch_binary')
    p.add_argument('output')
    p.add_argument('--check', choices=['reload','integration','legacy_inference','exporter'], default='reload')
    p.add_argument('--integration-test-source', help='Copy an updated test main; all production headers remain frozen')
    args = p.parse_args()
    binary, output = Path(args.batch_binary).absolute(), Path(args.output).absolute()
    build = Path(str(binary)+'.build')
    if output.exists() or Path(str(output)+'.link_manifest.json').exists():
        p.error('Select a new output path.')
    manifest = json.loads((build/'source_manifest.json').read_text())
    if manifest['exit_code'] != 0 or digest(binary) != manifest['binary_sha256']:
        p.error('Batch build incomplete or binary changed.')
    frozen = build/'source'
    for name, expected in manifest['sources'].items():
        if digest(frozen/name) != expected:
            p.error('Frozen source changed: '+name)
    objects_dir = build/'objects'
    v = variables((objects_dir/'Makefile').read_text())
    objects = [x for x in v['OBJECTS'] if Path(x).name != 'hydro_batch_runner.o']
    if len(objects) != len(v['OBJECTS'])-1:
        p.error('Expected exactly one HydroBatch main object.')
    sources = {
        'reload': 'Hydro/experiments/gistohq_sligo/hydrofusion_batch_bundle_reload_check.cpp',
        'integration': 'Hydro/experiments/gistohq_sligo/hydrofusion_ffn_engine_integration_test.cpp',
        'legacy_inference': 'Hydro/tests/inference_runner_test.cpp',
        'exporter': 'Hydro/tests/experiment_exporter_test.cpp'}
    main_source = frozen/sources[args.check]
    extra_includes = []
    if args.integration_test_source:
        if args.check != 'integration': p.error('Updated source option is test-main-only.')
        updated = Path(args.integration_test_source).resolve(strict=True)
        main_source = Path(str(output)+'.source.cpp')
        if main_source.exists(): p.error('Updated test source copy already exists.')
        main_source.write_bytes(updated.read_bytes())
        extra_includes = ['-I'+str(frozen/'Hydro/experiments/gistohq_sligo')]

    main_object = Path(str(output)+'.o')
    if main_object.exists():
        p.error('Verifier object already exists.')
    # These are the only Make placeholders used by this qmake project.
    flags = [x for x in v['CXXFLAGS'] if x != '$(DEFINES)']+v['DEFINES']+['-UNDEBUG']+v['INCPATH']
    libs = [x for x in v['LIBS'] if x != '$(SUBLIBS)']
    if any('$(' in x for x in flags+libs):
        p.error('Unexpected unexpanded Make variable.')
    # The batch binary does not use inference, so compile these three missing
    # evaluation objects once, against its SAME frozen headers and defines.
    supplemental = objects_dir/'inference_check_objects'
    key = hashlib.sha256(json.dumps({'sources':manifest['sources'],'flags':flags},sort_keys=True).encode()).hexdigest()
    cache = supplemental/'manifest.json'
    extra_sources = ['Hydro/evaluation/artifact_loader.cpp', 'Hydro/evaluation/inference_runner.cpp',
                     'Hydro/evaluation/shared_scale_inference.cpp']
    extra_objects = [supplemental/(Path(x).stem+'.o') for x in extra_sources]
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['key'] != key or any(digest(x) != saved['objects'][str(x)] for x in extra_objects):
            p.error('Supplemental object cache changed.')
    else:
        supplemental.mkdir(exist_ok=False)
        for src, obj in zip(extra_sources, extra_objects):
            subprocess.run(v['CXX']+flags+['-c',str(frozen/src),'-o',str(obj)],cwd=objects_dir,check=True)
        cache.write_text(json.dumps({'key':key,'objects':{str(x):digest(x) for x in extra_objects}},indent=2)+'\n')
    compile_command = v['CXX']+flags+extra_includes+['-c',str(main_source),'-o',str(main_object)]
    link_command = v['LINK']+v['LFLAGS']+['-o',str(output),str(main_object)]+objects+list(map(str,extra_objects))+libs
    subprocess.run(compile_command, cwd=objects_dir, check=True)
    subprocess.run(link_command, cwd=objects_dir, check=True)
    Path(str(output)+'.link_manifest.json').write_text(json.dumps({
        'batch_binary_sha256':digest(binary), 'source_sha256':digest(main_source),
        'binary_sha256':digest(output), 'objects':{x:digest(objects_dir/x) for x in objects},
        'supplemental_objects':{str(x):digest(x) for x in extra_objects},
        'compile_command':compile_command, 'link_command':link_command},indent=2)+'\n')
    print('Built verifier from frozen batch objects:',output)


if __name__ == '__main__':
    main()
