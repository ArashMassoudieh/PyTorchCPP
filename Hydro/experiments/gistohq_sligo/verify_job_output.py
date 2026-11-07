#!/usr/bin/env python3
"""Read-only structural verification of a development job export.

Checks an actual ZIP directory/CRC and LibTorch-style member layout WITHOUT
unpickling or loading code. Validates prediction timestamps before numeric
values, exact configured split intervals, duplicate rows and residuals.

Optional independently frozen config/support declarations permit exact identity
and requested-support checks. Without them the report explicitly remains
partial. ZIP integrity is never semantic model-load or reproduction evidence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path, PurePosixPath

SEAL_X = 77016
REQUIRED_PREDICTIONS_COLUMNS = {'approach', 'index', 'split', 'x', 'observed', 'predicted', 'residual'}
SPLITS = {'train', 'validation', 'test', 'beyond_test_end'}


class JobOutputVerificationError(ValueError):
    pass


def _integer(value, label, minimum=0, maximum=SEAL_X-1):
    if isinstance(value, bool):
        raise JobOutputVerificationError(f'{label}: boolean is not an integer')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise JobOutputVerificationError(f'{label}: invalid integer') from exc
    if not math.isfinite(number) or number != int(number) or not minimum <= number <= maximum:
        raise JobOutputVerificationError(f'{label}: expected finite integer in [{minimum},{maximum}]')
    return int(number)


def _checkpoint_filename(approach):
    return ''.join(c.lower() if c.isalnum() else '_' for c in approach)+'.pt'


def _archive(path):
    if path.stat().st_size < 1024:
        raise JobOutputVerificationError('checkpoint too small for a registered multiscale export')
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [v.filename for v in members]
            if not names or len(names) != len(set(names)):
                raise JobOutputVerificationError('empty/duplicate ZIP members')
            if any(v.flag_bits & 1 or v.file_size > 512*1024*1024 for v in members):
                raise JobOutputVerificationError('encrypted or oversized ZIP member')
            if sum(v.file_size for v in members) > 2*1024*1024*1024:
                raise JobOutputVerificationError('ZIP uncompressed size exceeds verifier limit')
            if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
                raise JobOutputVerificationError('unsafe ZIP member path')
            roots = {n[:-len('data.pkl')] for n in names if n.endswith('/data.pkl') or n == 'data.pkl'}
            if len(roots) != 1:
                raise JobOutputVerificationError('expected one LibTorch-style data.pkl root')
            prefix = roots.pop()
            if prefix+'version' not in names or not any(n.startswith(prefix+'data/') and not n.endswith('/') for n in names):
                raise JobOutputVerificationError('missing LibTorch-style version/tensor members')
            version = archive.read(prefix+'version').strip()
            if not version.isdigit() or int(version) < 1:
                raise JobOutputVerificationError('invalid archive version member')
            bad = archive.testzip()
            if bad is not None:
                raise JobOutputVerificationError(f'ZIP CRC failure: {bad}')
            return len(members)
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError) as exc:
        raise JobOutputVerificationError(f'invalid ZIP archive: {exc}') from exc


def verify_job_output(job_dir, *, approach, expected_d, required_splits=('validation','test'),
                      expected_config=None, expected_support=None):
    job_dir = Path(job_dir)
    paths = {'config':job_dir/'experiment_config.json', 'predictions':job_dir/'predictions.csv',
             'checkpoint':job_dir/'models'/_checkpoint_filename(approach)}
    for label,path in paths.items():
        if not path.is_file():
            raise JobOutputVerificationError(f'missing {label} file {path}')
    expected_d = _integer(expected_d, 'expected D', 1, 3650)
    try:
        config = json.loads(paths['config'].read_text())
    except (ValueError, OSError) as exc:
        raise JobOutputVerificationError('unreadable config') from exc
    if not isinstance(config, dict):
        raise JobOutputVerificationError('config must be an object')
    d = _integer(config.get('multiscale_daily_history_days'), 'config D', 1, 3650)
    if d != expected_d:
        raise JobOutputVerificationError('config identity mismatch: D')
    bounds = [_integer(config.get(k), k) for k in ['train_end_x','validation_end_x','test_end_x']]
    if not bounds[0] < bounds[1] < bounds[2]:
        raise JobOutputVerificationError('config split bounds must be chronological')
    if expected_config is not None:
        if not isinstance(expected_config, dict) or not expected_config:
            raise JobOutputVerificationError('expected config must be a nonempty frozen declaration')
        for k,v in expected_config.items():
            if k not in config or config[k] != v or isinstance(config[k],bool) != isinstance(v,bool):
                raise JobOutputVerificationError(f'config identity mismatch: {k}')
    required = set(required_splits)
    if not required or not required <= SPLITS:
        raise JobOutputVerificationError('invalid required split declaration')
    declared_support = None
    if expected_support is not None:
        if not isinstance(expected_support, dict) or not required <= set(expected_support) or not set(expected_support) <= SPLITS:
            raise JobOutputVerificationError('expected support must declare every required split')
        declared_support = {}
        for split,values in expected_support.items():
            if not isinstance(values, (list, tuple, set)):
                raise JobOutputVerificationError('expected support must contain timestamp arrays')
            times = [_integer(v, 'expected support x') for v in values]
            if len(times) != len(set(times)):
                raise JobOutputVerificationError('duplicate expected support timestamp')
            declared_support[split] = set(times)
    member_count = _archive(paths['checkpoint'])
    by_split = {}; seen = set(); indices = set(); last_x = -1
    with paths['predictions'].open(newline='') as f:
        reader = csv.DictReader(f)
        if not REQUIRED_PREDICTIONS_COLUMNS <= set(reader.fieldnames or []):
            raise JobOutputVerificationError('predictions missing required column')
        for row in reader:
            # Check every row's timestamp before approach selection or value access.
            x = _integer(row.get('x'), 'prediction timestamp')
            if row.get('approach') != approach:
                continue
            if x in seen or x <= last_x:
                raise JobOutputVerificationError('duplicate or unordered prediction timestamp')
            seen.add(x);last_x=x
            index = _integer(row.get('index'), 'prediction index', maximum=10**9)
            if index in indices:
                raise JobOutputVerificationError('duplicate prediction index')
            indices.add(index)
            split = row.get('split')
            actual = 'train' if x <= bounds[0] else 'validation' if x <= bounds[1] else 'test' if x <= bounds[2] else 'beyond_test_end'
            if split != actual:
                raise JobOutputVerificationError(f'prediction split interval mismatch at x={x}')
            try:
                o,p,r = (float(row[k]) for k in ['observed','predicted','residual'])
            except (ValueError,TypeError,KeyError) as exc:
                raise JobOutputVerificationError('non-finite or unparseable prediction values') from exc
            if not all(math.isfinite(v) for v in [o,p,r]):
                raise JobOutputVerificationError('non-finite or unparseable prediction values')
            if not math.isclose(r,p-o,rel_tol=1e-5,abs_tol=1e-6):
                raise JobOutputVerificationError('residual must equal predicted minus observed')
            by_split.setdefault(split,set()).add(x)
    if not seen:
        raise JobOutputVerificationError('header-only stub-output or no target approach rows')
    if not required <= set(by_split):
        raise JobOutputVerificationError('missing required split rows')
    if declared_support is not None:
        for split,times in declared_support.items():
            if by_split.get(split,set()) != times:
                raise JobOutputVerificationError(f'exact support mismatch: {split}')
    return {'job_dir':str(job_dir),'approach':approach,'expected_d':expected_d,'config_declared_d':d,
            'checkpoint_size_bytes':paths['checkpoint'].stat().st_size,
            'checkpoint_structurally_valid_archive':True,'checkpoint_zip_members':member_count,
            'checkpoint_semantically_loaded_via_real_inference_path':False,
            'full_scientific_verification':False,
            'independent_config_declaration_verified':expected_config is not None,
            'exact_declared_support_verified':declared_support is not None,
            'exact_support_splits':sorted(declared_support or {}),
            'rows_by_split':{k:len(v) for k,v in by_split.items()},'total_rows_for_approach':len(seen),'nonfinite_rows':0,
            'sha256':{label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in paths.items()}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_dir');parser.add_argument('--approach',default='lstm_multiscale')
    parser.add_argument('--expected-d',type=int,required=True)
    parser.add_argument('--required-split',action='append',dest='splits')
    parser.add_argument('--expected-config',type=Path,help='independently frozen JSON config')
    parser.add_argument('--expected-support',type=Path,help='JSON mapping split to exact eligible native x values')
    args=parser.parse_args(argv)
    report=verify_job_output(args.job_dir,approach=args.approach,expected_d=args.expected_d,
                            required_splits=args.splits or ('validation','test'),
                            expected_config=json.loads(args.expected_config.read_text()) if args.expected_config else None,
                            expected_support=json.loads(args.expected_support.read_text()) if args.expected_support else None)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
