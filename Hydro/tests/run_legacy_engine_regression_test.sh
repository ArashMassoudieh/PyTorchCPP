#!/usr/bin/env bash
# Build and exercise actual legacy model/CLI paths using fabricated data only.
# Usage: run_legacy_engine_regression_test.sh [NEW_OUTPUT_DIRECTORY]
# No real package is opened; no research batch is launched.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_root="${1:-$(mktemp -d /tmp/hydropinn_legacy_regression.XXXXXX)}"
mkdir -p "$work_root"
work_root="$(cd "$work_root" && pwd)"
if [[ -e "$work_root/fixture" || -e "$work_root/build" ]]; then
  echo 'Refusing to reuse existing regression outputs.' >&2
  exit 2
fi
mkdir "$work_root/build"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$work_root/build"
qmake "$repo_root/verify_legacy_engine.pro" -o Makefile.test CONFIG+=debug 'QMAKE_CXXFLAGS_DEBUG=-O0 -g0' > "$work_root/qmake_test.log" 2>&1
make -f Makefile.test -j"${HYDRO_TEST_BUILD_JOBS:-4}" > "$work_root/build_test.log" 2>&1
./legacy_engine_regression_test "$work_root/fixture" > "$work_root/model_test.log" 2>&1
# The two targets use the same dependency objects; only the real CLI main is added.
qmake "$repo_root/HydroBatch.pro" -o Makefile.cli CONFIG+=debug 'QMAKE_CXXFLAGS_DEBUG=-O0 -g0' > "$work_root/qmake_cli.log" 2>&1
make -f Makefile.cli -j"${HYDRO_TEST_BUILD_JOBS:-4}" > "$work_root/build_cli.log" 2>&1
python3 - "$work_root" "$repo_root" <<'PY'
import csv, hashlib, json, math, shutil, subprocess, sys, tempfile
from pathlib import Path
r=Path(sys.argv[1]);batch_dir=Path(tempfile.mkdtemp(prefix='_legacy_cli_',dir=Path(sys.argv[2])/'Hydro/experiments/gistohq_sligo'));cases={'ffn':'ffn','physics':'pinn','residual':'ffn_pinn','highflow':'lstm','multiscale':'lstm_pinn','urban':'lstm_pinn'};results=[]
for case,mode in cases.items():
    cfg=r/'fixture/configs'/case/'experiment_config.json'
    shutil.copy2(cfg,batch_dir/f'{case}.json')
    batch=batch_dir/f'{case}.batch';batch.write_text(f'{mode} {case}.json\n')
    shutil.copy2(batch,r/f'{case}.batch')
    out=r/'cli_outputs'/case
    with (r/f'cli_{case}.log').open('w') as log:
        p=subprocess.run([str(r/'build/HydroBatch'),str(batch),str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if p.returncode:raise RuntimeError(f'{case}: CLI failed ({p.returncode}); inspect its log')
    job=out/case;export=json.loads((job/'experiment_config.json').read_text());count=0
    with (job/'predictions.csv').open() as f:
        for row in csv.DictReader(f):
            x=float(row['x'])
            if not(math.isfinite(x) and x==int(x) and 0<=x<77016):raise RuntimeError('bad synthetic timestamp')
            expected='train' if x<=export['train_end_x'] else 'validation' if x<=export['validation_end_x'] else 'test' if x<=export['test_end_x'] else 'beyond_test_end'
            if row['split']!=expected:raise RuntimeError(f'{case}: wrong split at {x}')
            if not all(math.isfinite(float(row[k])) for k in ('observed','predicted')):raise RuntimeError('nonfinite synthetic values')
            if case=='multiscale' and row['approach']!='lstm_multiscale':raise RuntimeError('multiscale export mislabeled')
            if case=='urban' and row['approach']!='lstm_pinn':raise RuntimeError('urban profile incorrectly relabeled multiscale')
            count+=1
    if not count:raise RuntimeError('empty exported predictions')
    summary=job/'batch_summary.csv'
    # Batch summary is written at the common CLI output root.
    summaries=list(out.glob('*summary*.csv'))
    if not summaries:raise RuntimeError('batch summary missing')
    for summary in summaries:
        with summary.open() as f:
            rows=list(csv.reader(f));width=len(rows[0])
            if any(len(row)!=width for row in rows[1:]):raise RuntimeError('summary field/header mismatch')
    results.append({'case':case,'mode':mode,'rows':count,'config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest()})
(r/'verification.json').write_text(json.dumps({'synthetic_only':True,'cli_cases':results,'model_checks_log':str(r/'model_test.log')},indent=2)+'\n')
shutil.rmtree(batch_dir)
print('PASS: six real CLI paths, exact exported split boundaries and multiscale/urban checkpoint labels.')
PY
printf 'Synthetic verification evidence: %s\n' "$work_root"
