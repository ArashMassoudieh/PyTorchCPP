#!/usr/bin/env python3
"""Wait for bounded campaign, verify and analyze it once; never launches fitting."""
import argparse
from datetime import datetime,timezone
import json
import re
from pathlib import Path
import shutil
import time
from analyze_hydrofusion_all_methods import analyze,sha


def finalize(root):
    root=Path(root).resolve(strict=True)
    if (root/'analysis_closeout.json').exists():raise FileExistsError('Already finalized')
    plan=json.loads((root/'screen_plan.json').read_text())
    output=root/'all_method_diagnostics.json'
    report=analyze(plan['source_campaign'],output,root)
    expected_test=report['splits']['test']['n'];expected_val=report['splits']['validation']['n']
    if len(plan['jobs'])!=13 or sum(j['neural'] for j in plan['jobs'])!=12:
        raise ValueError('Unexpected extension budget')
    for job in plan['jobs']:
        log=(root/'fits'/job['name']/'fresh_reload.log').read_text()
        match=re.search(r'PASS all_test_rows=(\d+) production_target_free_checks=(\d+) max_abs=([0-9.eE+-]+) validation_rows=(\d+)',log)
        if not match or int(match[1])!=expected_test or int(match[4])!=expected_val or int(match[2])<2:
            raise ValueError('Missing/incomplete independent bulk reload proof: '+job['name'])
    # Preserve results rather than modifying a paper or declaring advancement.
    evidence=root/'executables';evidence.mkdir(exist_ok=True)
    for key in ('binary','verifier'):
        source=Path(plan[key]);target=evidence/source.name
        if target.exists():
            if sha(target)!=sha(source):raise ValueError('Archived executable mismatch')
        else:shutil.copy2(source,target)
    record=dict(completed_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(root/'screen_plan.json'),
                analysis_sha256=sha(output),analysis_script_sha256=report['script_sha256'],
                completed_neural_fits=12,completed_physical_runs=1,original_ffn_fits_repeated=0,
                all_test_rows_per_job=expected_test,validation_rows_per_job=expected_val,
                report=str(output),binary_sha256=sha(plan['binary']),verifier_sha256=sha(plan['verifier']),
                manuscript_modified=False,automatic_push=False,additional_training_authorized=False)
    with (root/'analysis_closeout.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    parent=root.parents[1]
    note=f"\n\n## All-method campaign completed by saved pipeline — {record['completed_utc']}\n\nCodex-owned campaign, not Claude progress. All12 approved extension neural fits and1fixed physical run completed. All-row fresh reload and exact support/scaler/q_ref checks passed before this analysis. Original six FFN fits were reused without refitting. Read `{output}` and its Markdown companion for validation/test NSE,KGE,bias,flow-duration,training-threshold regimes and paired seeds. `analysis_closeout.json` pins the machine-readable result/executables. Scientific interpretation and any scoped results commit remain for supervisor review; this receipt does not self-accept manuscript promotion. No extra training or push. Persistent12hmonitor remainsACTIVE.\n"
    with (parent/'CODEX_SUPERVISION_LOG.md').open('a') as f:f.write(note)
    state_path=parent/'CODEX_SUPERVISION_STATE.json';state=json.loads(state_path.read_text())
    state['research_budget_status'].update(completed_extension_neural_fits=12,completed_extension_physical_runs=1,remaining_approved_extension_fits=0)
    state['scope_note']='All approved extension jobs completed and hash-checked all-method analysis saved; independent scientific review/commit pending. No repeat FFN fitting, paper edit, push or extra budget.'
    state['codex_checkpoint']='All-method extension completed; saved pipeline analysis pending supervisor interpretation'
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    print('COMPLETE '+str(output),flush=True)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--wait',action='store_true');a=p.parse_args()
    try:
        if a.wait:
            # Ceiling accommodates thirteen bounded 8h jobs and 1h verifications.
            deadline=time.monotonic()+13*9*3600+600
            while not (a.root/'screen_completion.json').exists():
                if (a.root/'screen_failure.json').exists():raise RuntimeError('Campaign failed; no retry or analysis promotion')
                if time.monotonic()>deadline:raise TimeoutError('Bounded campaign completion wait expired')
                time.sleep(10)
        finalize(a.root)
    except BaseException as exc:
        failure=a.root/'analysis_failure.json'
        if not failure.exists():
            with failure.open('x') as f:json.dump(dict(error=str(exc),no_retry=True),f,indent=2);f.write('\n')
        raise
