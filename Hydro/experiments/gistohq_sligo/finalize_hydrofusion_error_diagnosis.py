#!/usr/bin/env python3
"""Finish read-only checkpoint diagnosis once exports exist; no training/push."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from statistics import mean
import time
from analyze_hydrofusion_error_diagnosis import analyze
from analyze_hydrofusion_all_methods import sha,verified_jobs


def finish(source,extension,root):
    root=Path(root).resolve(strict=True);source=Path(source);extension=Path(extension)
    verified_jobs(source);verified_jobs(extension)
    report=analyze(source,extension,root,root/'diagnosis.json')
    groups={}
    for j in report['neural'].values():groups.setdefault(j['family']+' / '+j['mode'],[]).append(j)
    lines=['# Error diagnosis: fixed physical recipe and neural flow compression','',
           'Read-only diagnosis of saved checkpoints. No refitting, calibration, post-hoc correction, or original output replacement.','',
           '## Physical input and volume accounting','',
           '| Split | Matched rainfall mm | PET demand mm | Effective input mm | Observed discharge mm | Input / observed |',
           '|---|---:|---:|---:|---:|---:|']
    for split,entry in report['physical'].items():
        s=entry['matched_hour_sums_mm'];lines.append(f"| {split} | {s['p_mm_h']:.3f} | {s['pet_mm_h']:.3f} | {s['effective_mm_h']:.3f} | {s['observed_mm']:.3f} | {entry['effective_input_to_observed_ratio']:.3f} |")
    lines += ['', 'These are sums over matched hours, not full-year basin closure. The fixed router uses gain1 and routes all max(P-PET,0) into two outflow stores. There is no infiltration/soil-water/runoff-generation process. runoff_coeff0.7 is the fast-store partition, NOT a 0.7 rainfall-runoff loss coefficient. Dry-hour PET cannot deplete stored water; even all PET subtraction alone would not explain away the input/observed gap.',
              '', 'The exact production ledger closes to roundoff. Native m³/s-to-mm/h conversion uses the declared23,754,600m² area and independently matches adapter output. This rules out a simple conversion/ledger arithmetic error in the checked path; it does not independently validate the delineated area against the gauge drainage polygon.',
              '', '## Confirmed target-aggregation mismatch','',
              'The router returns endpoint instantaneous flow, but observed targets are hourly means. A zero-storage one-hour unit rainfall probe gives endpoint0.47103316576920246 versus interval mean0.27202786292788839mm/h. State integration itself is exact. Any fix MUST introduce versioned reference/recipe semantics and preserve old checkpoint reconstruction; never silently change causal_two_reservoir_v1 under exported models.',
              '', '| Split | Original endpoint NSE | Interval-mean diagnostic NSE | Original PBIAS % | Interval-mean PBIAS % |', '|---|---:|---:|---:|---:|']
    for split,entry in report['physical'].items():
        a=entry['fixed_reference_metrics']['endpoint_q'];b=entry['fixed_reference_metrics']['interval_mean_q']
        lines.append(f"| {split} | {a['nse']:.6f} | {b['nse']:.6f} | {a['pbias_percent']:.3f} | {b['pbias_percent']:.3f} |")
    lines += ['', 'Fixed-equation diagnostic only, no calibrated candidate. Averaging does not fix the volume excess. The imposed24h meteorological lag also means the physical router sees yesterday\'s storm; its state is never corrected with observed Q. Lag is a disclosed conservative availability assumption, not a measured product latency. Do not substitute future/current unavailable forcing to improve scores.',
              '', '## Existing neural checkpoints','',
              '| Family / target | Mean training PBIAS % | Mean test PBIAS % | Test clipped fraction | Test transformed variance explained |',
              '|---|---:|---:|---:|---:|']
    for label,jobs in groups.items():
        train_bias=mean(j['splits']['train']['physical']['pbias_percent'] for j in jobs)
        test_bias=mean(j['splits']['test']['physical']['pbias_percent'] for j in jobs)
        clipped=mean(j['splits']['test']['clipped_fraction'] for j in jobs)
        explained=mean(j['splits']['test']['target_space']['variance_explained'] for j in jobs)
        lines.append(f'| {label} | {train_bias:.3f} | {test_bias:.3f} | {clipped:.6f} | {explained:.6f} |')
    lines += ['', 'Transformed variance explained is computed in each arm\'s own log-target space, not physical-discharge NSE or squared Pearson correlation. Residual/direct target spaces differ; do not rank models across these columns as if targets were identical.',
              '', 'Training and checkpoint selection minimize standardized log-space MSE, not physical flow-volume error or KGE. Exponentiating a log point estimate does not generally yield an unbiased arithmetic-flow mean. Saved training predictions and high-flow partitions test whether compression already exists before validation/test. This supports an objective/reference diagnosis but does not isolate one cause from representation, optimization, regularization and covariate shift.',
              '', 'Train-only Q90 is used for all tail diagnostics. Native matched test support has a larger high-flow share than training; this is descriptive development distribution shift, not a test-selected threshold. Full per-seed target-space, clipping and tail measurements are in diagnosis.json.',
              '', '## Next bounded work','',
              '1. Implement and synthetically validate an explicitly versioned interval-mean physical reference, preserving all v1 checkpoint reloads. This is a correctness fix, not demonstrated skill improvement.',
              '2. On training data only, audit rainfall/PET aggregation, gauge-versus-delineated area and water-loss/runoff-generation assumptions before any physical calibration proposal. Routing-rate changes alone do not remove surplus water over complete recession.',
              '3. Specify one opt-in physical-volume-aware neural objective or training-only bias method, retain the log-loss control, and freeze validation-only selection/limits before a new separately approved matched budget. No unsupported loss flags, global test-based multiplier, or automatic extension.',
              '', *report['limits'], '', 'Original paper and frozen research outputs are unchanged. No new research fitting or push.']
    with (root/'diagnosis.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    record=dict(completed_utc=datetime.now(timezone.utc).isoformat(),diagnosis_sha256=sha(root/'diagnosis.json'),
                report_sha256=sha(root/'diagnosis.md'),source_script_sha256=sha(__file__),checkpoint_capacity_diagnosis_sha256=sha(root/'checkpoint_ceiling_diagnosis.json'),existing_neural_checkpoints_diagnosed=len(report['neural']),
                fitting_performed=False,original_outputs_modified=False,new_recipe_skill_claim=False)
    with (root/'diagnosis_closeout.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    paper=root.parents[1]
    note=f"\n\n## Read-only error diagnosis finished — {record['completed_utc']}\nCodex-owned diagnostic, not Claude progress. All18existing neural checkpoints diagnosed on Train/Validation/Test without optimizer steps; original exported test predictions reverified first. Read `{root/'diagnosis.md'}` and diagnosis.json. Physical inputs exceed observed matched-hour discharge already3.627x onTrain; gain1 routes all max(P-PET,0), runoff_coeff is partition not loss, and dryPET cannot deplete storage. Exact ledger/unit conversion checks pass. Confirmed endpoint-versus-hourly-mean reference mismatch; averaging alone barely affects huge volume bias. Preserve v1 checkpoints: correctness fix requires versioned interval-mean recipe, never silent replacement. Neural log-objective/clipping/tail statistics are per-seed; test-based correction or new training is not authorized. Next tasks:versioned interval-mean synthetic fix, training-only physical input/loss/area audit, one prospective neural objective confirmation proposal. New training needs separate budget. Original paper/frozen outputs unchanged; no push. Diagnostic ownership released; persistent12hmonitor remainsACTIVE.\n"
    for name in ('CODEX_SUPERVISION_LOG.md','CLAUDE_SUPERVISOR_TASKS.md'):
        with (paper/name).open('a') as f:f.write(note)
    print('DIAGNOSIS COMPLETE '+str(root/'diagnosis.md'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','extension','root'):p.add_argument(k,type=Path)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    try:
        if a.wait:
            end=time.monotonic()+21600
            while not (a.root/'neural_export_verification.json').exists():
                if time.monotonic()>end:raise TimeoutError('Diagnostic export did not finish within6h')
                time.sleep(10)
        finish(a.source,a.extension,a.root)
    except BaseException as exc:
        path=a.root/'diagnosis_failure.json'
        if not path.exists():
            with path.open('x') as f:json.dump(dict(error=str(exc),no_training_retry=True),f,indent=2);f.write('\n')
        raise
