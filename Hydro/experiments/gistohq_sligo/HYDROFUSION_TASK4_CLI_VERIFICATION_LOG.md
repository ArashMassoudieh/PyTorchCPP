> Historical Claude verification record, preserved as written. The independent
> Codex takeover found and fixed additional seed, validation-selection, mode-export
> and integer-boundary issues. See `TASK4_CLOSURE_20261118.md` for current
> acceptance/evidence; this record alone does not close the earlier crash gate.

# HydroFusion Task 4 -- real-CLI dispatch verification log

Saved, repeatable record of the exact commands and results used to verify
the `ffn_hydrofusion` dispatch path through the real `HydroBatch` binary.
Not a commit -- this file is itself uncommitted, like the rest of Task 4's
work.

The job configs/batch files and the two orchestration scripts
(`run_hydrofusion_task4_cli_checks.sh`,
`compare_regression_against_baseline_build.sh`) live alongside this file
in `Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/` --
deliberately **not** in a `.scratch_*/` directory, since `/.scratch_*/` is
excluded by `.gitignore` and would never survive a clean checkout. (An
earlier version of this log pointed at `.scratch_hydrofusion_smoke/`,
which is gitignored; the checks below were re-run from this tracked
location to confirm the move didn't change anything.)

Build used for the "after" runs below: `build-hydrobatch-dev/HydroBatch`,
commit `e4c5409` (2026-11-17) + this session's uncommitted working-tree
changes, binary sha256 `77bcf69d43fa346a2326059ae1c3fa1f836ccff6e148d3706dc078ba97f191ed`.

Source provenance for "after": base commit `e4c5409e93497f226ad6eeb87fa0614d1bbeeb36`;
`git diff HEAD` over the 7 modified files
(`Hydro/batch/hydro_batch_runner.cpp`, `Hydro/evaluation/artifact_loader.cpp`,
`Hydro/evaluation/experiment_exporter.cpp`, `Hydro/evaluation/experiment_loader.cpp`,
`Hydro/evaluation/inference_runner.cpp`, `Hydro/models/hydro_run_types.h`,
`HydroBatch.pro`) sha256 `41546c38a17329cf1f179a96a1d8b86094d2f76225aa07e8a3de403e27b5d156`;
new untracked source files feeding this build, sha256 each:
`ffn_hydrofusion_wrapper.h`=`98fa4587998764a51c9b24701a2dc8173354f5b2bfb440e16ea630017523f489`,
`ffn_hydrofusion_wrapper.cpp`=`2f50c3aecc642d3448a9dfe8219db9d8ae7a94604114210f8ca9e6477aa4ef09`,
`hydrofusion_dataset_assembly.h`=`d5132c76b57b578bc7597bfc0878db6bbe3d4168ed29e5a04f6968549afcebb1`,
`hydrofusion_synthetic_fixture.h`=`6eef65f3263b09df5141e943db82533b35fc3a15e4bd39c2ac99b1f71b913b7c`,
`hydrofusion_metadata_sidecar.h`=`9cdc19eb79474d2d298269aecc401f7c8d1e54caf5f844aa4ef8c1d4a114e29a`.

Source provenance for "baseline" (section 5 only): git commit
`e4c5409e93497f226ad6eeb87fa0614d1bbeeb36`, checked out clean into an
isolated `git worktree` (no working-tree modifications; `Utilities`
submodule pinned to `a1cd112419f7da5ea28401d6bfc113488f463a27` via
`git submodule update --init`).

**Provenance gap, not yet closed**: the baseline *binary*'s own sha256
was not captured before its build directory was removed. Source and
submodule hashes identify what was *fed to the compiler*; they do not
identify what the compiler actually *produced* -- only the binary's own
hash does that, and that is what would be needed to prove, after the
fact, that the exact file executed for the baseline run is the one this
provenance record describes. The `__DATE__`/`__TIME__` embedding means
binary hash is not useful for detecting whether two *separate* builds of
identical source match each other, but it is still the only thing that
identifies *this specific build's executable* as distinct from, say, a
build that silently picked up stale object files or the wrong worktree.
Not rebuilding solely to backfill this hash now (per instruction); going
forward, both the source/submodule hashes and the binary's own sha256
will be captured immediately after each build, before any cleanup.

## 1. Config/sidecar cutoff-desync fix

`FFNHydroFusionWrapper::train()` previously defaulted `train_end_x`/
`validation_end_x`/`test_end_x` to `{150,220,280}` internally when the job
config left them unset, but could not write that choice back into the
`HydroRunConfig` object `hydro_batch_runner.cpp` later passes to
`exportRun()` (the wrapper only receives `const HydroRunConfig&`). Result:
`experiment_config.json` showed `-1/-1/-1` while the HydroFusion metadata
sidecar correctly showed `150/220/280` -- a config/sidecar desync.

Fix: the wrapper now requires all three cutoffs to be set explicitly in
the job config (throws otherwise), so the exported config and the sidecar
read from the same source of truth by construction. See
`Hydro/models/ffn_hydrofusion_wrapper.cpp`.

Verified (re-run from the tracked location via the saved script):
```
Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/run_hydrofusion_task4_cli_checks.sh \
  build-hydrobatch-dev/HydroBatch /tmp/hydrofusion_task4_cli_checks_out
# -> PASS: log_residual (expected success, got success)
grep -E "train_end_x|validation_end_x|test_end_x" \
  /tmp/hydrofusion_task4_cli_checks_out/log_residual/hydrofusion_ffn_batch_smoke/experiment_config.json
# -> test_end_x: 280, train_end_x: 150, validation_end_x: 220
grep -E "train_end_x|validation_end_x|test_end_x" \
  /tmp/hydrofusion_task4_cli_checks_out/log_residual/hydrofusion_ffn_batch_smoke.hydrofusion_metadata.txt
# -> train_end_x=150 validation_end_x=220 test_end_x=280   (now agrees)
```

## 2. Both HydroFusion target modes, through the real CLI

| job (in `run_hydrofusion_task4_cli_checks.sh`) | hydrofusion_target_mode | result |
|---|---|---|
| `log_residual` (`jobs.batch`) | `log_residual` | PASS -- `success=yes`, both sidecars written |
| `direct_log_q` (`jobs_direct_log_q.batch`) | `direct_log_q` | PASS -- `success=yes`, both sidecars written |

## 3. Both real-data rejection guards, through the real CLI

| job | trigger | result |
|---|---|---|
| `guard_hydro_package` (`jobs_guard_test.batch`) | `use_hydro_package=true`, bogus path | PASS -- job fails, exit 1, exact guard message |
| `guard_csv_data` (`jobs_guard_test_csv.batch`) | `use_csv_data=true`, bogus csv_path | PASS -- job fails, exit 1, exact guard message |

Both produce the exact guard message, checked verbatim by the script:
```
FFNHydroFusionWrapper: this mode is synthetic-fixture-only for now (HydroFusion Task 4
closure scope) -- it has no real-data loading path yet and must not silently substitute
synthetic data for a configured hydro_package_path/csv_path. Set use_hydro_package=false
and use_csv_data=false to run this mode's synthetic verification path explicitly.
```

All of 1-3 (plus the 5-method regression job) are run together by:
```
Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/run_hydrofusion_task4_cli_checks.sh \
  <path-to-HydroBatch-binary> [output-root]
```
Confirmed output (binary: `build-hydrobatch-dev/HydroBatch`, sha256 above):
```
PASS: log_residual (expected success, got success)
PASS: direct_log_q (expected success, got success)
PASS: guard_hydro_package (expected failure with guard message, got it)
PASS: guard_csv_data (expected failure with guard message, got it)
PASS: regression_5method (expected success, got success)

=== 5 passed, 0 failed ===
```

## 4. Fresh-process bundle-reload-check (bit-exact reproduction)

Built via `build_hydrofusion_batch_bundle_reload_check.sh`, run as a
genuinely separate OS process against the real `jobs.batch` bundle:
```
PASS: n=59 max_abs_diff=0 against the real HydroBatch-exported predictions.csv for approach 'ffn_hydrofusion'
```

## 5. Unchanged behavior for the 5 pre-existing methods (ffn/lstm/pinn/ffn_pinn/lstm_pinn)

Matching aggregate metrics in `batch_summary.csv` does not prove the
underlying predictions are unchanged -- aggregate statistics can agree
while individual rows differ. The actual check is a row-level diff of the
raw `predictions.csv` output, via the saved
`compare_regression_against_baseline_build.sh` script, between two
binaries:

- **"after"**: `build-hydrobatch-dev/HydroBatch` (this session's edits, sha256 above)
- **"baseline"**: a from-scratch build at the actual pre-session `HEAD`
  (`e4c5409` -- nothing this session has touched is committed, so `HEAD`
  already is the pre-HydroFusion-session code), built in an isolated
  `git worktree` at `/tmp/pytorchcpp_baseline_worktree` so the comparison
  never touched the working tree. Worktree removed after the comparison.

```
Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/compare_regression_against_baseline_build.sh \
  <after-binary> <baseline-binary> /tmp/hydrofusion_task4_regression_compare
```
Result:
```
PASS: ffn -- predictions.csv byte-identical across 220 rows
PASS: lstm -- predictions.csv byte-identical across 197 rows
PASS: pinn -- predictions.csv byte-identical across 220 rows
PASS: ffn_pinn -- predictions.csv byte-identical across 220 rows
PASS: lstm_pinn -- predictions.csv byte-identical across 197 rows

=== 5 passed, 0 failed ===
```
Every row of `predictions.csv` (not just the aggregate metrics derived
from it) is byte-identical between the two binaries for all 5 pre-existing
methods. This directly supports -- for these 5 methods, on this
deterministic synthetic job set only -- that the shared-file edits this
session made (`artifact_loader.cpp`, `inference_runner.cpp`,
`experiment_exporter.cpp`, `hydro_batch_runner.cpp`) did not change their
output. It does not by itself rule out behavior differences outside what
these specific jobs exercise.

(Metrics are poor by design -- 8-unit hidden layers, ~30 epochs, a tiny
synthetic fixture. This check establishes correctness/reproducibility of
dispatch, not predictive accuracy.)

## Status note: the earlier optimized-build crash

**Unresolved.** The production commit remains on hold while this is open.

The "torn compile from concurrent editing" explanation is a **hypothesis**,
not an established root cause. The evidence for it -- differing `.text`
section size and differing `nm --print-size` symbol sizes for
`HydroArtifactLoader::loadForInference` and `runIntegration` between the
crashing binary and later clean binaries, with both of those functions
defined in files that were being actively edited during that specific
background build -- is consistent with that hypothesis, but different
code sizes between two builds do not by themselves establish *why* the
code differs; they do not rule out a genuine, reproducible defect that
happens to be sensitive to code layout, compiler flags, or timing in some
other way. No controlled experiment has been run that actually reproduces
a crash by editing a file mid-compile and rules out alternative
explanations. 20+ subsequent clean builds not crashing is absence of
recurrence under normal (non-concurrent-edit) conditions, not a test of
the hypothesis itself, and not a code fix.

Treated as open until either (a) a controlled reproduction attempt
confirms the concurrent-edit hypothesis (or rules it out in favor of
something else), or (b) the user accepts the current evidence as
sufficient. No further production commit until then, per standing
instruction.

## Everything above remains uncommitted.
No training was launched; no reserved-period data (x>=77016) was touched.
