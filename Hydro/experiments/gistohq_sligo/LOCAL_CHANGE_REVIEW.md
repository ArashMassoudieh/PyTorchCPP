# Local research-file inventory and remaining review

Historical scripts and manifests are preserved in their original paths.
`HISTORICAL_SCRIPT_INVENTORY.json` records the exact 47 files and SHA256 values.
Preservation is not a claim that their scientific outputs or current execution
paths have been revalidated. The 42 Python sources were parsed and compiled
without execution; five JSON files (one template and four manifests) were parsed.

## Historical source groups

- `analyze_*`, `rescore_*`, `test_analyze_failure_atlas.py`: annual, event,
  matched-support and area-sensitivity analysis from earlier stages.
- `audit_*`: forcing coverage, MRMS catalog/export and native-discharge audits.
- `build_*`: versioned package/forcing variants.
- `generate_*`, `run_*`, `rank_regional_donors.py`, `check_intensity_consumer.py`:
  earlier screen generation, calibration, execution and donor inventory.
- `decade_20261022/**/run_manifest.json` and
  `improvement_loop_20261023/**/*manifest.json`: original batch provenance.

Some historical scripts perform I/O, start subprocesses or overwrite fixed
output paths at module scope. Do not import them to discover functionality,
run test discovery over non-test sources, or rerun completed batches. Inspect
the source and create a separately reviewed, guarded adapter if reuse is needed.
Full-period utilities are historical source only: they are not authorized to
read reserved observations. Absolute local paths and external sibling-package
dependencies have intentionally been preserved, not silently rewritten.

## Original mixed-source review completed

All 18 originally modified source/test files have now been reviewed in scoped
groups (including the earlier launcher fixes). The final seven model/batch
files were built in an isolated HEAD archive with the pinned Utilities
submodule, then exercised through fabricated-package model tests and six
actual CLI runs. No real watershed training/evaluation was rerun.

Final checks cover exact train/validation/test endpoints across missing hours,
beyond-test labels, fixed-state routing independent of target discharge,
optional MRMS-CV feature transport with stable existing support, rejection of
incompatible latent/CV layout, nonzero high-flow/volume loss activation,
matched daily/control support, and daily checkpoint reload.

Three issues exposed during this review were fixed: a circular residual-header
include; export relabeling that was not restricted to the actual multiscale
profile; and a missing optional volume term in the daily-history arm while its
matched control used that term. Volume/high-flow weights and spatial-CV
features remain disabled by default; verification is not a scientific
recommendation to enable the previously rejected experimental recipes.

Synthetic compatibility against the frozen source-build executable confirms
exact predictions for 944 plain-LSTM and 840 daily-history rows when optional
features are off and legacy ratio splits are used. Explicit cutoffs now apply
to plain LSTM as requested; the older executable ignored those endpoints, so
that intentional correction must not be described as identical old behavior.

Saved harness: `Hydro/tests/run_legacy_engine_regression_test.sh`,
`legacy_engine_regression_test.cpp` and root `verify_legacy_engine.pro`.
It uses new output directories and never opens real-basin observations.
Evidence/source/submodule/binary hashes are preserved under the Dropbox paper
`audits/legacy_engine_finalization_20261117/` directory. Historical outputs,
checkpoints and their provenance limitations remain unchanged.

This verifies the tested source and synthetic paths, not every possible
option combination, interactive GUI operation or new real-data improvement.
Claude's newer HydroFusion files are a separate active workstream.

## Generated build files

`/build-verify-stageb/` is local qmake output and verification runtime evidence.
It is ignored, not deleted. Source project `verify_stageb_checkpoint_reproduction.pro`
remains tracked. Logs/runner and a build-file hash inventory were additionally
preserved under the Dropbox paper `audits/repo_housekeeping_*` directory before
ignoring the build tree. Do not use the ignore rule as authorization to delete
original verification evidence or checkpoints.

## Config transport review completed

The loader/config types/exporter and exporter regression test were reviewed
as a separate settings-preservation change. Their exact four-file snapshot
passed `run_experiment_exporter_test.sh` in an isolated HEAD checkout with
no other outstanding source patches. Exported configs now retain chronological
cutoffs and additional routing/training settings instead of requiring sidecar
recovery for newly exported runs. Historical outputs remain untouched.

This verifies settings transport, not performance or behavioral acceptance of
optional high-flow/volume/MRMS features; their model-side changes remain a
separate review. Existing partial-source-build provenance for old experiments
is not retroactively rewritten by this source commit.

## Routing helper and cutoff regression review completed

The optional fixed-initial-flow argument in `two_reservoir_routing.h` and
its routing/scientific-core regression tests passed in an isolated HEAD
checkout containing only those three changes. Explicit initial flow is
independent of observed discharge, resets at segment boundaries, and
omitting the argument preserves legacy initialization. The LibTorch routing
test also covers analytic routing, gradients and checkpoint compatibility.
This does not yet accept the separate model-wrapper call-site changes.

## Regional progress logging review completed

The logging-only change in `lstm_regional_pretrain.cpp` adds basin/epoch
progress and elapsed time without changing training expressions or checkpoint
selection. It passed an isolated source compilation with format warnings
treated as errors; existing validation rejects invalid batch sizes before
the new progress calculation. No regional research job was rerun.

The subsequent final engine review above closes the seven model-side files
with isolated model/CLI tests and a frozen-executable compatibility check.
