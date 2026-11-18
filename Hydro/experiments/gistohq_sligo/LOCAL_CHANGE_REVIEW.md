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

## Remaining mixed source changes

The following pre-existing groups remain visible and uncommitted pending
independent scoped review; do not hide them with ignore rules:

| Group | Files | Review needed |
|---|---|---|
| Batch dispatch/export labeling | `hydro_batch_runner.cpp` | Separate multiscale export-label support from optional summary columns; verify CLI export/reload integration |
| Model splits and routing | `ffn_wrapper.cpp`, `pinn_wrapper.cpp`, `residual_pinn_wrapper.h` | Exact chronological cutoffs, segment reset/initial-flow behavior and regression coverage |
| Experimental learning/forcing | `lstm_pinn_wrapper.cpp`, `lstmnetworkwrapper.cpp`, `hydro_tensor_builder.h` | Separate urban/volume/high-flow/MRMS-CV changes; verify zero-option behavior and dependencies |

Several files contain multiple groups' changes; this table is a review map,
not a proposed blanket staging list. In particular, historical successful
runs do not establish that these uncommitted C++ patches are in a clean
checkout or reproducible from HEAD alone. Source/build provenance remains
relevant to any artifact release.

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

Seven model-side files remain pending as listed in the table above. Their
changes must be reviewed/tested by behavior, especially exact split support,
multiscale export labels, fixed-initial-flow call sites, and zero-weight
compatibility for the optional losses/features, before committing them.
