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
| Batch/config/export | `hydro_batch_runner.cpp`, `experiment_loader.cpp`, `experiment_exporter.cpp`, `hydro_run_types.h`, `experiment_exporter_test.cpp` | Separate multiscale export-label and split-boundary provenance fixes from optional loss/feature fields; verify clean-checkout roundtrip |
| Model splits and routing | `ffn_wrapper.cpp`, `pinn_wrapper.cpp`, `residual_pinn_wrapper.h`, `two_reservoir_routing.h`, `reservoir_routing_test.cpp`, `scientific_core_test.cpp` | Exact chronological cutoffs, segment reset/initial-flow behavior and regression coverage |
| Experimental learning/forcing | `lstm_pinn_wrapper.cpp`, `lstmnetworkwrapper.cpp`, `lstm_regional_pretrain.cpp`, `hydro_tensor_builder.h` | Separate urban/volume/high-flow/MRMS-CV and pretraining logging changes; verify zero-option behavior and dependencies |

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
