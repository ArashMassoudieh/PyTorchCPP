# Experimental HydroFusion FFN verification path

`ffn_hydrofusion` is a synthetic-only, opt-in one-hour forecast prototype.
It is not a new result for Sligo Creek, an implementation for all five models,
or a replacement for the existing FFN/physical-residual hybrid approaches.

## Contract

- Explicit `hydrofusion_synthetic_verification: true`; package/CSV source flags
  are rejected before reading either source. The fixture is exactly 300 hourly
  rows defined in `hydrofusion_synthetic_fixture.h`; generic synthetic-profile
  knobs do not change it.
- Explicit integer `train_end_x`, `validation_end_x`, `test_end_x`, below the
  development seal; origin and target must be in the same partition.
- Twenty causal features in the saved order/units. Inputs are raw physical
  values, with runoff in mm/h. No future discharge or forcing is a predictor.
- `hydrofusion_target_mode` is `direct_log_q` or `log_residual`. Both use the
  same data builder, positive training-target median `q_ref`, standardization,
  FFN architecture, ordinary MSE, and validation checkpoint-selection path.
- The initial prototype supports Adam with `standardize` normalization only.
  `random_seed` is applied before model construction, like existing wrappers.
- Input/target scalers fit Train only. `train()` receives explicit validation
  tensors; assigning its Test slot does not itself enable checkpoint selection.
  Histories/final loss use standardized transformed-target MSE; the separate
  validation/test discharge metrics use raw physical runoff.
- Prediction `x` is target time; the separate map retains origin time. The
  stored mode, `q_ref`, horizon, feature schema, bounds, model hash and scaler
  hash must agree before reconstruction. Metadata and time map are registered inside the managed
  artifact manifest; modified metadata fails integrity checking. Internal output is mm/h; any public
  m3/s display must explicitly multiply by 19.0893/3.6.

Use `HydroFusionInferenceSession` for physical discharge. The generic session
rejects this approach by default instead of returning log/residual targets as
though they were discharge. Only explicit internal transformed-target callers
may opt into `HydroInferenceOutputSpace::TransformedHydroFusionTarget`.
This adapter accepts already-built physical features; a native-observation
package loader and operational availability/latency validation remain future work.

## Reproducible synthetic checks

The saved CLI fixture directory is
`Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/`.
Its runners refuse existing output directories and retain executable hashes.
The standalone metadata, log-target and feature-builder tests need no model fit.

Initialize the existing Utilities submodule first (`git submodule update --init Utilities`).
LibTorch and Qt/Armadillo are required, as for HydroBatch. Build into a NEW
path from an isolated, hash-verified source copy:

```
python3 Hydro/experiments/gistohq_sligo/build_hydrofusion_frozen.py HydroBatch /tmp/new-HydroBatch
bash Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/run_hydrofusion_task4_cli_checks.sh /tmp/new-HydroBatch
python3 Hydro/experiments/gistohq_sligo/link_hydrofusion_check_from_batch_build.py /tmp/new-HydroBatch /tmp/new-reload-check
```

The last helper uses that completed build's objects and frozen headers. It also
supports `--check integration`, `--check legacy_inference`, and `--check exporter`.
Never regenerate wrappers in a source tree another compiler is reading.
Exporter/loader caller-size guards reject a known class of stale aggregate-layout
mismatch; these checks do not replace clean, consistent compilation or prove all
possible ABI/ODR defects absent.

The preserved historical crashing ELF had a caller using model-map offset 1088
and a callee using offset 1120. Timestamp differences were not the diagnosis.
The recorded disassembly and a reproduced invalid free establish inconsistent
layouts in that ELF; the exact live-edit/stale-build event was not independently
reconstructed. New builds use copied consistent sources, normal optimization,
explicit layout-rejection tests, and source/binary manifests.

No real-basin training is authorized by these tests or by successful reloads.
Accuracy, event skill, data-quality assumptions and all-five-method improvements
require separate implementation and a prospectively approved matched experiment.

## Additive five-method family

The original FFN path above remains compatible. A separate matched FFN/LSTM,
causal physical baseline and two physical-log-residual hybrids are now available
through explicit synthetic-only CLI approaches. See
`Hydro/experiments/gistohq_sligo/hydrofusion_five_method_checks/README.md` and
`ALL_FIVE_ACCEPTANCE_20261118.md` for exact contracts and verification.
These additions do not remove source guards, enable GUI training or establish
real-basin improvements. The research proposal is saved and not launched.
