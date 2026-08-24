# HydroPINN

HydroPINN is the hydrology-focused Qt + LibTorch application in this repository.
It is intended to move the project forward from generic neural-network training
toward repeatable rainfall-runoff and water-balance experiments that compare
purely data-driven models against physics-informed models.

## Current scope

HydroPINN currently supports five local approaches from the same GUI and run
configuration:

| Approach | Purpose | Typical first use |
| --- | --- | --- |
| FFN | Fast supervised feed-forward baseline. | Check whether the data split and normalization are sensible. |
| FFN + PINN | Feed-forward model with a hydrology/ODE residual penalty. | Test whether physics regularization improves the FFN baseline. |
| LSTM | Supervised sequence model with recurrent memory. | Model temporal persistence without manual lag expansion. |
| LSTM + PINN | Recurrent model plus physics residual penalty. | Compare sequence memory and physics regularization together. |
| PINN | Physics-first standalone baseline. | Inspect what the configured residual explains without supervised data loss. |

See [`APPROACHES.md`](APPROACHES.md) for equations, loss definitions, and model
semantics. See [`DATASET_CONTRACT.md`](DATASET_CONTRACT.md) for the versioned
input package that GIStoOHQ and other acquisition tools should produce, and
[`GISTOOHQ_PIPELINE.md`](GISTOOHQ_PIPELINE.md) for the generic acquisition/cache
architecture and thin HydroPINN export adapter.

## Watershed-oriented PINN inputs

HydroPINN now includes two hydrology-specific synthetic generators for PINN and
water-balance experiments:

| Synthetic profile | Columns / signals | PINN use |
| --- | --- | --- |
| `watershed_balance` | time, effective precipitation, evapotranspiration, temperature, soil storage, groundwater storage, total storage, impervious fraction, runoff target | Primary watershed scenario with storm pulses, snowmelt contribution, infiltration, soil storage, groundwater recharge/baseflow, quick runoff, and impervious-area runoff. |
| `rainfall_runoff` | time, rainfall, evapotranspiration, temperature, soil storage, runoff target | Smaller event-scale rainfall-runoff baseline for mass-balance checks. |

For `water_balance` PINN training, HydroPINN uses the leading watershed columns
`[time, precipitation/effective precipitation, evapotranspiration, temperature,
total watershed storage, ...]` and applies a residual of the form `P - ET - Q - dS/dt`.
Extra watershed columns remain available to the supervised model as explanatory
features while the residual keeps a direct mass-balance interpretation.

## GUI workflow

1. **Data tab**
   - Start with `watershed_balance`, then use `rainfall_runoff` as the smaller event-scale comparison case.
   - Switch to CSV when running observed hydrology data.
   - Use zero-based x/y column controls for CSV files.
   - Export generated synthetic data when a comparison should be reproducible.
2. **Hydro Workflow tab**
   - Review the in-app recommended run order and forward path.
3. **Network Structure tab**
   - Configure hidden layers and activations.
   - Enable time-lagged FFN inputs for FFN-family approaches.
   - Keep lag settings disabled for LSTM-family approaches, where sequence memory
     is handled by the recurrent backend.
   - LSTM sequence length is an independent configuration value; running FFN lag
     optimization does not silently change the LSTM receptive window.
4. **Training tab**
   - Set epochs, batch size, learning rate, chronological train/validation/test split, and PINN weights.
   - Choose the PINN physics profile that matches the experiment.
   - Use **Train All** to compare all approaches under one configuration.
5. **Prediction, Performance Assessment, Plot, and Logs tabs**
   - Replot stored predictions, inspect metrics, compare target/predicted curves,
     analyze residuals, and review run logs.
   - Load an exported experiment directory in the Prediction tab to validate and
     retain its configuration, checkpoints, and scaler states for artifact-backed
     inference. Exported predictions are restored immediately for plotting and
     review. The programmatic inference runner can execute all five checkpoint
     families on compatible physical input tensors or LSTM sequences using the
     training-fitted input and target scalers. Checkpoints load directly from
     their verified in-memory bytes through a zero-copy stream view, avoiding
     temporary-file I/O and a second checkpoint-sized allocation on each run.
     Reuse `HydroInferenceSession` for repeated predictions so model construction,
     scaler import, and checkpoint deserialization occur only once.
     The GUI prepares one reusable session per checkpoint while loading an
     experiment, so corrupt or architecture-incompatible archives fail before use.
     For LSTM approaches, `predictSeries` builds all overlapping sequence windows
     with a tensor view operation instead of a per-window allocation loop.
     Select **Run loaded checkpoint on current Data source** to execute prepared
     GUI sessions on a Hydro Package or CSV selected in the Data tab without retraining. The
     package profile and forecast-feature semantics must match the exported run;
     only the package directory and catchment identity may change.
     FFN-family inference reconstructs the exported per-feature lag expansion and
     aligns observations and timestamps after the maximum lag automatically.
     Training and inference share the same CSV tensor builder, preventing parser
     or feature-order drift between checkpoint creation and later execution.
     FFN training and inference also share lag parsing, feature-column mapping,
     tensor expansion, and leading-row alignment.
   - Run `tests/run_inference_runner_test.sh` with `LIBTORCH_PATH` configured to
     verify export/reload/checkpoint round trips across all five approaches.
6. **GA tab**
   - Run lag-structure optimization for FFN and FFN + PINN workflows.

## Build

```bash
mkdir -p build-hydropinn
cd build-hydropinn
qmake ../HydroPINN.pro LIBTORCH_PATH=/path/to/libtorch TORCH_CXX11_ABI=1
make -j"$(nproc)"
./HydroPINN
```

The qmake project links Qt Widgets, Qt Charts, LibTorch, Armadillo, OpenMP, and
the shared NeuroForge utility/model sources needed by HydroPINN.

## Suggested results and plot tabs

For this watershed-first app, the next result views should emphasize hydrologic
interpretability as much as generic prediction error:

- **Hydrograph + hyetograph:** runoff target/prediction lines with rainfall or
  effective-precipitation bars.
- **Mass-balance residuals:** `P - ET - Q - dS/dt` through time, plus mean bias,
  RMSE, signed cumulative residual, and residual distribution.
- **Cumulative water balance:** cumulative precipitation, evapotranspiration,
  runoff, and storage change to expose drift.
- **Flow-duration / peak-flow diagnostics:** high-flow and low-flow behavior,
  peak timing error, peak magnitude error, and runoff-volume error.
- **Regime-conditioned metrics:** split results by wet/dry periods, soil-storage
  state, groundwater-storage state, and impervious quickflow dominance.
- **Experiment table export:** one row per approach with data loss, physics loss,
  NSE/KGE/RMSE/MAE/bias, peak timing error, and key configuration values.

## Suggested next development milestones

- Replace placeholder GA controls with a full GA configuration dialog that shares
  more of NeuroForge's hyperparameter-search behavior.
- Add GUI inference from compatibility-checked model artifacts.
- Expand calibrated watershed-process residuals for snow accumulation/melt, infiltration capacity, groundwater exchange, channel routing, and evapotranspiration stress as field assumptions become available.

The current synthetic workflow is still a software-validation stage. Artifact
inference now has a five-approach export/reload integration test; field-data
calibration and broader scientific validation remain necessary before paper
experiments.

Standalone physics-only PINN runs deliberately do not select checkpoints using
observed validation discharge, because that would introduce supervised model
selection into the nominally physics-only baseline. Their validation MSE
remains a post-training diagnostic; physics-based early stopping is a separate
future option.

FFN and LSTM supervised runs now fit normalization exclusively on the training
partition and inverse-transform predictions before validation/test metrics are
computed. PINN-capable runs reject normalization for now: enabling it safely
requires inverse-transforming predictions and physical forcing/state variables
inside the differentiable residual rather than applying conservation to scaled
quantities.

`DDRRLoader` now validates and loads canonical observation CSV exports into
per-catchment series, preserves UTC timestamps, calculates elapsed physical
hours independently for each catchment, and converts observed discharge to
`mm/h` using declared catchment areas.

The loader can also open a generic package directory through `manifest.json`,
resolve observation and catchment-attribute files safely, read `area_m2` by
stable catchment ID, enforce the declared schema/profile, and reject unresolved
package QC errors. Declared observation, catchment-attribute, and
variable-metadata SHA-256 digests are verified before parsing. Required
variable names and canonical units are enforced when the package declares
`variables_file`; full generic asset-catalog ingestion remains producer-side
work.

All four trainable wrapper families accept the same package/catchment
configuration and build named physical tensors through the shared package
loader. The Data tab exposes **Hydro Package** directory, catchment ID, and
profile controls.

Package-backed PINN runs infer their physical timestep from the elapsed-hour
column instead of trusting a manually entered `physics_dt`. Current training
backends reject irregular package intervals explicitly; this prevents silently
applying one finite-difference timestep to gapped or irregular observations.

The Performance tab's **Export Experiment...** action writes a deterministic
artifact directory containing
`experiment_config.json`, one-row-per-approach `metrics.csv`, and long-form
`predictions.csv`, and `training_history.csv`. Each prediction is labeled as
training, validation, or test. PINN-capable water-balance runs also export
`physics_residuals.csv` with the same partition labels. Validation-selected
model checkpoints are written beneath `models/` with checksums in `models.csv`.
Per-epoch validation history and fitted scaler state are
written to `training_history.csv` and `scalers.csv`. Every export includes
`environment.json`; package-backed exports also preserve the accepted source
manifest as `dataset_manifest.json` and record its SHA-256 release fingerprint
in `provenance.json`.
`HydroArtifactLoader` verifies model-manifest format, safe relative paths, file
sizes, checkpoint formats, and SHA-256 digests before returning checkpoint
bytes to a future inference session. It also reloads and validates the exported
input and target scaler states so inference can reproduce the transformations
that were fitted exclusively on the training partition. The inference artifact
entry point loads the experiment configuration, checkpoints, and scalers as one
bundle, then rejects missing counterparts, unknown approaches, and checkpoint
formats that do not match the selected model family.
`HydroExperimentLoader` reads the exported configuration back into a validated
`HydroRunConfig`, providing a programmatic rerun boundary without silently
falling back to current GUI defaults. The Performance tab can apply that
configuration back to the GUI through **Load Experiment Config...**.

## Scientific-safety rules

- Runs use chronological train, validation, and test partitions. The validation
  partition is reserved for model and physics-weight selection; test metrics
  must not drive retries or hyperparameter choices.
- LSTM + PINN evaluates one ordered full-training-sequence physics gradient per
  epoch rather than repeating it inside every shuffled supervised mini-batch.
- `PINNWrapper` is the explicit fifth approach and enforces physics-only training.
  For water balance with independently known precipitation, ET, and storage,
  interpret it as a diagnostic residual solver rather than a freely identifiable
  rainfall-runoff model.
- Field-data conservation must use compatible physical units and strictly
  increasing timestamps. `RRPhysics::waterBalanceResidualAtTimes` supports
  variable physical time steps; normalized plotting coordinates must not be
  substituted for physical elapsed time.
