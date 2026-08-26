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

## GIStoOHQ temporal handoff

The native-resolution GIStoOHQ temporal export requires hourly harmonization, unit conversion, and explicit validity masks before training. The versioned consumer decisions and the storage-profile compatibility blocker are documented in [`GISTOOHQ_HOURLY_HANDOFF.md`](GISTOOHQ_HOURLY_HANDOFF.md).

## GUI workflow

1. **Data tab**
   - Start with `watershed_balance`, then use `rainfall_runoff` as the smaller event-scale comparison case.
   - Switch to CSV when running observed hydrology data.
   - Use zero-based x/y column controls for CSV files. Malformed rows, missing
     configured columns, and invalid or out-of-range numbers are reported with
     their source line rather than being silently discarded.
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
     Recurrent replay also applies the trainer's minimum two-sample sequence
     window, preventing undersized saved settings from changing input shapes.
     The builder accepts quoted numeric fields and LF or CRLF records, and rejects
     malformed quoting, partial numeric values, and non-finite numeric values
     instead of silently rewriting or truncating them.
     FFN training and inference also share lag parsing, feature-column mapping,
     tensor expansion, and leading-row alignment. Invalid, duplicate, empty, or
     excess lag groups are rejected instead of being silently shifted or dropped.
     Artifact CSV manifests accept both Unix LF and Windows CRLF line endings.
     Imported scaler states are revalidated at the tensor boundary, including
     method, shape overflow, numeric finiteness, and non-zero scale checks.
     Scaler fitting likewise rejects non-floating or non-finite training tensors
     without replacing a previously fitted, valid scaler state.
     Metric evaluation requires aligned, non-empty, finite observation and
     prediction vectors rather than silently truncating mismatched series.
     Peak diagnostics additionally require aligned timestamps and split labels
     with at least one finite held-out test sample.
     Time-integrated physics residuals require aligned, finite, strictly
     increasing timestamps so cumulative drift cannot reverse or skip intervals.
     Successful metric rows must contain finite, non-negative MSE, RMSE, and MAE,
     and the exported RMSE must agree with the square root of MSE.
     `loadForInference` cross-checks the complete approach sets and recomputes
     prediction and residual summaries before accepting a bundle. The returned
     `HydroInferenceArtifacts` includes those merged, validated run results, so
     callers do not need to reopen and reconcile the summary CSV files.
     It also restores training histories and build-environment metadata. For
     package-backed runs, the copied dataset manifest is hashed again and must
     match the SHA-256 fingerprint recorded in `provenance.json`.
     All recomputable prediction, peak-flow, and residual metrics are compared
     with `metrics.csv`, not only the three primary error measures.
     Export now performs the same structural preflight before creating a run
     directory, so misaligned series, bad partitions, missing checkpoints,
     incompatible formats, and incomplete scalers cannot produce partial bundles.
     Experiment identifiers must be single filenames; absolute paths, parent
     traversal, and nested path components are rejected before any lock or
     staging directory is created.
     Configuration strings are emitted with complete JSON control-character
     escaping and decoded on reload, preserving paths and identifiers without
     producing malformed configuration documents. Reload also supports standard
     Unicode escapes and surrogate pairs while rejecting malformed or unpaired
     sequences. Numeric, integer, Boolean, and string values must terminate at a
     JSON member boundary; fractional integers, overflow, and token suffixes are
     rejected rather than partially parsed. The flat experiment schema also
     rejects duplicate fields instead of silently choosing one occurrence.
     Exported scientific summaries are recomputed from held-out predictions and
     residual samples, preventing stale in-memory metrics from being serialized.
     Files are written into a sibling staging directory and atomically renamed
     only after every stream closes successfully; existing run directories are
     never overwritten and failed exports clean up their staging data. A lock
     directory and uniquely reserved staging name prevent concurrent exporters
     from deleting or publishing over one another. Lock cleanup is exception-safe,
     including failures encountered while reserving a staging directory. On
     POSIX systems the lock records its owning process, allowing a later export
     to recover a lock left behind by a process that no longer exists while
     preserving locks owned by live or unverifiable processes.
     `artifact_manifest.csv` records the size and SHA-256 digest of every file in
     the completed bundle; inference reload rejects modified, missing, or
     unlisted files before parsing any artifact content. Its explicit schema
     version is checked before parsing, so incompatible future formats fail
     clearly instead of being interpreted as the current layout.
   - Run `tests/run_inference_runner_test.sh` with `LIBTORCH_PATH` configured to
     verify export/reload/checkpoint round trips across all five approaches.
     CI jobs can set `HYDRO_REQUIRE_LIBTORCH_TESTS=1` so a missing LibTorch
     installation fails instead of silently skipping the runtime tests.
   - Reloaded experiments restore exported physics-residual series as well as
     predictions, so residual and cumulative-drift plots remain available.
     Residual timestamps and partition labels must match prediction artifacts.
   - Predictive and peak diagnostics are recomputed from the restored held-out
     test rows rather than trusting potentially stale summary values.
     Exported MSE/RMSE/MAE must agree with those recomputed values before training
     and validation summaries are restored into the GUI.
6. **GA tab**
   - Run lag-structure optimization for FFN and FFN + PINN workflows.
   - Stop requests cancel between training trials without applying a partial
     candidate selection; an active backend epoch run completes first.

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

Current run summaries and `metrics.csv` include signed peak-timing error,
peak-magnitude error percentage, and RMSE for the highest and lowest observed
10% of flows on the held-out test partition. The Plot tab provides observed and
predicted flow-duration curves using exceedance probabilities for the same
held-out samples; regime-conditioned peak diagnostics remain a future view.
PINN-capable water-balance runs also summarize finite physical residuals with
mean bias, RMSE, and a timestep-integrated signed cumulative residual. The Plot
tab can compare cumulative residual drift across all stored PINN approaches.

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
