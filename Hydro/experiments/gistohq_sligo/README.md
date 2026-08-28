# GIStoOHQ Sligo Creek tuning suite

This directory records the first verified real-data HydroPINN runs using the GIStoOHQ `HydroPINNExport` schema 1.2 handoff for Sligo Creek and provides loadable experiment configurations for the supervised tuning sweep.

## Verified handoff

Dataset root (relative to a PyTorchCPP checkout located beside GIStoOHQ):

```text
../GIStoOHQ/examples/SligoCreek/outputs/sligocreekdemo_data/hydropinn
```

The adapter accepts the GIStoOHQ native temporal assets, harmonizes them to hourly rows, converts observed USGS discharge to runoff depth using the producer-supplied catchment area, and uses the rainfall-runoff profile.

Verified integration baselines:

| approach | normalization | memory | test MSE | RMSE | MAE | R2 / NSE | PBIAS |
|---|---|---:|---:|---:|---:|---:|---:|
| FFN | none | 1-step basic input | 0.0023148554 | 0.048112944 | 0.023749184 | -0.74230916 | -109.21215% |
| LSTM | none | 6 h | 0.0010477584 | 0.032369097 | 0.020267447 | 0.21346491 | 25.261933% |

The standardized supervised GUI batch sweep completed successfully with 8/8 jobs. The FFN results were:

| FFN history | test MSE | RMSE | MAE | R2 / NSE | KGE | PBIAS |
|---:|---:|---:|---:|---:|---:|---:|
| 6 h | 0.000973655652 | 0.031203456 | 0.021779988 | 0.26794018 | 0.46281356 | -3.27149% |
| 12 h | 0.00110382663 | 0.033223887 | 0.026622682 | 0.17094644 | 0.00330646 | 82.57351% |
| 24 h | 0.00144605443 | 0.038027022 | 0.025719309 | -0.08494933 | -0.15056458 | 48.54938% |
| 48 h | 0.00118982065 | 0.034493777 | 0.026462316 | 0.10916488 | -0.11062032 | 85.39312% |

The standardized LSTM results were:

| sequence | test MSE | RMSE | MAE | R2 / NSE | KGE | PBIAS |
|---:|---:|---:|---:|---:|---:|---:|
| 6 h | 0.00071896305 | 0.026813486 | 0.017926307 | 0.46028621 | 0.44712800 | 35.821542% |
| 12 h | 0.00063226145 | 0.025144810 | 0.015492653 | 0.52706653 | 0.58526386 | 25.960245% |
| 18 h | pending | pending | pending | pending | pending | pending |
| 24 h | 0.00077293130 | 0.027801642 | 0.014191798 | 0.42560192 | 0.65068227 | 13.499405% |
| 48 h | 0.00100671620 | 0.031728792 | 0.016946389 | 0.26013532 | 0.46901425 | 40.709851% |

These are tuning results on the current chronological split, not final cross-validated scientific results. In the current implementation R2 is the standard `1-SSE/SST` coefficient of determination and is therefore numerically identical to NSE on the same held-out series; both labels are retained for ML and hydrology reporting.

## Sweep design

The supplied configurations hold architecture, optimizer, data split, and seed fixed while varying one major memory setting at a time. All use a chronological 80/10/10 train/validation/test partition and `standardize` normalization.

LSTM sequence lengths:

```text
6, 12, 18, 24, 48 h
```

FFN forcing histories:

```text
6, 12, 24, 48 h
```

For the FFN configurations a single lag group applies the listed lags to every forcing feature. The LSTM keeps sequence memory internally and does not use the FFN lag builder. The 18 h LSTM configuration was added to resolve the current 12--24 h performance interval without extending toward longer memory.

## GUI run

From the PyTorchCPP repository root, launch HydroPINN, choose **Load Experiment Config...**, load one JSON file from this directory, then run the intended plain supervised approach (FFN or LSTM). Export the completed experiment after each run so predictions, split labels, scalers, losses, checkpoints, and hydrologic metrics are retained.

### Inputs + Output plot

The Plot tab now exposes **Inputs + Output** for all current data sources rather than synthetic data only.

- Synthetic plots the generated model inputs plus generated target.
- CSV plots the configured model input columns plus the configured output/target column.
- GIStoOHQ HydroPINNExport plots precipitation, temperature, relative humidity, wind speed, solar radiation, PET, and observed runoff on the common physical-time axis. Input series use the left axis and the output uses a separate right axis so runoff remains visible despite differing forcing magnitudes.

Generic non-GIStoOHQ Hydro package plotting remains intentionally gated until its feature-name contract is explicit.

### GUI batch run

The GUI exposes **Run Config Batch...** in a Batch toolbar and Batch menu. It accepts the same `.batch` files as the command-line runner, asks for an output directory, launches `HydroBatch`, and streams output into a modeless progress dialog. The main HydroPINN window remains responsive while the external batch process is active.

Build `HydroBatch` at least once before using the GUI batch action:

```bash
mkdir -p build-hydrobatch
cd build-hydrobatch
qmake ../HydroBatch.pro CONFIG+=PowerEdge
make -j4
```

The LSTM-only batch now contains 5 jobs; the full supervised batch contains 9 jobs.

```bash
./HydroBatch \
  Hydro/experiments/gistohq_sligo/lstm_sweep.batch \
  Hydro/experiments/gistohq_sligo/batch_outputs/lstm
```

```bash
./HydroBatch \
  Hydro/experiments/gistohq_sligo/supervised_sweep.batch \
  Hydro/experiments/gistohq_sligo/batch_outputs/supervised
```

Batch console output and `batch_summary.csv` now include `r2`. If an existing summary uses the older schema without R2, HydroBatch archives it as `batch_summary.pre_r2*.csv` and starts a clean R2-aware summary instead of appending misaligned rows.

The current GIStoOHQ field package intentionally supports only `ffn` and `lstm` in these batch files. PINN modes remain excluded until a separately versioned rainfall-runoff physics contract exists.

## Model-selection rule

Use validation metrics for selecting memory/normalization settings. Report held-out test metrics only after choosing the configuration. Do not choose a configuration from test NSE/R2 or PBIAS.

For hydrologic comparison, prioritize NSE/R2, KGE, PBIAS, peak timing/magnitude error, high-flow RMSE, low-flow RMSE, and hydrograph/flow-duration plots in addition to MSE.

## PINN scope

The current GIStoOHQ field package contains observed discharge and meteorological/PET forcings but no observed storage. `FFN + PINN`, `LSTM + PINN`, and standalone `PINN` therefore remain intentionally disabled for this handoff. Do not synthesize or zero-fill storage. A separate versioned rainfall-runoff physics profile must be designed before enabling those approaches.
