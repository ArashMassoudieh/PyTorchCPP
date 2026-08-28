# GIStoOHQ Sligo Creek tuning suite

This directory records the first verified real-data HydroPINN runs using the GIStoOHQ `HydroPINNExport` schema 1.2 handoff for Sligo Creek and provides loadable experiment configurations for supervised tuning.

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

The standardized supervised GUI batch sweep completed successfully with 9/9 jobs. The FFN results were:

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
| 18 h | 0.00073325923 | 0.027078760 | 0.017639857 | 0.45357971 | 0.30935615 | 42.590887% |
| 24 h | 0.00077293130 | 0.027801642 | 0.014191798 | 0.42560192 | 0.65068227 | 13.499405% |
| 48 h | 0.00100671620 | 0.031728792 | 0.016946389 | 0.26013532 | 0.46901425 | 40.709851% |

These are tuning results on the current chronological split, not final cross-validated scientific results. In the current implementation R2 is the standard `1-SSE/SST` coefficient of determination and is therefore numerically identical to NSE on the same held-out series; both labels are retained for ML and hydrology reporting.

## Memory sweep design

The supplied fixed configurations hold architecture, optimizer, data split, and seed fixed while varying one major memory setting at a time. All use a chronological 80/10/10 train/validation/test partition and `standardize` normalization.

LSTM sequence lengths:

```text
6, 12, 18, 24, 48 h
```

FFN forcing histories:

```text
6, 12, 24, 48 h
```

The memory sweep indicates that FFN 6 h is the only competitive dense-lag configuration, while LSTM 12 h and 24 h are the useful recurrent finalists. The 18 h run did not improve either endpoint. Stage-1 architecture tuning therefore intentionally focuses on FFN 6 h and LSTM 12/24 h rather than expanding the memory grid further.

## Stage-1 architecture and activation sweep

`generate_hyperparameter_sweep.py` creates a controlled architecture/activation sweep from the verified baseline JSON files. Generated JSON files and the generated manifest live under `generated_sweep/` and are ignored by Git; the generated batch file is `hyperparameter_stage1.batch`.

Default Stage 1 keeps `learning_rate=0.003`, `batch_size=32`, `seed=42`, `epochs=150`, chronological splitting, and standardization fixed while testing:

- FFN lag-6: hidden architectures `16`, `24`, `32`, `48`, `16,16`, `24,24`, `32,16`, `32,32`, `48,24`; activations `tanh`, `relu` (18 runs).
- LSTM 12 h and 24 h: hidden architectures `16`, `24`, `32`, `48`, `24,24`, `32,32`; activations `tanh`, `relu` (24 runs).
- Total default Stage-1 experiments: 42.

Generate the sweep:

```bash
cd Hydro/experiments/gistohq_sligo
python3 generate_hyperparameter_sweep.py
```

To include sigmoid as a diagnostic activation as well:

```bash
python3 generate_hyperparameter_sweep.py --include-sigmoid
```

The generator also accepts `--epochs`, `--learning-rate`, `--batch-size`, and `--seed`, but the default Stage-1 comparison should keep those fixed. Learning rate, batch size, and seed tuning should be performed only after narrowing the architecture/activation candidates.

Run the generated sweep from the GUI with **Run Config Batch...** and select `hyperparameter_stage1.batch`, or from the command line:

```bash
./build-hydrobatch/HydroBatch \
  Hydro/experiments/gistohq_sligo/hyperparameter_stage1.batch \
  Hydro/experiments/gistohq_sligo/batch_outputs/hyperparameter_stage1
```

## Batch summaries

`HydroBatch` records the configuration alongside the metrics. The summary now includes:

```text
experiment_id
mode
lstm_sequence_length
input_lags
hidden_layers
activation
learning_rate
batch_size
random_seed
normalization
success
final_loss
validation_mse
test_mse
rmse
mae
r2
nse
kge
correlation
pbias
volume_error_percent
peak_timing_error
peak_magnitude_error_percent
high_flow_rmse
low_flow_rmse
```

CSV fields containing commas, notably `hidden_layers` and `input_lags`, are quoted correctly. If an existing `batch_summary.csv` uses an older schema, HydroBatch archives it as `batch_summary.pre_hyperparams*.csv` and starts a clean summary before the new run.

Rerunning an experiment does not delete the previous export. Existing experiment folders are moved to `.previous`, `.previous.2`, and so on before the new result is exported.

## GUI run

From the PyTorchCPP repository root, launch HydroPINN, choose **Load Experiment Config...**, load one JSON file from this directory, then run the intended plain supervised approach (FFN or LSTM). Export the completed experiment after each run so predictions, split labels, scalers, losses, checkpoints, and hydrologic metrics are retained.

### Inputs + Output plot

The Plot tab exposes **Inputs + Output** for all current data sources rather than synthetic data only.

- Synthetic plots the generated model inputs plus generated target.
- CSV plots the configured model input columns plus the configured output/target column.
- GIStoOHQ HydroPINNExport plots precipitation, temperature, relative humidity, wind speed, solar radiation, PET, and observed runoff on the common physical-time axis. Input series use the left axis and the output uses a separate right axis so runoff remains visible despite differing forcing magnitudes.

Generic non-GIStoOHQ Hydro package plotting remains intentionally gated until its feature-name contract is explicit.

### GUI batch run

The GUI exposes **Run Config Batch...** in a Batch toolbar and Batch menu. It accepts the same `.batch` files as the command-line runner, asks for an output directory, launches `HydroBatch`, and streams output into a modeless progress dialog. The main HydroPINN window remains responsive while the external batch process is active.

Build `HydroBatch` after batch-runner changes:

```bash
mkdir -p build-hydrobatch
cd build-hydrobatch
qmake ../HydroBatch.pro CONFIG+=PowerEdge
make -j4
```

The current GIStoOHQ field package intentionally supports only `ffn` and `lstm` in these batch files. PINN modes remain excluded until a separately versioned rainfall-runoff physics contract exists.

## Model-selection rule

Use validation metrics for selecting architecture, activation, learning rate, batch size, and seed settings. Report held-out test metrics only after choosing the configuration. Do not choose a configuration from test NSE/R2 or PBIAS alone.

For hydrologic comparison, prioritize NSE/R2, KGE, PBIAS, peak timing/magnitude error, high-flow RMSE, low-flow RMSE, and hydrograph/flow-duration plots in addition to MSE.

After Stage 1, retain only the strongest few configurations for Stage 2 learning-rate/batch-size tuning. Then run the final candidates across multiple seeds and, for publication-quality selection, blocked or rolling temporal validation.

## PINN scope

The current GIStoOHQ field package contains observed discharge and meteorological/PET forcings but no observed storage. `FFN + PINN`, `LSTM + PINN`, and standalone `PINN` therefore remain intentionally disabled for this handoff. Do not synthesize or zero-fill storage. A separate versioned rainfall-runoff physics profile must be designed before enabling those approaches.
