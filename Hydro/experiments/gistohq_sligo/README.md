# GIStoOHQ Sligo Creek tuning suite

This directory records the first verified real-data HydroPINN runs using the GIStoOHQ `HydroPINNExport` schema 1.2 handoff for Sligo Creek and provides loadable experiment configurations for the supervised tuning sweep.

## Verified handoff

Dataset root (relative to a PyTorchCPP checkout located beside GIStoOHQ):

```text
../GIStoOHQ/examples/SligoCreek/outputs/sligocreekdemo_data/hydropinn
```

The adapter accepts the GIStoOHQ native temporal assets, harmonizes them to hourly rows, converts observed USGS discharge to runoff depth using the producer-supplied catchment area, and uses the rainfall-runoff profile.

Verified integration baselines:

| approach | normalization | memory | test MSE | RMSE | MAE | NSE | PBIAS |
|---|---|---:|---:|---:|---:|---:|---:|
| FFN | none | 1-step basic input | 0.0023148554 | 0.048112944 | 0.023749184 | -0.74230916 | -109.21215% |
| LSTM | none | 6 h | 0.0010477584 | 0.032369097 | 0.020267447 | 0.21346491 | 25.261933% |

The standardized LSTM memory sweep is now also verified to propagate sequence length correctly:

| sequence | test MSE | RMSE | MAE | NSE | PBIAS |
|---:|---:|---:|---:|---:|---:|
| 6 h | 0.00071896305 | 0.026813486 | 0.017926307 | 0.46028621 | 35.821542% |
| 12 h | 0.00063226145 | 0.025144810 | 0.015492653 | 0.52706653 | 25.960245% |
| 24 h | 0.00077293130 | 0.027801642 | 0.014191798 | 0.42560192 | 13.499405% |
| 48 h | 0.00100671620 | 0.031728792 | 0.016946389 | 0.26013532 | 40.709851% |

These are tuning results on the current chronological split, not final cross-validated scientific results.

## Sweep design

The supplied configurations hold architecture, optimizer, data split, and seed fixed while varying one major memory setting at a time. All use a chronological 80/10/10 train/validation/test partition and `standardize` normalization.

LSTM sequence lengths:

```text
6, 12, 24, 48 h
```

FFN forcing histories:

```text
6, 12, 24, 48 h
```

For the FFN configurations a single lag group applies the listed lags to every forcing feature. The LSTM keeps sequence memory internally and does not use the FFN lag builder.

The experiment loader propagates `lstm_sequence_length` into subsequent GUI-created run configurations. After loading a sequence configuration, verify the run log reports the requested value, for example:

```text
LSTM sequence length=24
```

## GUI run

From the PyTorchCPP repository root, launch HydroPINN, choose **Load Experiment Config...**, load one JSON file from this directory, then run the intended plain supervised approach (FFN or LSTM). Export the completed experiment after each run so predictions, split labels, scalers, losses, checkpoints, and hydrologic metrics are retained.

If PyTorchCPP and GIStoOHQ are not sibling directories, edit `hydro_package_path` in the JSON files.

## Batch run

A headless batch runner is available for repeated configuration sweeps. Build it separately from the GUI:

```bash
qmake HydroBatch.pro CONFIG+=PowerEdge
make -j4
```

Run the LSTM-only sweep:

```bash
./HydroBatch \
  Hydro/experiments/gistohq_sligo/lstm_sweep.batch \
  Hydro/experiments/gistohq_sligo/batch_outputs/lstm
```

Run the complete supervised FFN + LSTM sweep:

```bash
./HydroBatch \
  Hydro/experiments/gistohq_sligo/supervised_sweep.batch \
  Hydro/experiments/gistohq_sligo/batch_outputs/supervised
```

Batch files contain one job per line:

```text
lstm lstm_standardize_seq12.json
ffn  ffn_standardize_lag6.json
```

Config paths are resolved relative to the `.batch` file. Each job is loaded through the same `HydroExperimentLoader`, executed through the same FFN/LSTM wrappers as the GUI, and exported under its `experiment_id`. The batch output directory also receives `batch_summary.csv`, giving one comparison row per completed job. A failed job is reported and the runner continues with the remaining jobs; the process returns nonzero if any job fails.

The current GIStoOHQ field package intentionally supports only `ffn` and `lstm` in these batch files. PINN modes remain excluded until a separately versioned rainfall-runoff physics contract exists.

## Model-selection rule

Use validation metrics for selecting memory/normalization settings. Report held-out test metrics only after choosing the configuration. Do not choose a configuration from test NSE or PBIAS.

For hydrologic comparison, prioritize NSE/KGE, PBIAS, peak timing/magnitude error, high-flow RMSE, low-flow RMSE, and hydrograph/flow-duration plots in addition to MSE.

## Plot-button applicability

The general comparison plots work with stored supervised FFN/LSTM results: target vs predicted, 1:1 scatter, approach subplots, residuals, absolute-error CDF, Taylor diagram, and flow-duration curves.

Two plot actions are intentionally context-specific:

- **Synthetic Inputs + Output** is available only when the Data source is `Synthetic`; the GUI disables it for CSV and Hydro Package inputs instead of presenting a button that silently has no applicable data.
- **Cumulative Physics Residual (PINN only)** requires a successful physics-informed run containing stored physics residuals. Plain FFN/LSTM results do not define that quantity.

## PINN scope

The current GIStoOHQ field package contains observed discharge and meteorological/PET forcings but no observed storage. `FFN + PINN`, `LSTM + PINN`, and standalone `PINN` therefore remain intentionally disabled for this handoff. Do not synthesize or zero-fill storage. A separate versioned rainfall-runoff physics profile must be designed before enabling those approaches.
