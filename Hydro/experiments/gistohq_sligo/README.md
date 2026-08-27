# GIStoOHQ Sligo Creek tuning suite

This directory records the first verified real-data HydroPINN runs using the GIStoOHQ `HydroPINNExport` schema 1.2 handoff for Sligo Creek and provides loadable experiment configurations for the next supervised tuning sweep.

## Verified handoff

Dataset root (relative to a PyTorchCPP checkout located beside GIStoOHQ):

```text
../GIStoOHQ/examples/SligoCreek/outputs/sligocreekdemo_data/hydropinn
```

The adapter accepts the GIStoOHQ native temporal assets, harmonizes them to hourly rows, converts observed USGS discharge to runoff depth using the producer-supplied catchment area, and uses the rainfall-runoff profile.

Verified baseline results from the first complete integration run:

| approach | normalization | memory | test MSE | RMSE | MAE | NSE | PBIAS |
|---|---|---:|---:|---:|---:|---:|---:|
| FFN | none | 1-step basic input | 0.0023148554 | 0.048112944 | 0.023749184 | -0.74230916 | -109.21215% |
| LSTM | none | 6 h | 0.0010477584 | 0.032369097 | 0.020267447 | 0.21346491 | 25.261933% |

These are integration baselines, not tuned scientific results.

## Sweep design

The supplied configurations intentionally hold architecture, optimizer, data split, and seed fixed while varying one major memory setting at a time. All use a chronological 80/10/10 train/validation/test partition and `standardize` normalization.

LSTM sequence lengths:

```text
6, 12, 24, 48 h
```

FFN forcing histories:

```text
6, 12, 24, 48 h
```

For the FFN configurations a single lag group applies the listed lags to every forcing feature. The LSTM keeps sequence memory internally and does not use the FFN lag builder.

## How to run

From the PyTorchCPP repository root, launch HydroPINN, choose **Load Experiment Config...**, load one JSON file from this directory, then run the intended plain supervised approach (FFN or LSTM). Export the completed experiment after each run so predictions, split labels, scalers, losses, checkpoints, and hydrologic metrics are retained.

If PyTorchCPP and GIStoOHQ are not sibling directories, edit `hydro_package_path` in the JSON files.

## Model-selection rule

Use validation metrics for selecting memory/normalization settings. Report held-out test metrics only after choosing the configuration. Do not choose a configuration from test NSE or PBIAS.

For hydrologic comparison, prioritize NSE/KGE, PBIAS, peak timing/magnitude error, high-flow RMSE, low-flow RMSE, and hydrograph/flow-duration plots in addition to MSE.

## PINN scope

The current GIStoOHQ field package contains observed discharge and meteorological/PET forcings but no observed storage. `FFN + PINN`, `LSTM + PINN`, and standalone `PINN` therefore remain intentionally disabled for this handoff. Do not synthesize or zero-fill storage. A separate versioned rainfall-runoff physics profile must be designed before enabling those approaches.
