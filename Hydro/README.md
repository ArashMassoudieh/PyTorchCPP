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
semantics.

## Watershed-oriented PINN inputs

HydroPINN now includes two hydrology-specific synthetic generators for PINN and
water-balance experiments:

| Synthetic profile | Columns / signals | PINN use |
| --- | --- | --- |
| `rainfall_runoff` | time, rainfall, evapotranspiration, temperature, soil storage, runoff target | Small event-scale rainfall-runoff baseline for mass-balance checks. |
| `watershed_balance` | time, effective precipitation, evapotranspiration, temperature, soil storage, groundwater storage, impervious fraction, runoff target | Broader watershed scenario with storm pulses, snowmelt contribution, infiltration, soil storage, groundwater recharge/baseflow, quick runoff, and impervious-area runoff. |

For `water_balance` PINN training, HydroPINN uses the leading watershed columns
`[time, precipitation/effective precipitation, evapotranspiration, temperature,
soil_storage, ...]` and applies a residual of the form `P - ET - Q - dS/dt`.
Extra watershed columns remain available to the supervised model as explanatory
features while the residual keeps a direct mass-balance interpretation.

## GUI workflow

1. **Data tab**
   - Start with `rainfall_runoff` or `watershed_balance` for watershed PINN smoke tests.
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
4. **Training tab**
   - Set epochs, batch size, learning rate, train/test split, and PINN weights.
   - Choose the PINN physics profile that matches the experiment.
   - Use **Train All** to compare all approaches under one configuration.
5. **Prediction, Performance Assessment, Plot, and Logs tabs**
   - Replot stored predictions, inspect metrics, compare target/predicted curves,
     analyze residuals, and review run logs.
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

## Suggested next development milestones

- Replace placeholder GA controls with a full GA configuration dialog that shares
  more of NeuroForge's hyperparameter-search behavior.
- Add named hydrology datasets and metadata validation for required forcing,
  discharge, storage, and timestamp columns.
- Persist HydroPINN experiment configurations so runs can be reloaded exactly.
- Add export actions for metrics, residuals, and predictions across all five
  approaches.
- Expand calibrated watershed-process residuals for snow accumulation/melt, infiltration capacity, groundwater exchange, channel routing, and evapotranspiration stress as field assumptions become available.
