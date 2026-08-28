# GIStoOHQ physics-informed methods

This note defines the storage-free field-data path used to enable the three physics-informed HydroPINN approaches for the Sligo Creek GIStoOHQ export:

- `ffn_pinn` — FFN + PINN
- `lstm_pinn` — LSTM + PINN
- `pinn` — standalone physics-only PINN

## Why observed storage is not required

The GIStoOHQ field package contains precipitation, PET and other meteorological forcings plus observed runoff, but it does not contain an observed catchment-storage time series. The field-data PINN path therefore must not pretend that any weather column is storage and must not derive storage from observed runoff.

For physics modes, `loadHydroPackageTensors()` enables a latent conceptual storage state on the longest contiguous hourly segment. The state is generated only from precipitation and PET using a linear-reservoir recession:

```text
I(t) = max(P(t), 0) - max(PET(t), 0)
S_latent(t+dt) = max(0, S_latent(t) + dt * (I(t) - k S_latent(t)))
```

with the current runtime default

```text
k = 0.08 h^-1
```

Observed runoff is never used to construct `S_latent`.

The physics-informed feature layout is

```text
[time, precipitation, PET, temperature, S_latent, RH, wind, solar]
```

while the plain supervised FFN/LSTM path remains unchanged and continues to use its verified six-forcing feature contract.

## Water-balance residual

The existing water-balance residual is then evaluated as

```text
r(t) = P(t) - PET(t) - Q_pred(t) - dS_latent/dt
```

and the physics loss is the mean squared residual. Because `S_latent` obeys the conceptual reservoir equation above, this is equivalent to encouraging predicted runoff to be consistent with reservoir release while still allowing FFN + PINN and LSTM + PINN to fit observed runoff through their supervised data term.

The standalone PINN sets the supervised data weight to zero and is therefore a physics-first diagnostic rather than a replacement for calibrated rainfall-runoff prediction.

## Contiguous-segment rule

Finite-difference storage residuals must not span missing hours. The GIStoOHQ adapter labels contiguous hourly blocks with `segment_id`; the current PINN implementation uses the longest contiguous block and verifies that its time step is one hour.

## Normalization

The current PINN residual is evaluated directly in physical rainfall/PET/runoff/storage units. The initial real-data physics configs therefore use

```text
normalization = none
```

until inverse-scaled residual plumbing is implemented.

## Generate and run the three-method baseline

From the GUI:

```text
Batch -> Sweep Presets -> GIStoOHQ Physics Methods
```

This runs `generate_physics_methods.py`, which creates

```text
gistohq_physics_methods.batch
```

with one FFN + PINN, one LSTM + PINN and one standalone PINN experiment. Then use

```text
Batch -> Run Config Batch...
```

and select that batch file.

From the command line:

```bash
cd Hydro/experiments/gistohq_sligo
python3 generate_physics_methods.py

../../../build-hydrobatch/HydroBatch \
  gistohq_physics_methods.batch \
  batch_outputs/physics_methods
```

## First comparison and next tuning step

The first run is a baseline check, not final calibration. Compare the three physics methods against the supervised FFN/LSTM finalists using test-independent validation metrics plus physics-residual RMSE. After verifying stable training, tune at least:

- physics-loss weight;
- conceptual recession coefficient `k`;
- FFN/LSTM learning rate and architecture inherited from the supervised tuning stages;
- LSTM sequence length if the physics-informed optimum shifts away from the supervised optimum.

The recession coefficient is a conceptual model parameter, not an observed-storage calibration. It should be selected from validation behavior and hydrologic plausibility, then checked under blocked/rolling temporal validation before publication use.
