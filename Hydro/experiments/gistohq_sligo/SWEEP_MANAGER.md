# Unified HydroPINN Sweep Manager

The HydroPINN GUI exposes **Batch > Sweep Manager...** as the central sweep interface for all five approaches:

1. FFN
2. FFN + PINN
3. LSTM
4. LSTM + PINN
5. PINN

## Method-aware grids

The manager does not create a blind Cartesian product. It applies parameters only where they are meaningful:

- **FFN**: architecture, activation, FFN lag specification, learning rate, batch size, seed.
- **FFN + PINN**: architecture, activation, learning rate, batch size, seed, physics weight, latent recession `k`.
- **LSTM**: architecture, sequence length, learning rate, batch size, seed.
- **LSTM + PINN**: architecture, sequence length, learning rate, batch size, seed, physics weight, latent recession `k`.
- **PINN**: architecture, learning rate, batch size, seed, latent recession `k`.

Standalone PINN fixes `data_weight=0` and `physics_weight=1`. A sweep over a constant multiplicative physics-loss weight would not change a physics-only optimum, so it is intentionally not multiplied into the PINN run count.

## Presets

### Five-method baseline

Generates one configuration for each of the five approaches using the current baseline architecture and physics settings.

### Supervised architecture/memory

Selects FFN and LSTM and fills the established architecture, activation, and sequence grids. The controls remain editable before generation.

### Physics Stage 1

Selects FFN + PINN, LSTM + PINN, and PINN. The default physics grid is:

- hybrid physics weight: `0.001,0.005,0.01,0.025,0.05,0.1`
- latent recession `k [1/h]`: `0.01,0.02,0.04,0.08,0.16`

With one architecture, learning rate, batch size, and seed this produces:

- FFN + PINN: 30 runs
- LSTM + PINN: 30 runs
- PINN: 5 runs
- total: 65 runs

## Generate Only

Creates:

- `unified_sweep.batch`
- `generated_unified/*.json`
- `generated_unified/unified_manifest.csv`

The generated JSON/CSV artifacts are ignored by git.

## Generate & Run

The GUI generates the sweep, asks for an output parent directory, creates a timestamped `unified_sweep_YYYYMMDD_HHMMSS` directory, starts `HydroBatch`, and displays a modeless live log with Stop/Close controls.

The final comparison table is written to:

`<timestamped-output>/batch_summary.csv`

## Latent-storage recession parameter

For backward compatibility with the existing experiment JSON schema, the unified generator serializes the conceptual latent-reservoir recession coefficient in `storage_coeff`. HydroBatch maps that value into `latent_storage_recession_per_hour` for physics modes before training. Therefore different `k` values in the Sweep Manager change the actual latent-storage trajectory used by the physics residual.

Physics-informed GIStoOHQ runs continue to use physical-unit residuals (`normalization=none`) and the latent storage state is constructed from precipitation and PET only; observed runoff is not used to build the storage state.
