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
- **FFN + PINN**: architecture, activation, learning rate, batch size, seed, physics weight, reduced-reservoir recession coefficient `k`.
- **LSTM**: architecture, sequence length, learning rate, batch size, seed.
- **LSTM + PINN**: architecture, sequence length, learning rate, batch size, seed, physics weight, reduced-reservoir recession coefficient `k`.
- **PINN**: architecture, learning rate, seed, reduced-reservoir recession coefficient `k`.

Standalone PINN fixes `data_weight=0` and `physics_weight=1`. Its corrected reduced-reservoir implementation enforces physics over the full collocation domain, so mini-batch size is not a meaningful standalone-PINN tuning parameter even though the legacy experiment schema still contains a batch-size field.

## Presets

### Five-method baseline

Generates one configuration for each of the five approaches using the current baseline architecture and physics settings.

### Supervised architecture/memory

Selects FFN and LSTM and fills the established architecture, activation, and sequence grids. For paper-grade adaptive runs, FFN memory is evaluated as cumulative lag candidates (`1`, `1,2`, ..., `1,2,3,4,5,6`) rather than one fixed six-lag specification.

### Physics Stage 1

Selects FFN + PINN, LSTM + PINN, and PINN. The default physics grid is:

- hybrid physics weight: `0.001,0.005,0.01,0.025,0.05,0.1`
- reduced-reservoir `k [1/h]`: `0.01,0.02,0.04,0.08,0.16`

With one hybrid architecture, learning rate, batch size, and seed this produces 30 FFN + PINN runs and 30 LSTM + PINN runs. Standalone PINN is swept over its architecture and `k` candidates by the adaptive paper pipeline.

## Generate Only

Creates:

- `unified_sweep.batch`
- `generated_unified/*.json`
- `generated_unified/unified_manifest.csv`

The generated JSON/CSV artifacts are ignored by git.

## Generate & Run

The GUI generates the sweep, asks for an output parent directory, creates a timestamped output directory, starts `HydroBatch`, and displays a modeless live log with Stop/Close controls.

The comparison table is written to:

`<timestamped-output>/batch_summary.csv`

## Reduced-reservoir coefficient and legacy field names

The current reduced-reservoir physics is

`dQ/dt = k (Peff - Q)`, with `Peff = max(P - PET, 0)`.

No latent storage trajectory is reconstructed or supplied to this residual. For backward compatibility with existing experiment JSON files, the candidate coefficient `k` is still serialized through historical fields such as `storage_coeff` and `latent_storage_recession_per_hour`, and `use_latent_storage_physics` remains the legacy flag selecting the contiguous forcing-only physics layout. These names do **not** mean that a latent storage state is generated.

For controlled Synthetic `reduced_reservoir` experiments, `synthetic_reservoir_truth_k` defines the fixed ground-truth coefficient independently of the candidate/model `k`. Physics sweeps therefore vary the model coefficient while every candidate sees exactly the same synthetic target hydrograph.

Physics-informed GIStoOHQ runs use physical-unit residuals (`normalization=none`) and the forcing-only layout `[time, Peff, P, PET, ...]`. Observed runoff is used as the supervised target for hybrid methods and as the single initial-condition anchor for standalone PINN; it is not used to construct a storage input.

## Adaptive paper pipeline

`run_adaptive_full_pipeline.py` replaces the old fixed descendant stages for publication runs:

1. tune FFN/LSTM architecture and memory;
2. inherit those winners into physics tuning;
3. tune optimizer settings separately for each method while preserving earlier winners;
4. freeze each method and evaluate five random seeds;
5. export validation-selected winners, robustness summaries, frozen configs, and controlled-synthetic whole-domain known-truth metrics.

`run_paper_experiments.sh` runs the controlled synthetic preflight, the adaptive controlled synthetic study, the adaptive Sligo Creek study, and then builds manuscript tables and figures in one timestamped paper-run directory.
