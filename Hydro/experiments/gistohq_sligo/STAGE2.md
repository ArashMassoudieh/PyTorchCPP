# Sligo Creek Stage 2 tuning

Stage 2 tunes optimizer-scale hyperparameters only after the Stage-1 memory and architecture screen.

## Finalists carried forward

The current Stage-2 generator uses four supervised finalists:

| family | memory | hidden layers | activation metadata |
|---|---:|---|---|
| FFN | 6 h lags | `16,16` | `relu` |
| LSTM | 12 h | `32` | native LSTM nonlinearities |
| LSTM | 24 h | `24,24` | native LSTM nonlinearities |
| LSTM | 24 h | `32` | native LSTM nonlinearities |

The LSTM `activation` configuration is not treated as a Stage-2 hyperparameter because the current `HydroLSTM` implementation uses LibTorch's native LSTM nonlinearities internally.

## Default Stage-2 grid

The selected architecture, memory setting, chronological split, standardization, epochs, and seed are held fixed while varying:

```text
learning_rate = 0.001, 0.003, 0.005
batch_size    = 16, 32, 64
seed          = 42
epochs        = 150
```

This gives:

```text
4 finalists x 3 learning rates x 3 batch sizes = 36 runs
```

Generate the sweep from the GUI with:

```text
Batch > Sweep Presets > Stage 2 Learning Rate / Batch Size
```

or from the command line:

```bash
cd Hydro/experiments/gistohq_sligo
python3 generate_stage2_sweep.py
```

Generated files:

```text
hyperparameter_stage2.batch
generated_stage2/stage2_*.json
generated_stage2/stage2_manifest.csv
```

Run it with HydroBatch:

```bash
./build-hydrobatch/HydroBatch \
  Hydro/experiments/gistohq_sligo/hyperparameter_stage2.batch \
  Hydro/experiments/gistohq_sligo/batch_outputs/hyperparameter_stage2
```

## Selection rule

Rank Stage-2 candidates primarily by validation MSE and then check hydrologic behavior using KGE, PBIAS, high-flow RMSE, low-flow RMSE, peak timing, and peak magnitude error. The already-inspected test period should not be used as the sole tuning objective.

After Stage 2, retain the strongest FFN and strongest one or two LSTM configurations for a multi-seed stability check. A publication-quality final selection should then use a fresh holdout or blocked/rolling temporal validation rather than continuing to optimize against the same test period.
