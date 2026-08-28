# Sligo Creek Stage 3 robustness sweep

Stage 3 is the robustness phase after architecture/memory tuning (Stage 1) and learning-rate/batch-size tuning (Stage 2).

## Purpose

Stage 3 freezes the selected Stage-2 hyperparameters and varies only the random seed. This prevents choosing a final model from one favorable initialization and provides a distribution of performance for each finalist.

## Default seeds

```text
42, 123, 2026, 31415, 27182
```

With one FFN winner and two LSTM winners, the default design contains 15 runs.

## GUI

Use:

```text
Batch -> Sweep Presets -> Stage 3 Multi-seed Robustness...
```

Enter the actual Stage-2 winners:

- FFN hidden layers, activation, learning rate, and batch size;
- LSTM finalist 1 sequence length, hidden layers, learning rate, and batch size;
- LSTM finalist 2 sequence length, hidden layers, learning rate, and batch size;
- seed list and epochs.

The GUI calls `generate_stage3_sweep.py` and writes:

```text
hyperparameter_stage3.batch
generated_stage3/stage3_*.json
generated_stage3/stage3_manifest.csv
```

Then run `hyperparameter_stage3.batch` with **Run Config Batch...**.

## Selection rule

Do not select the final model from the single best seed. Summarize each finalist across seeds using mean and standard deviation of validation loss and hydrologic metrics. Inspect hydrographs, peaks, high/low-flow behavior, and flow-duration behavior as well.

Because the current test partition has already been inspected during development, publication-quality final performance should use a new untouched temporal holdout or blocked/rolling temporal validation after the Stage-3 configuration is frozen.
