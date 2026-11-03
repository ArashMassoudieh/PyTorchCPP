# Multiscale daily-history branch — design (implemented, not yet trained)

Source of truth: `improvement_loop_20261023` reports 62 (memory audit), 66-68 (design, budget, finalization) in the Dropbox paper-results tree. This file is the terse, code-adjacent version for anyone reading the repo without that tree.

## What this is

`Hydro/models/multiscale_lstm_module.h` adds an opt-in daily-resolution antecedent-history branch to the existing 24-hour `HydroLSTM` (`hydro_lstm_module.h`). A small daily LSTM (`DailyHistoryBranch`) processes `D` fully-completed prior calendar days of coarse forcing (daily total precipitation, daily total PET); its final `(h, c)` is transferred through two learned linear layers into the hourly LSTM's initial state, replacing today's implicit zero-initialization. Gauch et al. (2021, HESS)'s multi-branch MTS-LSTM variant.

**Why**: report 62 traced the production inference path (`shared_scale_inference.cpp`'s `makeSequences`) and confirmed every overlapping 24h window is independently zero-initialized — no memory beyond 24 hours regardless of segment length. This branch is the lowest-risk way to give the model longer antecedent information without re-architecting the recurrent cell itself (that direction is Task 4/PIDuaL, deliberately separate).

## Opt-in, not a replacement

- `HydroRunConfig::multiscale_daily_history_days` (default `0`) and `multiscale_daily_hidden_size` (default `4`), in `hydro_run_types.h`. `0` means fully disabled.
- `HydroLSTMImpl::forward(inputs)` (no args) is byte-for-byte unchanged. A new overload, `forward(inputs, h0, c0)`, is additive and only reached when a caller explicitly passes an initial state.
- Disabled mode constructs a plain `HydroLSTM` directly — never `MultiscaleHydroLSTM` — so an old checkpoint loads into the exact same class with the exact same archive keys it always had. `MultiscaleHydroLSTM` registers `hourly`/`daily` submodules and is a genuinely new checkpoint format, but only for genuinely new (enabled) training runs.

## Parameter counts, from the actual built modules (not estimated)

```
control (24h only):     5,153 params
multiscale (D-day, hidden=4): 5,601 params  (+448, +8.7%)
```

**This compares the multiscale architecture as a whole against the matched 24h control. Without a capacity-matched third arm — deliberately not included in the initial screen, see report 67 §1 — this comparison cannot isolate "benefit from antecedent memory" from "benefit from 448 extra parameters." Any effect found should be described as a property of the multiscale architecture, not attributed to memory specifically, until a capacity-matched confirmation stage (if the initial screen passes) disentangles them.**

## Data handling — leakage rule

A window starting at hour `windowStart` may only see fully-completed calendar days strictly before it (`Hydro/dataset/daily_history_builder.h`, `completedDayBlocks`). A day block must also lie entirely within the window's own segment (`hasSufficientSegmentHistory`) — windows without `D` full prior days in-segment are **excluded**, never padded or zero-filled. Both arms in the comparison are scored on the identical eligible-window set (report 67 §4) — the control's numbers in this comparison will not match the original frozen baseline's full-window numbers, by design.

## Tests

`Hydro/tests/multiscale_daily_history_test.cpp` (`build_multiscale_daily_history_test.sh`) — 10/10 passing: alignment arithmetic (including a mid-day-start case proving no partial-day credit), segment-boundary exclusion (including cross-segment), exact daily-aggregation values, tensor shapes, gradient flow through both branches, checkpoint save/load round-trip (new format), an **old plain-`HydroLSTM` checkpoint loading unchanged**, bit-exact zero-option reproduction, the ineligible-window fallback path (proven distinct from feeding zeros through the trained branch), and a structural proof that no day block ever reaches the real `kSealStartX=77016` development-cutoff boundary.

## Not yet done

Training-loop integration (`Hydro/models/lstmnetworkwrapper.cpp` constructing `MultiscaleHydroLSTM` and calling `daily_history::buildDailyHistoryTensors` when `multiscale_daily_history_days > 0`) is **not implemented** — moot while training is barred, but a real prerequisite before any manifested job can actually run. The manifests (`Hydro/experiments/gistohq_sligo/generate_multiscale_memory_screen.py`) use approach name `lstm_multiscale`, which does not yet exist in the training wrapper's approach dispatch.
