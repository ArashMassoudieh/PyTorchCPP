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

The daily branch's own inputs are standardized by a fitted scaler (`fitDailyScaler`/`scaleDailyInputs`, report 71), fit on training-eligible rows only and persisted as ordinary registered buffers inside this model's own checkpoint format — never `scalers.csv`, which stays untouched for every other approach.

## Matched-control arm (report 71-72)

`trainSharedScaleAblationMatchedControl` is a plain `HydroLSTM` (no daily branch) restricted to the *identical* eligibility mask the paired multiscale run computes for the same `multiscale_daily_history_days`, with its own independently-fit (eligible-training-only) scalers — opt-in via `HydroRunConfig::multiscale_matched_control` (default `false`). Checkpoint format/approach stay `torch-module-v1`/`lstm_pinn`, so the public `HydroInferenceRunner::runSharedScaleAblation` dispatcher (and every exporter/artifact-loader whitelist) needed no new format entry — it only needed a flag check routing matched-control configs away from the multiscale loader (report 72 fixed a real bug here: the dispatcher originally routed every `multiscale_daily_history_days > 0` config to the multiscale loader regardless of this flag).

## D-selection must compare on common validation support (2026-11-04)

Eligibility is monotonic in `D`: a window eligible for a larger `D` is automatically eligible for every smaller `D`, never the reverse. Scoring each `D` candidate on its *own* eligible validation set is therefore not a fair comparison — the largest `D`'s own set systematically excludes early-record windows the smaller candidates still face, including whichever of those are hard examples, so a longer history can look better purely by never being tested on them.

`Hydro/experiments/gistohq_sligo/select_multiscale_history_length.py` fixes this: it restricts every candidate's validation NSE to the intersection of all candidates' eligible windows for that fold (provably exactly the largest `D`'s own eligible set, verified from the actual exported `predictions.csv` rather than assumed — it raises if monotonicity does not hold), records each candidate's coverage/retained-window counts, and breaks ties toward the smaller (simpler) `D`. `test_select_multiscale_history_length.py` demonstrates both the failure mode (naive own-support comparison inflating a longer `D`'s apparent score) and the fix, including a case where the fix genuinely flips the winner, not just flattens an artifact. This governs phase 1 selection scoring only; it is not yet exercised against real job output because phase 1 has not run.

## Tests

`Hydro/tests/multiscale_daily_history_test.cpp` (`build_multiscale_daily_history_test.sh`) — 16/16 passing: alignment arithmetic (including a mid-day-start case proving no partial-day credit), segment-boundary exclusion (including cross-segment), exact daily-aggregation values, tensor shapes, gradient flow through both branches, checkpoint save/load round-trip including the daily scaler (bit-exact), an **old plain-`HydroLSTM` checkpoint loading unchanged**, bit-exact zero-option reproduction, the single-eligibility-mask design (no fallback forward), four input-validation gaps (empty input, duplicate/unordered/nonfinite hours), and a structural proof that no day block ever reaches the real `kSealStartX=77016` development-cutoff boundary.

A full synthetic-package smoke test (`Hydro/experiments/gistohq_sligo/synthetic_multiscale_smoke_test.cpp`, `build_synthetic_multiscale_smoke_test.sh`) exercises training → export → reload for both arms through the real public `HydroInferenceRunner` dispatcher against a real on-disk synthetic GIStoOHQ package (not the lower-level functions directly) — confirms identical eligible-window sets between arms and bit-level checkpoint-reload prediction parity.

## Status

Training dispatch, checkpoint export/import whitelists, the public inference dispatcher, the matched-control arm, the daily-feature scaler, and the synthetic end-to-end smoke test are all implemented and verified (reports 68-73; commit `22494c0`, confirmed pushed). **Not yet run**: any of the 27 real-data jobs (9 selection + 18 comparison) the manifests in `multiscale_memory_screen_manifests/` describe — every `multiscale_daily_history_days` field in the comparison phase is still an enforced-invalid sentinel pending phase 1, and phase 1 itself requires separate, explicit budget approval.
