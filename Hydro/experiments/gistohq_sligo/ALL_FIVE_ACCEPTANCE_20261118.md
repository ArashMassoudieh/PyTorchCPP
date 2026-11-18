# Five-method causal forecast implementation acceptance

Development implementation and synthetic verification are complete for FFN,
LSTM, fixed two-reservoir physics, FFN physical hybrid and LSTM physical hybrid.
This does not establish improved real-basin skill, add GUI training controls or
permit a real-data training campaign. All original same-hour methods remain.

## Implemented approaches

- ffn_causal_forecast: matched 24-origin eligibility, final-vector FFN; direct
  log-Q and log persistence residual.
- lstm_hydrofusion: identical eligible support/scales/predictors/target choices,
  recurrent sequence processing.
- pinn_causal_forecast: causal constant-input two-reservoir simulation with
  explicit fixed parameters and zero state at true gaps. No neural network,
  target assimilation, calibrated scale or fake training history.
- ffn_physical_forecast and lstm_physical_forecast: shared log physical-reference
  correction, finite inverse and final clipping. Neither guarantees conservation.

All are explicitly opted in, synthetic-only, actual HydroBatch dispatch modes.
Dedicated managed-bundle inference checks recipe/mode/shape/parameters/q_ref,
model/scaler binding, chronology and native history. It predicts from context
through origin without a target row. Generic inference rejects these approaches
rather than exposing log/correction targets as discharge. Existing ffn_hydrofusion
remains compatible, with a broader training support distinguished from the new
matched FFN candidate. Model format hydrofusion-forecast-v1 is versioned.

## Verified evidence

Final frozen source build: /tmp/hydrofusion_all_five_batch_final.build/
Acceptance runner: /tmp/hydrofusion_all_five_acceptance/
The final supervision entry records the durable Dropbox audit path. Source,
executable, fixture and command hashes are retained. Compilation used normal
-O2, not an inlining/optimization workaround. Vendor/legacy warnings remain.

Fresh-process parity compares each managed export with prediction from raw
hourly context, using atol=1e-5 + rtol1e-5*abs(expected). These are tolerance
passes, NOT bit-exact neural parity:

| Approach/target | Rows | Maximum absolute difference, mm/h |
|---|---:|---:|
| FFN matched/direct |36|1.1634133168e-7|
| FFN matched/persistence residual |36|1.3509365626e-8|
| LSTM/direct |36|1.1162985247e-7|
| LSTM/persistence residual |36|1.3949789546e-8|
| Two-reservoir physical |36|0|
| FFN physical residual |36|1.3550354017e-7|
| LSTM physical residual |36|1.0350702406e-7|

The existing FFN HydroFusion path also reproduces59rows with max difference0.
All seven new bundles pass exact time/observation/split/native-support-map checks,
target-absent inference, future forcing/discharge perturbation invariance, and
gap/missing-native rejection.35semantic corruption cases (five per bundle)
reject altered q_ref, recipe configuration, scalers, time maps and non-finite
predictions even after refreshing the corresponding manifest entry.

Actual CLI rejects20cases covering both package/CSV source flags, missing
explicit consent and unsupported extra objectives, across all five new methods.
Four neural methods reproduce identical CSVs with repeated seed42 and respond
to seed43. Training losses change and each saved checkpoint is the validation
minimum; FFN persistence selectedepoch3/12 and LSTM physical hybridepoch11/12,
exercising restoration before the last epoch. Physical-only history is empty.
These are reproducibility checks, not a seed-stability research analysis.

Original five methods, compared against the preserved independently hashed
prior executable /tmp/codex_hydrofusion_task4_batch_v4, retain BYTE-IDENTICAL raw
prediction CSVs: FFN220,LSTM197,physical220,FFN hybrid220,LSTM hybrid197;
1054rows total. This is bounded to these exact saved synthetic configurations.
The new comparison saves both executable hashes before launching either one.

Standalone native/sequence/development-file/physical suites pass ASan/UBSan;
LeakSanitizer is disabled because of the host restriction. The original25feature
and19log-target tests plus metadata strictness tests pass under these sanitizers.
Eight evaluator tests, seven runner-safety and four cutoff-loader tests pass.
No claim is made that every legacy test or LibTorch itself is sanitizer-clean.

## Shared evaluator and next research gate

Saved evaluator requires declared common native support, finite identities/values,
consistent observed values, Train-only Q90 and complete frozen event windows.
It reports NSE/KGE/RMSE/MAE, signed/absolute PBIAS, high-flow partition, event
volume/peak magnitude/timing, per-year and paired-seed deltas; undefined scores
remain null. The saved runner exercises all methods plus free persistence on
36frozen synthetic timestamps. Synthetic scores are not manuscript findings.

FIVE_METHOD_RESEARCH_PROPOSAL.md freezes a smallest six-fit first proposal,
resource ceilings and advancement criteria; later architectures/hybrids remain
separately approved stages. Nothing in that proposal has been launched. Real
producer qualifier extraction, independently accepted presealed input manifests
and explicit human research budget approval remain mandatory before real data.
No reserved observations, manuscript/original/frozen artifacts or pushes were
used or modified. The existing accepted StageB/multiscale campaign was not rerun.
