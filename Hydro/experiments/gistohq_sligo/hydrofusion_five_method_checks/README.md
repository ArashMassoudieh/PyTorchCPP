# Five-method causal forecast synthetic acceptance

This is a new opt-in family, not a change to the five existing same-hour models.

| Method | CLI approach | Target/reference |
|---|---|---|
| FFN matched forecast | ffn_causal_forecast | direct log Q / log persistence residual |
| LSTM matched forecast | lstm_hydrofusion | same two modes |
| Physical two-reservoir forecast | pinn_causal_forecast | fixed deterministic physics, no NN |
| FFN physical hybrid | ffn_physical_forecast | log physical-reference residual |
| LSTM physical hybrid | lstm_physical_forecast | same physical-reference residual |

All new neural modes use identical 24-origin eligible Train/Validation/Test
support and raw twenty-feature history. FFN consumes its final vector; LSTM
consumes the full sequence. Scalers fit the same eligible Train input tensors;
q_ref uses positive eligible Train target observations once each. Output is raw
mm/h, target time origin+1. Direct versus residual changes the representation,
not predictors, architecture, budget or checkpoint policy.

The older ffn_hydrofusion path remains compatible and is also checked. Its
broader training eligibility is not a matched architecture-control arm.

## Frozen causal physical recipe

Only explicit physics_profile=causal_two_reservoir_v1, pinn_initial_flow=0,
physics_dt=1, hydrofusion_target_mode=log_residual is accepted for the three
physical modes. storage_coeff is the fast rate k_f (1/h), lambda_decay the slow
rate k_s (1/h), runoff_coeff the fast fraction [0,1], forcing_gain a nonnegative
inflow multiplier. These meanings are local to this named candidate and must
not be retroactively assigned to old profiles. State is storage in mm, forcing
max(P-PET,0) mm/h. Exact constant-input integration advances state from previous
origin forcing; one-step forecast holds current origin-known forcing. True gaps
or invalid forcing reset to zero, never an observed discharge. No calibration
or state assimilation occurs. Native discharge history is used only to enforce
the common benchmark support for the physical baseline, not to set its state.

Physical hybrid correction is z(Qhat)=z(Q_phys)+f(x), with fixed multiplier1,
training-only q_ref, finite inverse and final nonnegative clipping. Zero f
recovers Q_phys. This differs from persistence-reference correction; the method
label, recipe checkpoint and exported config preserve that distinction. Neither
hybrid guarantees water-budget closure. The physical-only checkpoint contains
only a recipe tensor and constant q_ref=1 transport marker; no fake neural
network, fitted scale or training history. Its scalers explicitly use none.
Mass ledger uses exact integrated outflow; event plots/metrics integrate hourly
sampled Q by a one-hour rectangle rule and are not that exact state ledger.

## Production entry point and saved artifacts

HydroBatch dispatches every new approach. New format hydrofusion-forecast-v1
stores a versioned recipe tensor, q_ref and (neural modes only) trained weights.
The common managed bundle covers config, predictions, history, model/scalers,
metadata and origin-target map with hashes. On reload recipe/mode/shape/physical
parameters/q_ref must match; model and scaler binding is independently checked.
The dedicated HydroFusionForecastInference::predictAt accepts native hourly
context only through origin and builds predictors without reading a future
label. Generic inference deliberately rejects these approaches. GUI training
controls are not added, and UI acceptance is not claimed.

All modes reject package/CSV training and require explicit synthetic verification.
This implementation is NOT a real-data loader, real-basin skill result or new
research-training authorization. Do not bypass the guards to launch a campaign.

## Reproduce from a fresh output path

Initialize Utilities, then build from a frozen hashed source copy:

```
python3 Hydro/experiments/gistohq_sligo/build_hydrofusion_frozen.py HydroBatch /tmp/new-five-batch
python3 Hydro/experiments/gistohq_sligo/link_hydrofusion_check_from_batch_build.py /tmp/new-five-batch /tmp/new-five-reload --check five_forecast
python3 Hydro/experiments/gistohq_sligo/link_hydrofusion_check_from_batch_build.py /tmp/new-five-batch /tmp/new-ffn-reload --check reload
python3 Hydro/experiments/gistohq_sligo/hydrofusion_five_method_checks/run_checks.py /tmp/new-five-batch /tmp/new-five-reload /tmp/new-ffn-reload /tmp/new-five-results --baseline-binary /path/to/verified-earlier-HydroBatch
```

The runner hashes executable inputs, saves commands/stdout, fits only tiny
fabricated configs, checks all fresh bundles, rejects semantic corruption and
both real-source flags through actual CLI entry points, and evaluates declared
common support with free persistence. Old-five regression compares raw prediction
CSVs against the provided prior executable; the two binary hashes are retained.
Fresh extended parity is explicitly atol=1e-5+rtol1e-5*abs(expected), not a
bit-exact claim. Max errors are printed. Future forcing/discharge perturbation,
target-absent inference, missing-native history and gaps are checked independently.

Synthetic Q90 is computed from frozen sequence-eligible Train targets95..150;
common Test support245..280 is fixed by cutoffs/history, not skill. The single
fabricated window is a test of event accounting, not an identified real storm.
Evaluator per-year, signed/absolute bias, tail and paired-seed routines support
later approved analysis; one synthetic seed cannot establish seed stability.
No synthetic NSE ranking should be promoted to the paper.

The correction f in the hybrid equation means the inverse-scaled log residual,
not the raw standardized network output. Zero raw network weights need not
produce zero f when the target scaler has a nonzero offset. The zero-correction
unit check is applied in the explicitly reconstructed residual space. Peak-time
ties in the evaluator use the earliest maximum. No epoch/model selection uses
its reported Test metrics.
