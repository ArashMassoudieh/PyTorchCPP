# Five-method implementation scope (development only)

This implements a separate one-hour causal forecasting experiment family. It
preserves the existing five same-hour methods and all saved checkpoints. Passing
synthetic checks is not evidence of improved basin skill.

| Method | New opt-in approach | Reference / task |
|---|---|---|
| FFN | ffn_causal_forecast (matched); ffn_hydrofusion remains accepted | direct log Q or log persistence residual |
| LSTM | lstm_hydrofusion | same targets; complete 24-origin feature sequences |
| Two-reservoir baseline | pinn_causal_forecast | fixed physical simulation; no neural fit |
| FFN physical hybrid | ffn_physical_forecast | log physical-reference residual |
| LSTM physical hybrid | lstm_physical_forecast | identical physical-reference residual |

Physical recipe: two positive linear reservoirs, fixed zero initial flow at
real gaps, one-hour exact constant-input updates, origin-known max(P-PET,0)
held for the forecast interval. State at origin uses only preceding forcing.
No target discharge assimilation, observed segment-start initialization,
future measured rainfall, calibration search or volume/intensity objective.
Raw storage is mm, inflow/outflow mm/h. Physical parameters are explicitly
supplied and pinned in exported configuration; this is a separately labeled
causal candidate, not a silent change to the original baseline.

Physical hybrid algebra: z(Qhat)=z(Q_phys_forecast)+f(features), with fixed
coefficient 1 and the existing training-only positive-target q_ref. This is
NOT persistence-residual correction. Zero correction recovers Q_phys. Both
hybrids share reconstruction. Neural correction does not conserve water.
The physical-only baseline has no fitted q_ref or scalers; any transport
placeholder in the common metadata is explicitly constant and not a transform.

Neural comparisons share 24-origin eligibility where sequences are required.
FFN physical hybrid uses the final 20-feature origin vector on that same support.
The new ffn_causal_forecast uses identical sequence-eligible training and evaluation
support as LSTM, consuming the last feature vector only. The already accepted
ffn_hydrofusion remains untouched and its broader training support is not an
architecture-isolating control.

Ordered implementation: physical router/ledger + unit checks; shared four-mode
wrapper and explicit new checkpoint formats; CLI/export/managed inference;
frozen build + actual synthetic CLI/fresh-process tests + old-method regression;
unified matched-support diagnostics + a priced NOT-LAUNCHED research proposal.
Real data, GUI controls, research training, manuscript result promotion and
all-five accuracy claims are outside this synthetic implementation acceptance.
