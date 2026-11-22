# Context and history recovery recipe (v4)

This is opt-in development work. It does not establish improved skill, a final
ranking, or the user's 0.80 NSE / squared-Pearson-R² target. Existing v1–v3
checkpoint recipes keep their old architecture and equations.

`hydrofusion_context_recipe` selects `zero25`, `absolute25`, or `anchored25`.
All append five channels to the original twenty: current physical hourly mean,
next physical hourly mean (both mm/h), fast and slow storage at origin (mm),
and log1p(Q_origin/q_ref) minus log1p(current_physical_mean/q_ref).
The reference is persistence, the next physical mean, or
max(0,Q_origin+next_physical_mean-current_physical_mean), respectively.
The zero25 control zeros all five added channels but requires the same context
availability. This is a same-width information ablation, not equal effective
capacity. Neural corrections can leave a fixed physical/neural convex interval.
The corrected output is not guaranteed to close the physical water ledger.

`hydrofusion_history_hours` is an ascending list of elapsed-hour offsets beginning
at zero, e.g. `0,1,3,6,12,23`. FFN concatenates the full feature vector at those
steps, newest first. This first implementation uses one explicit history list
for all channels, rather than claiming the full legacy per-input lag GUI applies.
LSTM requires this field to be `0` and uses `lstm_sequence_length` instead.
V4 supports sequence lengths 2–168, subject to complete native history and split
support; a comparison must freeze identical eligible timestamps across arms.
The pilot fixes all sequences to24 and the FFN list above. No history sweep.
Legacy lag controls are rejected in the forecast path instead of silently ignored.

Current/next means and reservoir stores come from the same causal valid-time
routing steps as v3. A dry forecast copy never overwrites committed history.
Each step uses only information available at that step's origin; forcing delays,
release provenance, missing records, gaps and native Q constraints remain active.
The known current Q assumes zero discharge latency; radar release remains an
assumed one-hour retrospective scenario, not operational certification.

Training, managed CLI/GUI forecast reload and verification share context/channel
and reference math. The v4 checkpoint recipe binds width, reference, sequence,
history, objective and execution scope; a managed context-schema sidecar fixes
channel names/units. Original metadata explicitly describes the first20 channels.
Input/target scalers, q_ref and flow-loss scale use final eligible training only.

`hydrofusion_development_only: true` is a real two-split mode. A separately
cropped native package must end at validation_end_x. The loader checks this
before observation conversion; assembly refuses later identities and never
constructs a test split. Exported predictions are labelled validation. Generic
CSV metric column names inherited from the exporter describe this declared
validation scope, not an unseen test result. A development bundle refuses
inference for target identities beyond validation_end_x.

The new physical-settings evaluation is a separate24-candidate radar-only
train/validation search. It found no better validation NSE than the default;
therefore the neural architecture pilot retains fixed physical settings.
`run_hydrofusion_context_pilot.py` requires completed synthetic preflight and pins
source scripts, binaries, configs, support and inputs before six new seed42 fits.
Both FFN and LSTM compare zero25/absolute25/anchored25;50epochs, hidden32,
Adam .001, batch64, fixed log_flow_mse_v2, sequential, no GPU or retries.
The selector uses validation only and keeps MAE/high-flow/bias constraints.
The pilot never launches confirmation fits automatically. A one-seed improvement
requires confirmation; a failed gate and unmet0.80 target must be reported.

Peak metrics can select validation explicitly; the default remains test only.
Generic exported local-tail metrics retain the existing descriptive quantile
definition. Advancement high-flow RMSE is independently recomputed using Q90
from eligible training observations only, not the validation quantile.
