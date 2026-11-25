# Joint internal selection and ten-year readiness

The 2026-11-24 joint screen is a NEW finite campaign. It must not relaunch any
closed preissue, radar, SiLU, context, or physical calibration stage.

The accepted 2015–2016 native training package and weather sidecar are physically
cropped at target17543. Its new internal split fits through target8759 (2015)
and selects through17543 (2016). Real native feature assembly yields6111inner
training and6943inner validation rows. A24-hour sequence cannot straddle the
boundary; all neural and physical candidates use exactly this support.

Eight new neural fits compare32vs64hidden units for FFN, LSTM, FFNhybrid and
LSTMhybrid, seed42,50epochs. All other recipe settings remain as in the verified
v5 held-rate pilot. Each fit is sequential with8h/2CPU/noGPU/12GiB/250MB caps,
no retries or automatic expansion. Width64 must pass all five frozen internal
NSE/R²/MAE/tail/bias guards against32; otherwise retain32. Every completed model
must pass fresh native reload on all6943rows plus12target-free API checks,
training-only scaler/q_ref/flow-scale verification and independent receipt/hash
and scalar score checks. Check current claim/completion/failure before action.

A separate27-shape discharge-initialized physical prototype fits gain using only
internal TRAIN, then selects on internal validation under all5guards against
persistence. This is not the unchanged free-running PINN. Its method/analysis
initial water/idealized observed-Q latency limitations remain. Retain the original
dry physical production reference if no prototype clears the gates.

No2017validation observations or test data enter this screen. Physical feature
rows are copied from the previously audited TRAIN-only feature file, checked
against the new common support. This is not a repeated old fit: the internal
training period and selection protocol are different.

## After internal selection

Freeze each family's chosen width, selected epoch and physical shape before
any final full-TRAIN refit. The current production wrapper selects checkpoint
weights by validation loss. A distinct explicit fixed-epoch/last-checkpoint
policy, exported and verified through real CLI/reload, is required before a new
final refit can use2017only for reporting instead of checkpoint selection.
There are NO outer refits in this campaign's claim. Do not silently perform them
or reuse its spent budget. Existing completed seed42jobs must not be duplicated
as if they were new evidence. Any final changed-policy refit needs a separately
frozen finite plan under standing human research authorization.

A better internal score is not a verified improvement on2017or a paper finding.
Preserve old seed42results, previous bias/timing failures, all scripts and inputs.
Later cross-seed confirmation and a proper prospective holdout remain necessary.

## Fixed-epoch refit policy

The opt-in `hydrofusion_checkpoint_selection=fixed_last_epoch_v1` saves only
that run's final epoch. The internally selected epoch becomes the new run's
`epochs`; outer validation losses are diagnostics and cannot select weights.
The default `best_validation_v1` preserves historical checkpoint selection.
The fixed policy is exported in config and bound into the model recipe, so
managed reload rejects a changed policy even after public file hashes are updated.
Unknown, duplicated, non-string or inapplicable policy fields fail closed.

`verify_hydrofusion_fixed_epoch_cli.py` runs all four forecast families on a
fabricated native fixture, changing only validation discharge and checking exact
weight/scaler equality, fresh reload and policy-tamper rejection. The separate
`run_hydrofusion_joint_fixed_epoch_refits.py` prepares four new fixed-policy
fits and one physical observer replay only after this acceptance. It freezes
internal width/epoch/shape decisions, refits physical gain using full TRAIN only,
and reports reused 2017 validation without test scoring or automatic expansion.
These are research refits, not ten-year evaluation or a general skill claim.

## Ten-year audit

`audit_joint_study_readiness.py` opens only three explicit JSON metadata files,
not observation or forcing CSVs. The old package declares2015–2024 but carries
area23.7546km² rather than the corrected gauge's16.7375km² and records a failed
`temporal.study_period_coverage` QC rule. Full87672-hour coverage is not proven.
The new native/rain recipe has only bounded development input acceptance.

A full ten-year run also crosses reserved target77016 (2023-10-15UTC). This audit
neither authorizes unsealing nor reads those targets. Extending inputs, reconciling
catchment/coverage, defining untouched evaluation and any explicit reserved-data
decision must precede a ten-year fitting claim. No automatic area rescaling,
missing-rain zero-fill, new network acquisition or old-output replacement.

Evidence: Dropbox HydroPINN paper/results/
`hydrofusion_joint_inner_selection_20261124/` and audits/
`joint_ten_year_readiness_20261124/review.json`. The12-hour supervisor remains
active until explicit human stop; Codex work must not become fakeClaudecheckpoints.


## Opt-in seeded TRAIN-window order

The forecast path can opt in with the existing `shuffle_training=true` field
and nonnegative `random_seed`. Accepted chronological configurations retain
`shuffle_training=false`; the global configuration defaults are unchanged.
Physical-only forecasts reject this option. A local RNG keyed by seed and epoch
permutes complete eligible TRAIN sample rows; it does not consume torch model
initialization RNG or reorder timestamps inside a window. Inputs, transformed
targets, physical references and observed-discharge loss terms share one index.
Scalers/q_ref/flow scale are fitted on canonical eligible TRAIN data before this
permutation. Physical contexts remain chronologically assembled.

The model recipe binds seeded training order (marker7,version1) and seed; fixed
checkpoint-policy binding remains distinct. Exported booleans/seed are managed
config fields. Rehashing public inventory after relabelling order or seed must
not permit reload. Reproducibility is checked within the pinned build/runtime;
std-library permutation portability across environments is not asserted.

This is an optimization-mode change, not a new input signal or physical method.
The four neural forecast families may use it; physical-only skill is unaffected.
Production acceptance requires synthetic full/partial-batch alignment, local-RNG
isolation, deterministic same-seed training, validation-Q counterfactual weight
isolation, native fresh reload, tamper rejection and disabled-path legacy byte
compatibility. Acceptance of these checks does not authorize reuse of a closed
research campaign or establish a performance gain. Any new chronological-fold
comparison needs a separate finite frozen plan, with no automatic expansion.
