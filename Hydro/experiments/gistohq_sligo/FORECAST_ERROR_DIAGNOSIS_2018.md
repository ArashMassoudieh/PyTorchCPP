# Why the bounded causal forecasts struggle

Read-only inspection of all18saved neural checkpoints and the fixed physical
reference. No optimizer steps, calibration, new research fits, test-selected
correction, manuscript edit or original-output replacement occurred.

## Physical reference: excessive effective input, not numerical water creation

On matched training hours, precipitation totals2018.000mm, PET demand662.249mm,
and max(P-PET,0) totals1841.890mm, versus507.881mm observed discharge:3.627times.
Test effective input1536.351mm versus556.298mm discharge:2.762times. These are
matched-hour sums, NOT a complete-year catchment water balance.

The fixed recipe routes all effective input through two outlet stores at gain1.
It has no soil-water/infiltration/runoff-generation loss model. runoff_coeff0.7
is the FAST/SLOW PARTITION, not a runoff-loss coefficient. PET subtracts only
instantaneous rainfall; dry-hour PET never depletes stored water. Thus most
PET demand is not realized: the training rainfall reduction is176.110mm rather
than662.249mm. Subtracting all PET alone would still leave surplus input.
Routing rates/partition alone cannot remove that surplus over a complete recession.

The exact hourly storage ledger closes within4.55e-13mm. Independent conversion
matches34,093native Q rows within8.89e-16mm/h. Rain mapping and daily-PET conversion
match exactly over35,017forcing-valid origins. No factor-of24/3600 or double-area
conversion error was found in this checked path. Actual gauge-drainage-polygon
agreement with the declared23,754,600m² area remains a distinct audit question.
Input forcing error and missing runoff losses are not causally separated here.

## Confirmed correctness defect: endpoint rate scored against an hourly mean

The router returns instantaneous end-of-hour flow; observed targets are native
hourly means. Starting from zero storage with one hour of unit effective rain:
endpoint0.47103316576920246, exact interval mean0.27202786292788839mm/h.
The state integrator is correct; the output aggregation contract is mismatched.

Rescoring that SAME fixed router with hourly integrated outflow changes test
NSE only from-4.058273 to-3.952594. PBIAS remains approximately+176.172%.
This confirms that aggregation is a real defect but not the main volume remedy.
No recipe/calibration sweep or new accepted performance result follows.

The imposed24h met lag also gives the physical router yesterday's storm, with
no measured-Q state correction. This is a disclosed conservative availability
assumption, not verified provider latency. Do not replace it with unavailable
future/current rainfall just to raise retrospective skill.

Any aggregation fix must be explicitly versioned. Never silently change
causal_two_reservoir_v1 or reconstruction of existing hybrid checkpoints:
the learned residual itself depends on the exact old reference.

## Learned direct FFNs cannot reach the largest observed peaks

All18checkpoints were inspected using the real loader and their own saved scalers.
For the fitted direct FFNs, bounded tanh outputs and final-layer weights imply
absolute Q ceilings1.373/1.743/1.489mm/h, whereas observed training/test peaks
are5.797/7.218mm/h. Respectively53/35/48test observations exceed these bounds.
These checkpoints cannot reproduce those peaks even with ideal inputs. The bound
alone limits their best possible test NSE to0.634/0.715/0.662; actual scores are
lower, so this is not the sole source of error.

These are properties of the CURRENT fitted weights, not immutable limits of the
architecture: a different fit can change them. LSTM direct bounds9.99–10.84mm/h
exceed observed peaks, so this specific bound does not explain its remaining
errors. Persistence-residual reconstruction has no corresponding fixed absolute
Q ceiling because it depends on Q at origin. Counts/derivation are saved in
checkpoint_ceiling_diagnosis.json and checkpoint_capacity.csv.

## Neural objective and high-flow compression

| Group | Training mean PBIAS % | Test mean PBIAS % | Test clipping fraction |
|---|---:|---:|---:|
| FFN direct |−12.428|−26.305|0|
| FFN persistence residual |−9.927|−15.979|0|
| LSTM direct |−10.815|−24.785|0|
| LSTM persistence residual |−9.592|−20.164|0|
| FFN physical residual |+1.083|−15.751|0.020945|
| LSTM physical residual |−8.938|−20.275|0.004253|

Thus clipping does not cause the direct/persistence-residual bias. Bias is present
on training data too. Their test squared error is more than99.97% concentrated
above the TRAIN-onlyQ90 threshold. High-flow share is10.0% on training versus19.3%
on test; mean discharge increases from0.03139 to0.06633mm/h. This is descriptive
shift on a reused development fold, not a test-selected threshold or blind result.

The fitted loss/checkpoint criterion is standardized log-target MSE, not arithmetic
flow-volume error, physical NSE or KGE. Direct models explain about92% of test
log-target variance while physical NSE is0.416/0.469. Exponentiating a log point
estimate generally does not return an unbiased arithmetic mean. Persistence-
residual heads explain only about12–14% of their log-innovation variance; their
strong reference does much of the prediction work. These different target-space
figures are NOT interchangeable model rankings or squared Pearson correlation.

The evidence supports a peak/objective/reference problem; it does not isolate
one cause from representation, optimizer dynamics, covariate shift or missing
contemporaneous forcing. It does not justify a test-derived correction multiplier.

## Next work, in order

1. Add an opt-in VERSIONED interval-mean physical reference with analytic pulse,
   ledger, gap/reset, source-availability and v1 reload-compatibility tests.
2. Audit training-only rainfall/PET aggregation, drainage-area alignment and
   runoff-generation/water-loss assumptions. Do this before physical calibration.
3. Specify one explicit physical-volume-aware neural loss or training-only bias
   method, retain the log-loss control, validate its export/reload semantics,
   and freeze a prospective matched budget/selection rule. Current extra-objective
   flags are rejected: do not silently enable an unsupported setting.

No additional research fits are approved by this diagnosis. All original scores
and the paper remain unchanged; correct code is not proof of better skill.

## Saved evidence and reproducibility

Dropbox HydroPINN paper/results/hydrofusion_error_diagnosis_2018/ contains all18
train/validation/test diagnostic CSVs, per-checkpoint logs, physical_inputs.csv,
checkpoint_capacity.csv, diagnosis.json/.md and diagnosis_closeout.json.
Every checkpoint first reverified all8,387test predictions and12target-free API
checks before raw target-space diagnostics. Shared immutable prepared tensors
were reused only after matching manifest/interval SHA256 and fixed cutoffs.
Six completed uncached CSVs match the cached implementation BYTE-FOR-BYTE.
The deliberately cancelled inefficient initial diagnostic attempt is preserved;
no research fitting was interrupted or repeated.

Saved sources: diagnose_hydrofusion_physical_inputs.cpp,
diagnose_hydrofusion_neural_errors.cpp, inspect_hydrofusion_checkpoint_capacity.cpp,
run_hydrofusion_error_diagnostics.py, analyze_hydrofusion_error_diagnosis.py,
analyze_hydrofusion_checkpoint_capacity.py and finalize_hydrofusion_error_diagnosis.py.
Builds used the previous accepted frozen batch headers/objects; live production
files were not changed.40Python test executions passed; the new torch-free
physical regression passed ASan/UBSan with leak detection disabled. No claim of
sanitized LibTorch. Source/link/binary evidence is retained in the diagnostic audit.
