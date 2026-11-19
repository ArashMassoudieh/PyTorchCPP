# Matched causal one-hour comparison: completed bounded study

The approved extension completed 12 neural fits and one fixed physical simulation.
The six earlier FFN fits were reused without retraining. Every extension bundle
fresh-reproduced all 8,387 test rows and regenerated 8,595 validation predictions.
Independent rehashing and complete metric recomputation exactly matched the saved
report. This verifies this bounded experiment, not generalization or deployment.

All methods share the 2015–2016 training, 2017 validation and 2018 test target
support, causal availability policy and native-source restrictions of the original
six-fit study. Neural settings: width32, 50epochs, Adam0.001, batch64, seeds42/43/44;
checkpoint is earliest validation standardized-log-MSE minimum. Two CPUs/no GPU,
12GiB hard address-space and eight hours per fit; no retries or extra fits.
FFN has705 parameters; LSTM6945. LSTM consumes24feature steps; FFN consumes the
last feature vector with engineered historical statistics. Shared eligibility
and width do not make this a capacity- or information-isolated architecture test.

| Method / target | Validation NSE | Test NSE | Test KGE | Test PBIAS % |
|---|---:|---:|---:|---:|
| FFN direct-log-Q |0.331537|0.416321|0.275295|-26.305|
| FFN persistence log-residual |0.614729|0.572354|0.518218|-15.979|
| LSTM direct-log-Q |0.342984|0.468602|0.336355|-24.785|
| LSTM persistence log-residual |0.534460|0.513005|0.411137|-20.164|
| FFN physical log-residual |−0.008159|0.052661|0.468143|-15.751|
| LSTM physical log-residual |0.130851|0.289318|0.370459|-20.275|
| Fixed physical baseline |−4.320727|−4.058273|−1.252802|+176.172|
| One-hour persistence |0.549940|0.535058|0.767526|-0.034|

Neural rows are three-seed means. Physical and persistence are single deterministic
results. Negative PBIAS is underprediction. Physical preset is uncalibrated:
fast1/slow0.1/fraction0.7/gain1, zero storage at true breaks, no observed-state
initialization or parameter sweep. Poor scores do not invalidate physics methods
in general. The physical baseline is a two-reservoir router, not PDE collocation.

Persistence-residual learning beats direct-log-Q in all three seeds on both splits
for FFN and LSTM. LSTM mean paired test gain is+0.044403 (population seedSD0.014417),
validation+0.191476 (SD0.004211). Nevertheless LSTM residual does not beat persistence
on mean test NSE. FFN residual has the best mean NSE here but worse KGE and volume
bias than persistence. Its variability ratio is about0.604 and mean ratio0.840:
compressed variability and mean flow explain its KGE deficit descriptively.
No loss-function cause or post-hoc correction is established by that observation.

Both physical hybrids lose to their own direct and persistence-residual controls
in every seed on validation and test. Diagnose forcing/PET/area consistency,
training-period water ledger and fixed parameter plausibility before proposing
physical calibration. Endpoint flow scoring and integrated reservoir outflow
must not be conflated. Neural corrections do not inherit water-budget closure.

Extension fit wall totaled2390.914s, verification additional; peak sampled RSS
554.6MiB.13fresh-process verifications cover109,031 test rows and156target-free
API checks. Maximum bulk reload difference1.2479526323438961e-7mm/h within
atol1e-5+rtol1e-5*abs(expected); not all runs were bit-exact.

Source/results: Dropbox HydroPINN paper/results/hydrofusion_remaining_methods_2018/
contains frozen screen_plan.json, completion records, all_method_diagnostics.json
and .md, analysis_closeout.json, model bundles and executable copies.
Independent reproduction: audits/hydrofusion_remaining_methods_2018_review/.
Launch/source evidence: audits/hydrofusion_remaining_methods_2018_launch/.
Saved analysis: analyze_hydrofusion_all_methods.py; each artifact/input is hashed.

Only one reused development fold; idealized retrospective availability, not
verified provider publication latency. Native qualifiers/support/area caveats
remain. No reserved observations, original/frozen output replacement, manuscript
edit or push. No prospectively frozen events, so no event-specific claims.
All approved fits are spent. New training, loss tuning or physical calibration
requires a separately explicit budget; no all-five improvement or paper-promotion
claim is accepted. Persistent supervision remains active until human stop.
