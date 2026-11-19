# Completed approved six-fit FFN screen

The causal timing fixes and all six approved research fits are complete. No
additional research fit, retry, push or manuscript change was performed.

Train2015–2016, Validation2017, Test2018; one-hour interval-end discharge forecasts.
Two matched FFN arms, seeds42/43/44, width32/tanh, Adam0.001, batch64,50epochs.
Shared16182/8595/8387 training/validation/test targets, identical input scalers
and training-only q_ref. Prepared source/qualifier/coverage/availability policy,
configuration, support and executable hashes were frozen before launch.

| Mean NSE | Direct log-Q | Persistence log-residual | Free persistence |
|---|---:|---:|---:|
| Validation | 0.331537 | 0.614729 | 0.549940 |
| Test | 0.416321 | 0.572354 | 0.535058 |

Residual-minus-direct NSE gains are positive for all three seeds on both splits.
Mean validation gain+0.283192 (population seed SD0.020097); mean test gain+0.156033
(SD0.007122). Validation MAE improves24.01% against direct prediction. All three
predeclared validation advancement conditions pass. This does not authorize
another training stage.

The persistence comparison has a material tradeoff:

| Test mean metric | Direct | Residual | Persistence |
|---|---:|---:|---:|
| KGE | 0.275295 | 0.518218 | 0.767526 |
| RMSE (m³/s) | 1.308854 | 1.120351 | 1.168187 |
| MAE (m³/s) | 0.145753 | 0.117328 | 0.138983 |
| PBIAS (%) | -26.305 | -15.979 | -0.034 |

Negative PBIAS means underprediction. Residual prediction improves NSE/MAE/RMSE
here, but persistence has better KGE and volume bias. Do not describe the result
as across-the-board superiority or water-budget closure. Discharge conversion
uses the actual declared23754600m² area, not a legacy display-area constant.

All6checkpoints were reloaded in independent processes and compared against ALL
8387test predictions per fit (50322 comparisons total), plus12target-free production
API checks per fit (72total). Largest all-row reload absolute difference:
1.6834840915747051e-7mm/h; tolerance1e-5+1e-5*abs(expected), not bit-exact for every
run. All8595validation predictions per fit were independently regenerated from
the saved checkpoints. Selected epochs: direct44/34/48; residual33/31/23 for
seeds42/43/44. Each equals the earliest minimum validation standardized-log-MSE.

Fit wall time total694.835s (includes preparation/fitting/export; fresh reload
verification is additional). Peak sampled RSS519.7MiB;42MB retained fit artifacts.
Hard12GiB address-space limit, two-CPU affinity, noGPU,8h/fit timeout and250MB/fit
artifact ceiling stayed in effect. No automatic retry or budget expansion.

## Interpretation limits and next work

- One reused development fold with three seeds: exploratory evidence, not an
  independent generalization result or a manuscript promotion decision.
- This discharge-informed one-hour task differs from earlier forcing-only or
  same-hour studies. Their NSE values are not interchangeable before/after scores.
- Only FFN was research-trained. The shared causal adapter passed fabricated
  CLI/reload checks for all five families; those are not all-five research results.
- Availability is explicitly idealized: zero Q-release delay,24h forcing/PET
  delays. Actual provider publication latencies are not verified.
- Source A-only qualifier/300s maximum support/75%coverage assumptions, native
  gaps, area and parent-prefix provenance caveats remain in the pinned inputs.
  Parent whole-file hashes were deliberately not computed across reserved data.
- Event windows were not prospectively frozen; no event-level result is claimed.

Next authorized work can diagnose the KGE/volume-bias tradeoff from existing
outputs and propose a training-only remedy and a matched LSTM comparison. No
post-hoc test-selected correction or further research fits are authorized by the
spent six-fit budget.

## Saved evidence and reproduction

Campaign: `/home/arash/Dropbox/HydroPINN paper/results/hydrofusion_causal_six_fit_2018/`.
Read `screen_plan.json`, `screen_completion.json`, `screen_summary_balanced.json`
and `.md`, per-fit `completion.json`/`fresh_reload.log`, and the input provenance.
Original NSE-first summary is preserved; balanced summary leaves every score and
advancement condition unchanged and makes KGE/volume tradeoffs visible.

Scripts: `run_hydrofusion_six_fit_screen.py`,
`analyze_hydrofusion_six_fit_screen.py`, `prepare_hydrofusion_causal_package.py`,
`slice_hydrofusion_producer_prefix.py`; C++ fresh verifier
`hydrofusion_native_ffn_reproduction.cpp`. Do not relaunch: the exclusive claim
intentionally blocks duplicate or resumed training. Analysis accepts a NEW output
filename and verifies frozen/completed file hashes before computing metrics.

Acceptance audit: `audits/hydrofusion_causal_native_20261118T234203Z/` under the
HydroPINN paper root. It retains source/build/link identities, binaries, fabricated
acceptance, legacy regression and the campaign closeout. No reserved targets,
original manuscript or frozen candidates were changed.
