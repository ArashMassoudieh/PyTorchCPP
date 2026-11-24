# Physical forecast recovery: component separation and discharge initialization

Two separate finite research stages were frozen and completed on 2026-11-24.
Neither changes the production physical recipe or existing hybrid checkpoints.
All fitting used training targets <=17543; validation was one frozen candidate
per stage on 7112 targets <=26303, with no test or reserved observations.

## Component separation — rejected

`run_preissue_component_screen.py` fits historical-storage response and future
instantaneous-rain response separately, optionally adding a constant background
source. It uses 27 saved TRAIN-only routing shapes from the previous calibration;
it never reruns the closed routing search. There are 54 new bounded linear fits.
`preissue_physical_components.py` solves the <=3-dimensional box problem by
exhaustive active-face enumeration. The independent verifier checks all 54 KKT
certificates, training support, fitting/selection and validation rejection.

Validation NSE/R² = .414241/.511062, bias -1.140%, MAE .028834 mm/h.
This fixes the gross volume bias but loses to the dry reference's NSE .448204
and worsens high-flow RMSE. The added background is an explicit empirical source,
not rainfall or a claim of catchment water-budget closure. No promotion.

## Observed-discharge state initialization — promising, held

The new experimental `hydrofusion_discharge_state_forecast.h` reconstructs a
fresh state at each origin from Qorigin, the native mean discharge of the hour
ending at that origin. Assuming dry recession over that preceding hour, divide
its observed outflow into fast/slow fractions f and 1-f. The initial storage of
each branch is Qbranch/(exp(k)-1). Its next-hour mean with no new rain is then
Qbranch*exp(-k). Available preissue rain minus available PET enters the next-hour
reservoir step. This is a causal discharge analysis, not the original free-running
physical baseline, a neural PINN loss, or full state assimilation with uncertainty.
The dry reconstruction assumption is an approximation during preceding-hour rain.

Initial water is counted explicitly as analysis storage for each one-hour ledger.
Repeated analyses are not a continuous basin water ledger. All neural methods
already use origin discharge, but changing the physical model's information use
must be stated explicitly. Qorigin's zero latency remains an idealized retrospective
assumption; archive rain timestamps are release proxies, not delivery certificates.

`extract_discharge_state_features.cpp` uses the production native loader and
hourly reservoir equations. It accepts a physically cropped training package and
verifies origin/target identities, source discharge, support and weather history.
No target observation is an input to `dischargeStateForecast()`.

`run_discharge_state_screen.py` declares 27 routing shapes, fits one gain bounded
[0,1] under a training volume constraint, and freezes the chosen JSON before
validation. The same selected formula is reproduced by a separately executed
C++ program for every training and validation row. Training selection is fast
rate .1/hour, slow rate .05/hour, fraction .7, gain 1.0.

Validation NSE/R² = **.671599/.690686**, versus persistence .548222/.599914.
MAE .014937 versus .015189 mm/h; high-flow RMSE .591906 versus .698795.
Bias +4.095838% versus +.516971% fails the prospective +2-point increase guard.
Thus the user's .60 dual-metric target is met on reused validation, but the model
is **not advanced**. Do not relax the gate after seeing this result.

The independent verifier uses scalar math to re-fit/recheck every training gain,
selection, input/artifact/source hashes, all 7112 forecasts, metrics and gates.
Observed-defined event windows match the earlier v5 diagnostic: 28 complete
windows, 9 incomplete excluded. Mean absolute peak-timing error is 2.464 hours
versus persistence's 1.0; a wrong peak in a long event contributes a 23-hour error.
These are descriptive diagnostics, not event-wise selection. No uniform skill claim.

The prototype is NOT integrated into HydroBatch/GUI dispatch, managed artifact
schemas, or production reload. The fresh C++ research process and JSON parameter
replay must not be described as production CLI acceptance. Integration, bias/timing
resolution and prospective confirmation remain separate work; no manuscript claim.

## Reproduce development checks

From repository root:

```sh
python3 -m unittest discover -s Hydro/experiments/gistohq_sligo -p test_preissue_physical_components.py
python3 -m unittest discover -s Hydro/experiments/gistohq_sligo -p test_discharge_state_screen.py
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
 Hydro/experiments/gistohq_sligo/discharge_state_forecast_test.cpp -o /tmp/discharge_state_test
/tmp/discharge_state_test
g++ -std=c++17 -O2 -fPIC -Wall -Wextra -Werror \
 Hydro/experiments/gistohq_sligo/extract_discharge_state_features.cpp \
 Hydro/dataset/hydro_checksum.cpp -o /tmp/extract_discharge_state \
 $(pkg-config --cflags --libs Qt5Core) -lcrypto
```

All 12 Python tests and 11 C++ checks pass. ASan/UBSan passes with leak detection
disabled because sandbox ptrace prevents LeakSanitizer. Production sources were
not changed; old campaigns and their failures remain immutable. Do not rerun these
closed research stages. Evidence and saved source/binary snapshots are in Dropbox
HydroPINN paper/results/hydrofusion_preissue_physical_components_20261124 and
hydrofusion_discharge_state_screen_20261124. See their independent_review.json.
