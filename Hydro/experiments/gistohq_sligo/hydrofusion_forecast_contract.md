# HydroFusion opt-in 1-hour forecast information contract (Task 1)

Scope: a new, opt-in forecasting path only. Does not change the existing frozen
six-feature model's input size, checkpoints, or default behavior. Source
motivation: Batmaz, Arin & Tekin (2026), JAWRA 62(6), e70172 (`batmaz2026hydrofusion`),
read at the level stated in `CLAUDE_HYDROFUSION_IMPLEMENTATION_TASKS_20261115.md`
(Methods 3.1-3.3, Tables 3-7; supplement S1-S11 lag/gate equations were not
supplied and are not reproduced here). This document and its companion schema
(`hydrofusion_forecast_contract_schema.json`) are the Task 1 deliverable: the
contract, schema, and prospective split/support rules, frozen before any
feature-builder code (Task 2) is written.

## 1. Horizon

Fixed at **1 hour** for this prototype. No other horizon is implemented or
silently swept. A sample's **origin** is an hourly row at exported-x index `o`;
its **target** is the hourly row at `o + 1`.

## 2. Availability, not just timestamp

An hourly row's value at index `h` (covering the interval `[h, h+1)`) is not
known at the start of that interval. Its **availability index** is
`avail(h) = h + 1 + latency_hours`, where `latency_hours` is an explicit,
named parameter (see schema `latency_hours`), not an implicit zero. This
prototype sets `latency_hours = 0.0` and labels the setting **idealized** in
the schema (`latency_model: "idealized"`) — it does not claim operational
telemetry readiness. Any future real-latency run must set `latency_model` to
a non-idealized value explicitly before being described as operational.

A sample's **decision time** is `avail(o)`. An input row at index `h` is
usable for a sample with origin `o` iff `avail(h) <= avail(o)`, which — since
every input row shares the same `latency_hours` — reduces to `h <= o`. **No
input feature may read index `h > o`.** The target row at `o + 1` is read
exactly once, as a label, never as a feature, threshold, origin flag, or
model input (matches `CLAUDE_HYDROFUSION_IMPLEMENTATION_TASKS_20261115.md`
Task 3's "future Q as a LABEL only" rule, generalized here to Task 1's
contract level since it governs sampling, not just the log-residual target).

**Caution on `latency_hours > 0` (independent-review correction, twice
corrected):** the `h <= o` reduction above is not merely "less ideal" once
`latency_hours` becomes positive — it can silently invalidate the whole
1-hour-ahead framing. The target row covers the absolute-time interval
`[o+1, o+2)`; decision time is `avail(o) = o + 1 + latency_hours`. For any
`latency_hours > 0`, decision time already falls *inside* that interval
(`o + 1 + latency_hours > o + 1`) — the forecast is being issued partway
through the very interval it is trying to predict. Once `latency_hours >=
1`, decision time reaches or passes the interval's own *end* (`o + 2`) --
at that point the target interval has already fully elapsed by the time a
decision could be made, so there is no future left to forecast at all. (An
earlier version of this paragraph compared decision time against
`avail(o+1) = o + 2 + latency_hours` instead of the interval's actual end
`o + 2` — those two quantities differ by exactly `latency_hours` and can
never cross each other by construction; that was the wrong threshold and
is corrected here.) This is a structural limit of the current origin/
target/horizon framing, not a detail `latency_model` alone communicates. This
prototype supports `latency_hours = 0.0` only; any `latency_hours > 0` run
requires a revised contract (a wider horizon, a different origin
definition, or both) before it can be called operational, not just a
relabeled `latency_model`. No code in this prototype currently threads
`latency_hours` through `buildSample` — it is a documented constraint on
future use, not yet an enforced runtime parameter.

## 3. Reserved-period seal — check before reading any value

Reuses the existing seal constant and ordering convention (duplicated per
file, matching this codebase's own established practice — there is no shared
header exporting it; see `evaluation/shared_scale_inference.cpp:130-131`,
`experiments/gistohq_sligo/verify_stageb_checkpoint_reproduction_parsing.h:29`):

```
kSealStartX = 77016   // development-only seal; holdoutMode stays false here
```

For every candidate sample, **both** `o >= kSealStartX` and
`(o + 1) >= kSealStartX` are checked against the raw x-index array **before**
indexing into any observation/feature array for that row — including when the
offending row belongs to a different approach sharing the same file, matching
the fail-closed guard already added to `verify_stageb_checkpoint_reproduction_parsing.h`
for the Stage B tool. A sample failing either check is rejected outright, not
truncated or clipped. `holdoutMode` defaults to `false` for every HydroFusion
entry point: only development-only packages are opened; a full
reserved-containing package is never opened and filtered afterward, because
(per the Task 1 investigation) no package-kind flag exists in
`GisToOhqPackageConfig`/`loadHydroPackageTensors` — enforcement is entirely
the caller's responsibility, exactly as the existing `requireDevelopmentOnly`
pattern already assumes.

## 4. Fixed input allowlist (v1 prototype)

No column outside this list may be read as a model input. Twenty features,
frozen here and only extended by a future, separately-reviewed contract
revision:

**Discharge-memory group** (native unit mm/h, reading
`observed_runoff_mm_per_hour`, gated by `discharge_valid` at every row read):

| name | definition |
|---|---|
| `q_lag_0h` | `Q(o)` |
| `q_lag_1h` | `Q(o-1)` |
| `q_lag_3h` | `Q(o-3)` |
| `q_lag_6h` | `Q(o-6)` |
| `q_lag_12h` | `Q(o-12)` |
| `q_lag_24h` | `Q(o-24)` |
| `q_trailing_mean_6h` | mean of `Q(o-5..o)` (6 complete hourly rows) |
| `q_trailing_mean_24h` | mean of `Q(o-23..o)` (24 complete hourly rows) |
| `q_trailing_mean_72h` | mean of `Q(o-71..o)` (72 complete hourly rows) |
| `q_trailing_var_6h` | population variance of the same 6-row window |
| `q_trailing_var_24h` | population variance of the same 24-row window |
| `q_trailing_var_72h` | population variance of the same 72-row window |
| `q_last_hourly_change` | `Q(o) - Q(o-1)` |
| `q_past_slope_6h` | `(Q(o) - Q(o-6)) / 6`, explicitly fixed to the 6h window (the spec only says "a simple past-only slope"; this prototype's specific choice, stated so Task 2 does not re-derive it ad hoc) |

**Meteorological-at-origin group** (the existing frozen model's six harmonizer
fields, read only at `h = o`, gated by their own existing per-field validity
flags and `all_forcings_valid`; no additional lags in v1):

| name | harmonizer field |
|---|---|
| `met_precip_mm_per_h` | `precipitation_mm_per_hour` |
| `met_temperature_c` | `temperature_c` |
| `met_relative_humidity_pct` | `relative_humidity_percent` |
| `met_wind_speed_m_per_s` | `wind_speed_m_per_second` |
| `met_solar_mj_per_m2_per_h` | `solar_energy_mj_per_m2_per_hour` |
| `met_pet_mm_per_h` | `pet_mm_per_hour` |

No snow/spring branch is included (Batmaz's snow regime is not established
for Sligo; not borrowed by analogy, per the source spec).

## 5. Two distinct masks (never conflated)

- **Predictor-availability mask**: true only if every row the allowlist reads
  — `o-71 .. o` for the discharge-memory group, `o` for the meteorological
  group — exists, is valid for the specific field(s) it feeds (not merely
  `all_forcings_valid`, since the discharge group needs `discharge_valid`
  specifically), and shares the **same segment id** as `o` (no cross-gap
  lookback). "Complete-window" means every calendar hour in the window has a
  real row; a missing hour voids the whole window rather than being
  substituted with an older value or zero.
- **Target-native mask**: true only if the target row at `o + 1` has
  `discharge_valid = true` (a real, non-interpolated observation). An
  interpolated target is not evaluation truth and may not be scored as if it
  were.

A sample needs the predictor-availability mask for any use (training or
inference) and additionally needs the target-native mask to serve as a
labeled training/evaluation example. These follow the same "applied
identically before training loss and before scoring, for every arm" policy
already established for the multiscale daily-history branch
(`models/multiscale_lstm_module.h:89-97`).

## 6. Segment ids from actual chronological gaps

Segment ids for the hourly discharge-memory lookups are derived the same way
as `daily_history_builder.h`'s `buildHourIndex`: build an hour-index map keyed
on the integer hour value of `x` (reject non-finite/non-integer x), reject
duplicate or non-monotonic timestamps, and assign a new segment whenever the
elapsed-hour gap between consecutive rows is not exactly 1 — never inferred
from row position. A lag or trailing-window read that would cross a segment
boundary fails the predictor-availability mask for that window.

## 7. Split-boundary reconciliation

Split semantics (not implementation) follow the existing
`configuredChronologicalSplit` / `resolveTestEndSequenceCount` path exactly.
Boundaries are half-open exactly as elsewhere in this codebase: `train_end`
is excluded from train and is validation's first index; `validation_end` is
excluded from validation and is test's first index; `test_end` is excluded
from test (`x >= test_end` is `"beyond_test_end"`). `train_end_x` /
`validation_end_x` / `test_end_x` are inclusive cutoffs in the exported-x
coordinate space (a row whose endpoint x equals the cutoff is the **last**
row included in that partition).

**Implementation note (independent-review correction):** this prototype's
code (`hydrofusion_feature_builder.h`'s `partitionForX`/`assignSampleSplit`)
does **not** call `configuredChronologicalSplit` itself — that function
takes a torch tensor and resolves windowed-sequence array positions, neither
of which exist at this builder's raw-per-row granularity, and this module
deliberately stays torch-free (Task 2's "built/tested without training"
requirement). `partitionForX` is an independent, narrowly-scoped x-space
adapter that reimplements only the cutoff comparison. Its parity with the
production path is verified against `resolveTestEndSequenceCountFromSorted`
(`dataset/resolve_test_end_x.h`) — the same pure, torch-free core the
production path itself calls — rather than assumed. The original version of
this section overstated this as calling the production path "only"; that
was inaccurate and is corrected here.

**Sample-to-split assignment rule (new, specific to this 2-row-wide sample):**
a sample with origin `o` and target `o + 1` is assigned to split `S` **iff
both `o` and `o + 1` resolve to the same partition `S`**. A sample whose
origin and target straddle a split cutoff (i.e. `o`'s partition differs from
`o+1`'s) is **discarded entirely** — never partially assigned to either side.
This is the concrete form of "no sequence/target may cross its assigned
split" for this contract.

**Historical-context exception, frozen explicitly:** input rows used for the
discharge-memory lookups (`o-71 .. o`) may fall in an earlier split than the
sample's own assigned split — e.g. a validation-origin sample may read
train-split discharge history. This is allowed because it is purely causal
historical data, not a future label. What is never allowed is the converse:
for a train-split sample, `target_x > train_end_x` (strictly past the
cutoff) is forbidden; `target_x == train_end_x` is allowed, consistent
with the inclusive exported-x cutoff (`train_end_x` itself is the last x
value included in train) -- no future label may enter training.

## 8. What this contract does not cover

Feature-builder implementation, the persistence benchmark, and the
log-residual target/reconstruction formulation are Task 2 and Task 3,
built against this frozen contract, not restated here. Integration/leakage
tests (Task 4) must exercise every rule above at its real entry point, not
merely unit-test the arithmetic in isolation.

**Caveat on `discharge_valid` (independent-review correction):** in this
prototype, `discharge_valid` is a caller-supplied boolean on `HourlyRow` --
the prototype trusts it (in addition to the numeric finite/nonnegative
checks added after the independent review below) but does not and cannot
prove it reflects genuine *native* (non-interpolated) discharge support.
That proof has to come from whatever adapter eventually populates
`HourlyRow` from the real `GisToOhq` loader (Task 4's responsibility when
this builder is wired to real data) -- a boolean field name alone does not
establish the guarantee the contract's section 5 describes.
