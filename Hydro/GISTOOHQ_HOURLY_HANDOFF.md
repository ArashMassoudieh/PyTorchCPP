# GIStoOHQ temporal package to HydroPINN hourly handoff

## Status and scope

This document defines the preprocessing boundary for the `SligoCreekDemo`
GIStoOHQ export (`2024-01-01` through `2024-12-31`, USGS gauge `01650800`).
It is the contract to implement before the three native-resolution temporal
assets are admitted to model training. GIStoOHQ remains responsible for source
acquisition, provenance, and QC. HydroPINN owns temporal alignment, validity
masks, model features, normalization, chronological splits, and lags/sequences.

The native export is not the canonical `observations.csv` package currently
accepted by `DDRRLoader`: it contains `manifest.json`, `variables.json`, and
`observations/temporal_1.csv` through `temporal_3.csv`. It therefore must not be
passed directly to `loadHydroPackageTensors` until a temporal-package adapter
implements this contract.

## Important compatibility blocker

The incoming variable set has meteorological forcing, PET/ET, and observed
discharge, but no observed `storage` variable. The existing HydroPINN
`water-balance` contract requires storage and deliberately refuses to synthesize
it. Calling the producer package `water-balance-v1` does not make it compatible
with that consumer contract.

The first adapter must expose this dataset as an observed rainfall-runoff
training table with a discharge-validity mask. Water-balance PINN execution must
remain disabled for this package until one of these separately reviewed changes
is made:

1. GIStoOHQ supplies a documented observed/storage-proxy variable; or
2. HydroPINN introduces a new, versioned latent-storage physics profile whose
   equations, initial conditions, identifiability constraints, and evaluation
   semantics do not claim that storage was observed.

No zero-filled or interpolated storage column is permitted.

## Canonical hourly grid

* Use UTC half-open bins `[T, T + 1 hour)` aligned to whole UTC hours.
* The requested leap-year interval produces 8,784 bins from
  `2024-01-01T00:00:00Z` through the bin beginning `2024-12-31T23:00:00Z`.
* Parse timestamps as instants; do not compare timestamp strings or apply a
  local timezone.
* Reject duplicate samples for the same variable and timestamp unless the
  producer declares a deterministic duplicate-resolution rule in provenance.
* Preserve a source-coverage count and coverage fraction for every aggregated
  value.

## Variable transformations

| Output | Source | Hourly transformation | Canonical unit |
| --- | --- | --- | --- |
| `P` | `PRECTOTCORR` | Interpret each hourly record in its declared `mm/day` unit as a rate and divide by 24. Do **not** sum 24 rate values. | `mm/h` |
| `T` | `T2M` | Arithmetic mean of finite samples in the hour. | `degC` |
| `RH` | `RH2M` | Arithmetic mean; reject values outside `[0, 100]`. | `%` |
| `wind` | `WS2M` | Arithmetic mean; reject negative values. | `m/s` |
| `solar` | `ALLSKY_SFC_SW_DWN` | Preserve the declared hourly interval energy for one-record bins; otherwise sum interval-energy contributions after overlap weighting. | `MJ/m2/h` |
| `PET` | `EVPTRNS` | Hold the daily rate constant over its 24 UTC hourly bins, convert `MJ/m2/day` to equivalent water depth with the declared latent heat, then divide by 24. | `mm/h` |
| `Q_observed` | USGS `00060` | Duration-weighted mean of finite instantaneous discharge samples overlapping the hour. | `m3/s` |
| `Q_runoff` | derived | Convert valid `Q_observed` using the package catchment area. | `mm/h` |
| `Q_valid` | derived | `1` only when the discharge coverage rule below passes; otherwise `0`. | mask |

For PET conversion, the adapter must record the latent heat value and formula in
its provenance. The initial contract uses `2.45 MJ/kg` and `1 kg/m2 = 1 mm`, so
`PET_mm_per_hour = EVPTRNS_MJ_per_m2_per_day / 2.45 / 24`. A future temperature-
dependent or product-specific conversion requires a schema/version change.

The adapter must validate the exact native and canonical units against
`variables.json`; it must not infer a conversion from a variable name alone.

## Discharge coverage and gaps

The gauge begins at `2024-01-01T01:00:00Z`, and internal gaps are expected.
HydroPINN must preserve both conditions rather than interpolate them.

* Compute duration coverage from the native sample support inside each hour.
* Set `Q_valid=1` when at least 75% of the hour is covered and no provider/QC
  error invalidates the source samples.
* Set `Q_valid=0` and leave `Q_observed`/`Q_runoff` missing otherwise.
* The first hourly bin is therefore expected to be invalid, not zero discharge.
* Do not bridge a gap across an hourly boundary and do not linearly interpolate
  discharge by default.
* Retain the `temporal.study_period_coverage` warning in derived provenance; it
  is not a reason to reject the entire package.

The 75% threshold is part of this versioned consumer contract, not a hidden
implementation constant.

## Forcing validity

Do not impute a missing forcing silently. Emit a validity bit per forcing and an
`all_forcings_valid` bit. A model-ready row requires all configured forcing
features to be valid. Daily PET may be expanded only when the daily source value
is finite and its UTC day is unambiguous.

The canonical intermediate table is:

```text
timestamp,P,T,RH,wind,solar,PET,Q_observed,Q_runoff,
P_valid,T_valid,RH_valid,wind_valid,solar_valid,PET_valid,Q_valid,all_forcings_valid
```

## Model dataset construction

After hourly harmonization, HydroPINN performs these operations in order:

1. select the declared feature set;
2. form contiguous valid segments (never allow a lag or LSTM window to cross an
   invalid forcing row or invalid required target);
3. make chronological train/validation/test partitions without shuffling;
4. fit scalers on training rows only;
5. apply FFN lags or LSTM sequences independently inside each partition; and
6. retain original timestamps and masks for metrics and plots.

A row with `Q_valid=0` may be retained for forcing-only inference, but it must not
participate in supervised loss or observed-flow metrics. Split boundaries must
not be moved merely to hide gaps, and scaler fitting must never include missing
or masked values.

## Required adapter validation

Before returning tensors, the future temporal-package adapter must verify:

* manifest and `variables.json` schema versions and checksums;
* the seven-variable full-data contract, while allowing the documented
  six-variable weather/PET-only inference package with no discharge target;
* exact units, temporal support, UTC ordering, duplicate identity, and finite
  values;
* an hourly grid of the declared study interval, including leap-year length;
* catchment area availability before discharge-depth conversion;
* no fabricated storage, target, or forcing values; and
* deterministic output and provenance containing every aggregation, conversion,
  threshold, and dropped/masked-row count.

## Implementation sequence

The pure C++ hourly harmonizer and its unit/conversion/gap tests are now
implemented in `dataset/gistohq_hourly_harmonizer.{h,cpp}`. It accepts validated
native series, produces mask-bearing hourly rows, and intentionally remains
independent of LibTorch. `dataset/gistohq_temporal_csv.{h,cpp}` now reads strict
wide or long temporal CSV assets, preserves empty values as missing, validates
canonical UTC timestamps, and rejects duplicate variable/timestamp identities
across files. `dataset/gistohq_model_rows.{h,cpp}` selects model-ready rows in a
stable six-feature order and records segment boundaries whenever forcing or
required-target validity is interrupted. `dataset/gistohq_tensor_builder.h`
converts those rows into feature, target, timestamp, validity, and segment tensors
and constructs LSTM windows and FFN lag expansions independently inside each
segment. `dataset/gistohq_package_adapter.{h,cpp}` now validates the seven-variable
full package or six-variable weather-only contract, checks declared native units,
discovers temporal assets, and runs CSV loading, harmonization, and model-row
selection as one operation. Study bounds and catchment area remain explicit
adapter inputs because they must not be guessed from observations. The GUI now
detects `schema_name=HydroPINNExport`, reads those authoritative manifest values,
and routes supervised FFN/LSTM runs through this adapter instead of `DDRRLoader`.
Storage-dependent PINN modes are rejected before dispatch.

HydroPINN requires producer schema `1.2` or newer. In that schema,
`study_start` is the first included UTC hour and `study_end` is the final included
UTC hour; the adapter converts the latter to its internal half-open grid boundary.
Older `1.1` exports are rejected with instructions to regenerate them rather than
inferring bounds from temporal CSV coverage.

1. Add a full manifest fixture copied from the producer schema and finish mapping
   its asset paths, checksums, profile, and QC status into the adapter.
2. Extend the hourly harmonizer tests with the full leap-year producer fixture.
3. Verify unit conversion, discharge gaps, daily PET, and coverage against that
   fixture in addition to the existing focused unit tests.
4. Preserve mask/segment identities through the existing FFN and LSTM split,
   scaling, lag, and sequence paths rather than only filtering invalid rows.
5. Enable water-balance physics only
   after the storage compatibility blocker is resolved.
