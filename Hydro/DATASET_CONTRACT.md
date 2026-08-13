# HydroPINN dataset contract v1

This contract is the boundary between data-acquisition tools such as GIStoOHQ
and HydroPINN. Producers acquire, harmonize, quality-control, and document data;
HydroPINN owns experimental splits, scaling, training, and evaluation.

## Package layout

```text
<dataset_id>/
  manifest.json
  observations.csv
  variables.json
  quality_control.csv
  catchments.geojson
  catchment_attributes.csv
  stations.geojson
  forecast.csv                 # optional
  raw/                         # immutable source responses
  provenance/                  # request parameters and checksums
```

`manifest.json` must declare `schema_name=hydropinn-dataset`, semantic schema
version, dataset/site identifiers, UTC study interval, timestep, file paths, and
SHA-256 checksums. `variables.json` must declare canonical name, unit, role,
source-native name, temporal/spatial support, aggregation, missing-value policy,
provider, product/station identifier, and retrieval time for every variable.

## Observation table

Required columns for the water-balance study are:

| Column | Unit | Meaning |
| --- | --- | --- |
| `timestamp` | UTC ISO-8601 | Canonical `YYYY-MM-DDTHH:MM:SS[.fraction]Z`; strictly increasing within each catchment. |
| `catchment_id` | dimensionless | Stable ID matching the catchment layers. |
| `precipitation` | mm/h | Catchment-average liquid-water equivalent. |
| `potential_et` | mm/h | PET with calculation/product documented. |
| `observed_discharge` | m3/s | Independently observed outlet target. |
| `storage` | mm | Catchment storage or explicitly documented proxy. |

Storage may be optional for non-water-balance profiles, but must never be
silently synthesized for a field-data water-balance experiment. Missing values
are empty/null and accompanied by QC flags; sentinel values such as `-9999` are
forbidden.

Rows may be interleaved across subcatchments. The composite
`(catchment_id, timestamp)` key must be unique, and timestamps must increase
within each catchment independently. Each catchment must contain at least three
observations so chronological train/validation/test partitioning is possible.

## Forecast table

Forecasts use long form with `issue_time`, `valid_time`, `lead_hours`,
`catchment_id`, `variable`, `value`, `unit`, `forecast_model`, `model_cycle`,
and optional `ensemble_member`. Retaining issue time prevents future-information
leakage in retrospective experiments.

## Required GIStoOHQ work

GIStoOHQ should implement `hydropinn-data`, `hydropinn-validate`, and
`hydropinn-report` commands that write and validate this package. Required
adapters are discharge, historical meteorology, PET/ET, archived forecast
meteorology, and available storage/state proxies. Static processing must emit
catchment area, elevation/slope, imperviousness, land-cover and soil fractions,
hydrography, outlet metadata, and stable subcatchment topology.

Before bulk acquisition, GIStoOHQ should create a Hickey Run gauge-reconnaissance
report containing gauge distance, provider/station ID, observed and modeled
drainage areas, area mismatch, record period, resolution, completeness, and
quality status. The runoff target is a go/no-go decision for the study.

GIStoOHQ must preserve raw responses and record provider, product/station ID,
request parameters, retrieval timestamp, native units/resolution, license,
processing operations, and checksums. It must not normalize data or choose
train/validation/test partitions.
