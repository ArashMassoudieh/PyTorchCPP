# Generic hydro dataset contract v1

This reusable contract is the boundary between data-acquisition tools such as
GIStoOHQ and downstream hydrology applications. Producers acquire, cache,
harmonize, quality-control, and document data. Consumers such as HydroPINN own
their experimental splits, scaling, training, and evaluation.

## Package layout

```text
<dataset_id>/
  manifest.json
  site_spec.json
  observations.csv
  variables.json
  quality_control.csv
  catchments.geojson
  catchment_attributes.csv
  stations.geojson
  forecast.csv                 # optional
  raw/                         # optional; none, referenced, or self-contained
  provenance/                  # request parameters and checksums
```

`manifest.json` must declare `schema_name=hydro-observations`, semantic schema
version, a consumer profile, dataset/watershed identifiers, UTC study interval, timestep, file paths, and
SHA-256 checksums. `variables.json` must declare canonical name, unit, role,
source-native name, temporal/spatial support, aggregation, missing-value policy,
provider, product/station identifier, and retrieval time for every variable.
The site specification, mutable asset catalog, and frozen package manifest are
separate documents. The manifest records their digests, included asset IDs,
producer version, package QC status, and raw-content inclusion/self-contained
status.

The current HydroPINN loader requires the string fields `schema_name`,
`schema_version`, `profile`, `dataset_id`, `observations_file`, and
`catchment_attributes_file`; `quality_control_file` and `variables_file` are
optional for backward compatibility. When `variables_file` is declared, every
required profile variable must have one metadata record and its unit must match
the canonical unit in the selected contract. HydroPINN rejects missing,
duplicate, or unsupported unit declarations rather than silently interpreting
them. Paths must be relative and remain inside the package. Package loading
rejects incompatible schema major versions, profile mismatches, and any QC
record with `severity=error`.

Producers may provide `observations_sha256`, `catchment_attributes_sha256`, and
`variables_sha256`. When declared, HydroPINN calculates SHA-256 from the exact
file bytes and rejects mismatches before parsing the corresponding asset.

## Observation table

The reusable `rainfall-runoff` profile requires timestamp, catchment ID,
precipitation, PET, and observed discharge. The `water-balance` profile extends
it by requiring storage:

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

## GIStoOHQ architecture

GIStoOHQ should remain a generic acquisition and catchment-data system, not a
HydroPINN-specific downloader. Its core commands should be `hydro-data fetch`,
`hydro-data validate`, and `hydro-data report`. A thin `export hydropinn` adapter
should map the generic catalog into this contract without embedding model,
normalization, lag, or split decisions in GIStoOHQ.

Provider adapters should cover discharge, historical meteorology, PET/ET,
archived forecast meteorology, and available storage/state proxies. Static
processing should emit catchment area, elevation/slope, imperviousness,
land-cover and soil fractions, hydrography, outlet metadata, and stable
subcatchment topology. Other consumers can export the same cached assets to
their own schemas.

Before bulk acquisition for any watershed, GIStoOHQ should create a gauge-reconnaissance
report containing gauge distance, provider/station ID, observed and modeled
drainage areas, area mismatch, record period, resolution, completeness, and
quality status. The runoff target is a go/no-go decision for each study; Hickey
Run is the first configured site, not a hard-coded special case.

GIStoOHQ must preserve raw responses and record provider, product/station ID,
request parameters, retrieval timestamp, native units/resolution, license,
processing operations, and checksums. It must not normalize data or choose
train/validation/test partitions.
