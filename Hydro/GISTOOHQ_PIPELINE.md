# Generic GIStoOHQ pipeline and HydroPINN export

## Design principle

GIStoOHQ should maintain a provider-neutral watershed catalog. API clients,
download caching, provenance, spatial processing, temporal harmonization, and QC
belong in its reusable core. HydroPINN is one export target, alongside GIS,
SWMM/OHQ, tabular analysis, and future modeling tools.

## Pipeline

1. **Site specification** — accept a watershed ID plus outlet, supplied polygon,
   or authoritative basin identifier; never branch on a site name.
2. **Reconnaissance** — discover gauges, stations, grids, forecast archives, and
   product coverage before bulk requests. Produce compatibility scores and a
   human-reviewable go/no-go report.
3. **Fetch** — provider adapters write immutable content-addressed raw responses.
   Cache keys include provider, endpoint/product, normalized request parameters,
   and product version.
4. **Catalog** — register every asset with provider, source ID, request time,
   native CRS/resolution/units, temporal coverage, license, checksum, and status.
5. **Spatial processing** — delineate or ingest catchments and derive generic
   terrain, hydrography, land-cover, imperviousness, soils, and infrastructure
   attributes with stable IDs.
6. **Temporal harmonization** — retain native series, then create declared UTC
   products using explicit aggregation, gap, timezone, and unit-conversion rules.
7. **QC** — retain provider flags and add transparent range, chronology,
   completeness, duplicate, drainage-area, and cross-source checks.
8. **Generic package** — materialize observations, forecasts, spatial assets,
   metadata, provenance, QC, and raw references without ML-specific decisions.
9. **Consumer adapters** — map the catalog to `export hydropinn`, GIS packages,
   or other formats. Adapters must be deterministic and versioned.

## Suggested modules

```text
providers/       # USGS, NOAA/NCEI, forecast archive, ET, soil moisture, etc.
cache/           # content-addressed raw storage and retry/offline behavior
catalog/         # provider-neutral asset and variable metadata
watersheds/      # outlet, delineation, topology, gauge compatibility
harmonize/       # UTC alignment, aggregation, units, gaps
qc/              # reusable validation rules and reports
exports/
  hydropinn/     # thin adapter to hydro-observations profiles
cli/             # fetch, validate, report, export
```

## CLI sketch

```bash
gistohq hydro-data reconnaissance --site sites/hickey_run.yaml
gistohq hydro-data fetch --site sites/hickey_run.yaml --offline-ok
gistohq hydro-data validate --site sites/hickey_run.yaml
gistohq hydro-data report --site sites/hickey_run.yaml
gistohq export hydropinn --site sites/hickey_run.yaml \
  --profile water-balance --output outputs/hickey_run_v1
```

The same commands should work for another watershed by changing only the site
specification and source-selection configuration.

## Implementation order

1. Define generic asset/catalog and site-spec schemas.
2. Implement cache, provenance, checksums, and offline reuse.
3. Implement gauge reconnaissance and drainage-area compatibility.
4. Implement discharge and historical meteorology adapters.
5. Add PET/ET, forecast archive, and state/storage adapters.
6. Implement temporal harmonization and generic QC.
7. Add static catchment processing and topology.
8. Implement the generic package writer.
9. Implement the versioned HydroPINN export adapter and validate its output with
   `HydroDatasetValidator` fixtures.

GIStoOHQ should not select features, impute across scientific gaps without a
declared rule, normalize values, construct neural-network lags, or choose
train/validation/test partitions.
