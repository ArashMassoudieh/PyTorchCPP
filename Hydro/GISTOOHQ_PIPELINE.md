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

## Separate versioned schemas

Do not combine user intent, mutable inventory, and a frozen release in one
manifest. Define five small schemas:

- **SiteSpec** records watershed identity/geometry, study interval, requested
  products, harmonization targets, and source-selection policies.
- **AssetCatalog** records native and derived assets, their extents, units, CRS,
  provider/source identity, retrievals, content digest, and lineage.
- **PackageManifest** freezes a release using the SiteSpec digest, catalog digest,
  included asset IDs, producer/version, generated time, QC status, available
  export profiles, and whether it is self-contained.
- **QCResult** uses stable rule IDs and `error`, `warning`, or `information`
  severity so consumer acceptance policies remain explicit.
- **ProvenanceActivity** records parent asset IDs, transformation/version,
  parameters, software version, start/end time, and output asset IDs. Derived
  processing always creates a new asset; it never mutates native data.

## Cache and retrieval identity

Keep request identity separate from byte identity:

```text
request_key = hash(provider, endpoint, canonical parameters, product version)
content_digest = sha256(raw response bytes)
```

A retrieval record connects both. This preserves provider revisions returned by
the same request and permits distinct requests to reference identical content.
Use atomic writes, locks, resumable retrieval, offline lookup, and an explicit
garbage-collection policy.

The immutable cache is local infrastructure, not automatically copied into each
portable package. Exports support `--include-raw none|referenced|all` and declare
whether the resulting package is self-contained and redistributable.

## Deterministic source selection

`auto` is a versioned policy object, not "choose nearest." It records constraints
such as maximum drainage-area mismatch, required record overlap, minimum
coverage, accepted status, topology requirements, scoring algorithm/version,
all candidates, scores, rejection reasons, and whether human confirmation is
required. Reconnaissance emits both `report.json` and `report.md`.

## Forecast identity

Forecast assets retain `issue_time`, `valid_time`, `lead_time`, `member`,
`variable`, and location/grid identity. Harmonization must never discard issue
time. A consumer may use a forecast only when `issue_time <= prediction_time`.

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
ohqbuild hydro-data reconnaissance --site sites/hickey_run.yaml
ohqbuild hydro-data fetch --site sites/hickey_run.yaml --offline-ok
ohqbuild hydro-data validate --site sites/hickey_run.yaml
ohqbuild hydro-data report --site sites/hickey_run.yaml
ohqbuild export hydropinn --site sites/hickey_run.yaml \
  --profile water-balance --output outputs/hickey_run_v1
```

The same commands should work for another watershed by changing only the site
specification and source-selection configuration.

Retain the existing `ohqbuild` CLI during migration. A future `gistohq` alias can
be introduced without making command renaming part of the data-architecture
milestone. Add the catalog/object-store layer underneath existing provider code,
keep the current OHQ workflow as a compatibility consumer, and create a separate
`SiteSpec` rather than expanding builder-specific settings indefinitely.

## Implementation order

1. Record architecture decisions for ownership, native/derived assets, cache
   identity, and adapter responsibilities.
2. Define SiteSpec, AssetCatalog, PackageManifest, QCResult, and provenance schemas.
3. Implement the object store, catalog, locking, checksums, and offline reuse.
4. Wrap one existing provider and prove registration/lineage end to end.
5. Implement gauge reconnaissance and deterministic selection policy.
6. Implement observed discharge, then historical meteorology and PET/ET.
7. Implement temporal harmonization, generic QC, and package freezing.
8. Implement the versioned HydroPINN export adapter and validate its output with
   `HydroDatasetValidator` fixtures.
9. Add archived forecasts after forecast dimensions are represented fully.
10. Add further spatial and state/storage products.

GIStoOHQ should not select features, impute across scientific gaps without a
declared rule, normalize values, construct neural-network lags, or choose
train/validation/test partitions.
