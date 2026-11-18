# Producer evidence audit — real-data integration remains gated

The five causal methods are implemented and synthetically verified. Their
real-package/CSV training guards remain enabled. No new research fits or actual
observation files were read for this work.

## Source findings

GIStoOHQ `ohqbuilder/watershed_data/temporal.py` writes exactly
`timestamp_utc,variable,value,native_unit,provider_qualifiers`, preserving blank
missing values and USGS qualifier sets. Its `hydropinn.py` exporter writes a
version 1.2 `water-balance-v1` manifest with asset hashes, area and study bounds.
The HydroPINN legacy CSV reader discards qualifier columns. Its hourly harmonizer
can interpolate short discharge gaps by default. Consequently, legacy
`discharge_valid` alone cannot establish the native support needed here.

The export does not declare native discharge sample support in its asset list.
Do not infer that support from neighboring timestamps or assume 300 seconds.
An independently justified, hash-frozen policy must supply it per asset.
Qualifier approval is neither a rating-curve uncertainty bound nor proof of
real-time data availability.

## Saved audit

`audit_hydrofusion_producer_package.py PACKAGE POLICY OUTPUT.json` inspects only
an independently prepared, wholly presealed export. The manifest must pass
inclusive-end bounds before any observation asset opens, even for hashing.
Manifest and asset hashes, canonical UTC, duplicates, path traversal/symlinks,
full numeric conversion, finite/domain checks and discharge units are checked.
Every row timestamp passes before numeric values are converted. Unknown
qualifier sets and missing samples contribute no native coverage; there is no
imputation. Each record terminates previous sample support, even when rejected.
Coverage is duration weighted and the accepted discharge mean divides by actual
covered seconds, matching the existing harmonizer's convention with interpolation
disabled. Output contains explicitly labeled half-open hourly intervals, not
forecast origins or an eligible model dataset. Partial-coverage averages are
not equivalent to full-hour measurements.

Policy keys (all mandatory): `manifest_sha256`, `source_package_id`,
`qualifier_policy_evidence`, `accepted_discharge_qualifier_sets` (exact CSV strings,
not substring matching), `minimum_native_hour_coverage`, and
`discharge_sample_support_seconds` (asset ID to integer seconds, 1–3600).
An explicit empty qualifier string may be accepted only if independently justified.
The policy hash binds its literal evidence explanation and all settings; it does
not independently authenticate them. Output creation is exclusive, preventing
accidental overwrite. Producer JSON/CSV inputs are never changed.

Run the 12 fabricated-data regression tests:

```sh
python3 -m unittest discover -s Hydro/experiments/gistohq_sligo -p test_hydrofusion_producer_audit.py -v
```

## Remaining acceptance gates, in order

1. Confirm producer timestamp semantics and forecast availability. A discharge
   average over [h,h+1) cannot be supplied at origin h as if already observed.
   Freeze interval-end labeling and latency before feature assembly. Similarly,
   the legacy harmonizer distributes a daily EVPTRNS energy value over all 24
   hours: same-day finalized totals may be unavailable at early origins. Use an
   explicitly causal lag/available product; do not silently assume availability.
2. Accept units/cadence/availability for every forcing, variables.json consistency,
   source QC and policies, qualifier rules, actual area and provenance. This
   audit inventories forcing values/units but does not harmonize or accept them.
3. Only then implement a shared real-source row loader feeding all five methods,
   freeze identical 24-origin eligible support and prospective fold boundaries,
   and verify managed export/fresh reload from a fabricated producer-format
   package through that loader. Preserve same-hour legacy behavior.
4. Present the smallest six-fit FFN research request with resolved inputs and
   measured feasibility. Explicit budget approval remains mandatory.

The audit always reports `research_ready=false` and `training_authorized=false`.
Passing it is a bounded evidence subdeliverable, not all-five real-data acceptance,
proof of accuracy improvement, or authorization to remove production guards.
