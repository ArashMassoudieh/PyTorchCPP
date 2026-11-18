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

## Causal adapter and prepared-source acceptance (2026-11-18)

The timing blockers above are now fixed in the additive prepared-data path:
`hydrofusion_causal_native_adapter.h` labels completed hourly intervals at their
END and uses only completed PET days released by the specified origin. Hourly
forcing and daily-PET delays are explicit; negative/unknown policies and nonzero
Q-release latency are rejected rather than silently changing the one-hour task.
The prepared-package recipe freezes zero Q delay and24-hour forcing/PET delays
as IDEALIZED HINDCAST assumptions. Actual operational publication latencies are
not established. The original same-hour/legacy loader is deliberately preserved.

`slice_hydrofusion_producer_prefix.py` can prepare the upstream presealed slice
without reading excluded target values: unbuffered exact timestamp read, then
stop before reading the first excluded record's variable/value/qualifiers. It
requires the producer's canonical sorted timestamp-first format; arbitrary
multiline CSV is rejected. Parent whole-asset hashes remain explicitly UNVERIFIED
because checking them would read the mixed-period reserved observations. Exact
copied prefixes and parent manifest are independently hashed and retained.

`prepare_hydrofusion_causal_package.py` checks source QC (only disclosed coverage
warnings accepted), declared-variable units against actual rows, native qualifier
policy, and creates a hash-bound interval CSV plus daily PET/provenance manifest.
It does not fit a model. `loadPreparedNativePackage` rechecks the manifest/CSV
hashes and domain/time policy, then calls the SAME causal adapter for all five
new families. The opt-in config requires explicit native manifest path+SHA and
synthetic verification=false. Conflicting sources, absent pins and legacy raw
package/CSV flags still fail. Managed bundles include the interval input and
native manifest; inference binds the latter's hash to the exported config. The
upstream audit/provenance records are retained with the campaign, not invented
from trained predictions. Original causal synthetic modes remain compatible.

Verified normal optimized frozen build:7 fabricated prepared-source variants
through actual CLI, all7 managed fresh reloads,2 all-row FFN reloads/validation
exports,8 actual CLI rejection cases, all under a hard12GiB virtual-memory cap.
The five original same-hour methods remain byte-identical on1054 regression
rows. Five standalone adapter suites pass ASan/UBSan (leak checking disabled for
host restrictions);15 producer/prefix +3 preparation +5 budget +9 evaluator
regressions pass. This is bounded acceptance, not general memory-safety proof.

User explicitly approved the initial six-fit FFN budget after these fixes.
The research plan uses one prospectively frozen calendar fold: Train2015–2016,
Validation2017, Test2018, endpoint cutoffs17543/26303/35063. All arms share16182/
8595/8387 eligible target timestamps, actual area23754600m², A-only discharge,
no Q interpolation, max300s sample support/75%coverage. The300s cap is a declared
conservative aggregation policy corroborated by presealed native cadence, not
a claim the export independently declares continuous observation support.
Estimated A;e records are excluded. POWER's legacy MJ/hr energy-density alias
follows the existing producer/consumer contract and remains disclosed.

The six-fit launcher freezes data/support/config/script/executable hashes before
any fit, takes an exclusive launch claim, pins two CPUs/noGPU, hard-caps address
space12GiB (stricter than RSS alone), wall time8h/fit and retained artifacts250MB,
checks50epochs and earliest validation-minimum selection, then reloads every
checkpoint in a fresh process. No automatic retry or larger campaign is allowed.
FFN reproduction checks ALL test rows and a bounded production target-free API
sample, and saves validation predictions separately. Test and validation metrics
are explicitly labeled; no test-based hyperparameter selection.

Reproduce fabricated acceptance:

```sh
python3 Hydro/experiments/gistohq_sligo/verify_hydrofusion_native_cli.py BATCH RELOAD FFN_REPRODUCTION NEW_OUTPUT
```

Actual research completion/results are recorded separately in the campaign;
passing these checks alone is not an improvement claim or manuscript promotion.
