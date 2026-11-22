# Independent rainfall availability for all five forecast families

`HydroFusionPreparedNative-v3` is an opt-in **input-package** schema. It is not
an improvement claim, an operational latency certificate, or a new model family.
Native-v1/v2 retain their existing shared hourly-weather delay and ET semantics.

Native-v3 preserves the v2 reference-ET fields and adds these required fields:

| Field | Meaning |
|---|---|
| `rainfall_delay_hours` | Age of the selected completed rainfall interval at forecast origin, integer 0..168 |
| `rainfall_release_delay_hours` | Assumed delivery delay after that interval ends, integer 0..rainfall_delay_hours |
| `rainfall_product` | `MRMS_RadarOnly_QPE_01H`, `AORC_hourly`, or `fabricated_verification`; no implicit mixture |
| `rainfall_availability_basis` | Exactly `assumed_fixed_hourly_delay` |
| `rainfall_availability_evidence` | Required, nonempty description of the scenario and its evidence/limitations |

MRMS radar-only hourly policy requires at least one hour of declared release
latency on this hourly interface. This prevents assuming an accumulation ending
exactly at issue time was already published then. It does **not** prove that one
hour is a historical upper bound; that must be reviewed upstream. Fractional-hour
or variable delivery schedules need a separate contract, not rounding here.

The selected rain interval ends at `origin - rainfall_delay_hours`. Its release
is that valid end plus `rainfall_release_delay_hours`; release must not exceed
origin. All other hourly weather retains `hourly_forcing_delay_hours`. Daily ET,
Q support, units, normalization area and next-hour target semantics are unchanged.
Missing rain remains missing; a valid zero remains dry. Native-v1/v2 reject these
new fields rather than silently ignoring them.

The shared adapter serves FFN, LSTM, the two-reservoir physical reference, FFN
physical residual and LSTM physical residual. Both nonphysical direct-log and
persistence-log-residual targets remain supported. Timing-aware physical v3
accepts already-released rain, including a release before the origin; numerical
routing still follows its valid-time timeline. Total valid-time age is bounded
before dry-gap iteration. Existing inputs whose release equals origin produce
unchanged results. The older event-time physics recipes are unchanged.

The native manifest hash binds timing/product fields to every managed bundle.
Production reload uses that same native loader. Freshness changes therefore need
a new manifest and cannot be silently patched into an existing archive.

## Verification

`test_hydrofusion_causal_native_adapter.cpp` checks independent rain/weather
selection, identity metadata, valid zero versus missing, future mutations,
invalid age/release policies and exact legacy-equivalent rows.
`test_hydrofusion_valid_time_physics.cpp` checks prior-release numeric parity and
rejects future releases, releases before valid time and excessive total age.

Run `verify_hydrofusion_independent_rain_cli.py NEW_BINARY NEW_RELOAD OLD_BINARY
NEW_OUTPUT` with the GIStoOHQ test environment. It constructs **fabricated**
300-hour packages and exercises seven variants across the five families with
same-hour release and earlier-release policies, plus legacy v2 controls. It
checks fresh-process output reproduction, old-binary row-level prediction parity
for legacy and existing timing-v3 controls, semantic rejection and manifest hash
binding. These small synthetic fits are development tests, not research results.

Before native research: inventory the complete candidate basin/time coverage,
verify product identity/units and explicitly review the release assumption.
Freeze identical eligible timestamps across both arms. If coverage changes,
old baseline scores are not a matched comparison. Preserve existing packages,
frozen outputs and reserved observations. A four-hour spatial pilot is not a
full-period input acceptance, and none of these tests predicts a score gain.

## Independent missingness and matched study support

Native-v3 optionally declares
`interval_mask_schema: independent_rain_and_common_origin_v1`. Its LF CSV adds
`rainfall_valid,common_origin_forcing_available` after `source_id`.
The first mask belongs to the raw rainfall interval, independently of the
legacy met-group validity. The second belongs to the corresponding **output
origin** (raw interval end), after the declared delays. Both accept only0/1.
Legacy packages retain their original fourteen-field schema and behavior.

The origin mask restricts feature/sample eligibility only. It does not rewrite
native discharge flags, zero-fill rainfall, or reset physical state. Physical
routing continues on each package's actual forcing stream. Each sequence must
pass the common origin mask at all24steps, and still satisfy its own native
Q/history, split and forcing checks. A true study mask cannot override a missing
predictor. Managed bundles already hash/copy the entire interval CSV, so both
masks remain bound to the archived input manifest.

`prepare_hydrofusion_radar_comparison.py` preserves all raw non-rain fields,
uses independently audited radar basin coverage>=75%, and compares AORC age24h
against radar-only age1h. The common causal mask intersects availability at
those two ages. This is a **source-and-age input-package comparison**, not an
isolated timing experiment or certified real-time forecast. One-hour radar
publication is an explicit retrospective assumption. Missing hours remain
excluded rather than replaced with dry weather or another product.

`audit_hydrofusion_radar_support.cpp` verifies identical eligible sequence
identities and observations across arms, plus unchanged adapted non-rain inputs.
`verify_hydrofusion_native_masks_cli.py` exercises all five families and both
nonphysical target modes, including managed reload and corruption rejection.
The native-family reproduction checker also recomputes archived input/target
scalers and the optional physical-loss scale from eligible training tensors.

`run_hydrofusion_radar_batch.py` freezes a26-job inventory:24neural fits
(4families ×2packages ×3seeds) and2fixed physical runs. It refuses duplicate
launches, changed inputs, support drift, extra jobs or expanded resource limits.
Each neural fit has50epochs,8hours,2CPUthreads,noGPU,12GiB and250MB ceilings;
checkpoint selection and advancement use validation only. No retries, automatic
expansion, universal improvement claim or manuscript promotion are implied.
