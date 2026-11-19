# Forcing review before another HydroPINN comparison

## Established from producer metadata and actual consumer source

The historical package requests NASA POWER hourly meteorology and daily
EVPTRNS at the intermediate downstream outlet. Both are provider-point
products. Being outside the gauge polygon does not prove different provider
cells or invalid values; cell identities and basin representativeness remain
unresolved. The copied OHQ reference topology supplies a gauge domain of
16.7375 km² inside the full 29.1837 km² basin. The independently listed USGS
area and the alternative GIS candidate remain separate provenance choices.
Full-domain area must not normalize internal-gauge discharge.

NASA's hourly API explicitly supplies precipitation in mm/hour. The existing
producer's narrow hourly PRECTOTCORR metadata correction matches this contract;
there is no demonstrated missing factor of 24 in this path.

EVPTRNS is evapotranspiration energy flux. Conversion by 2.45 MJ/kg and 24 hours
produces equivalent ET depth in mm/hour, as the consumer implements. This is
an energy/unit conversion, not a potential/reference ET estimation algorithm.
Current catalog metadata does not establish potential ET. Do not silently call
it validated PET, replace old archive values, or claim a quantified contribution
to prediction error without a new prospective comparison.

NASA currently states typical meteorology latency of 2–3 days and solar latency
of 5–7 days. These are not historical delivery logs. They do not establish
per-record availability, but they do not support treating the exploratory
24-hour forcing assumption as a demonstrated operational schedule. Keep the
existing experiment labeled idealized retrospective causal hindcast.

## Next input-release proposal

1. Confirm urban drainage/effective-area versus topographic-gauge domain and
   document the chosen normalization geometry/area explicitly.
2. Verify exact provider grid support from retained metadata if possible. If
   unavailable, propose a bounded 2015–2018 gauge-basin forcing acquisition;
   do not download or fit as part of this audit.
3. Use an explicitly documented potential/reference ET product or a specified
   reference-ET calculation with every required variable, unit, surface
   assumption, time aggregation and availability rule. Existing converted
   EVPTRNS can remain a clearly labeled historical ET-proxy baseline.
4. Freeze matched native support, chronological splits and all preprocessing
   choices before running comparisons. Separate input correction from objective/
   architecture changes; do not attribute a combined improvement to one cause.
5. Release a new source/hash-bound package only after review. The existing
   PENDING-SPATIAL-REVIEW package stays non-runnable. Request a fresh research
   budget afterward; prior authorized fits are complete.

The fixed-gain two-store model's missing runoff-generation/evaporative-storage
loss processes remain a separate issue. Correct ET semantics and gauge area
alone do not establish a correct physical model or guarantee better scores.

## Evidence and reproduction

Run `audit_hydrofusion_forcing_semantics.py CATALOG NEW_OUTPUT_DIRECTORY`.
It reads catalog metadata only, preserves input bytes and writes a review
report. It is not integrated as a production loader gate and never authorizes
training. The actual catalog report and saved script are under Dropbox
`HydroPINN paper/audits/hydrofusion_forcing_semantics_20261118/`.
Do not access the separate OHQ Watershed Models folder; use preserved copies.

Primary sources, checked 18 November 2026:

- https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- https://power.larc.nasa.gov/docs/faqs/data/
- https://gis.earthdata.nasa.gov/portal/rest/services/POWER/POWER_MONTHLY_METEOROLOGY_LST/ImageServer/multiDimensionalInfo
  (EVPTRNS definition; monthly service used only for parameter definition,
  not for the daily API's units, which come from the archived catalog.)
- https://www.fao.org/4/X0490E/x0490e04.htm
