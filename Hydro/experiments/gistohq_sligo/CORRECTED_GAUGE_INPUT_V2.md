# Corrected gauge inputs: versioned producer/consumer handoff

GIStoOHQ now owns the opt-in `ohqbuild data basin-forcing` command. It acquires
AORC hour-ending rainfall and gridMET daily short-grass reference ET over an
explicit polygon, retaining raw responses, URLs, hashes, exact intersection
weights and complete-cell masks. It does not download discharge, normalize
model targets, choose splits or train models.

HydroPINN's `prepare_corrected_gauge_package.py` verifies those exported files
and joins them to an independently bounded native-v1 gauge package. The command
requires the original producer catalog, the hash-bound original export manifest
and USGS station metadata. The declared discharge parent must identify the same
station as the forcing SiteSpec; area/outlet screening must pass. This metadata
chain is not an independent rating-curve or urban-drainage validation.

```
python3 Hydro/experiments/gistohq_sligo/prepare_corrected_gauge_package.py \
  ORIGINAL_NATIVE_MANIFEST NEW_GISTOHQ_FORCING_DIRECTORY NEW_NATIVE_DIRECTORY \
  --producer-catalog ORIGINAL_CATALOG \
  --original-export-manifest ORIGINAL_PRESEALED_EXPORT_MANIFEST \
  --station-metadata USGS_SITE_METADATA_RDB
```

Native discharge stays in m³/s. Only normalization area, rainfall, ET semantics,
source identity and the combined forcing-validity mask change. Auxiliary POWER
weather values stay unchanged and explicitly remain coarse provider-point
covariates, not basin averages. Discharge/native-support fields remain unchanged.
No missing values are imputed. Old producer audit fields are retained as
historical metadata rather than presented as a QC audit of the new rainfall.

## Native-v2 time and unit contract

`HydroFusionPreparedNative-v1` preserves its UTC-midnight daily ET-energy recipe.
`HydroFusionPreparedNative-v2` instead declares:

- `pet_input_semantics: short_grass_reference_et`
- `daily_pet_utc_start_hour: 7` for gridMET's nominal MST provider day
- daily `reference_et_mm_day` records, each with its explicit start epoch and mask

The shared loader converts daily reference depth to its internal energy-equivalent
representation solely to reuse the existing depth conversion. It does not infer
potential ET from EVPTRNS. The causal adapter selects the most recent completed
provider day whose end plus declared delay is at or before the origin. Thus a
07UTC-start day cannot appear at midnight or before its 07UTC completion.
For this idealized hindcast the inherited 24-hour forcing/ET delays remain
experimental assumptions, not an operational availability guarantee. Native-v1
with v2-only semantic fields is rejected rather than silently ignoring them.

The common loader makes this contract available to FFN, LSTM, physical reference
and both physical-residual families. Model fitting/skill remains a separate
question. The research input package itself grants no training permission.

## Real Sligo preparation

The chosen reference topographic gauge basin is SC15's upstream union,
16.7375 km², nested in the 29.1837 km² full watershed. USGS lists16.70542 km²;
the difference passes the explicitly documented1% metadata screen. Urban
storm-drain equivalence remains unverified. Older MRMS basin averages used a
different19.08849 km² NLDI polygon and were not relabeled/reused.

Real2015–2018 preparation retains35,064 hourly gauge intervals and acquires
AORC rainfall for every interval (including the next-year boundary accumulation).
There are1,460 complete gridMET provider days inside this bounded period;
missing early availability is explicit. Data, raw replies, hashes, scripts and
all verification evidence are saved under HydroPINN paper Dropbox. The separate
OHQ Watershed Models folder is off-limits.

## Engineering verification and limits

Use `verify_prepared_gauge_package.cpp` for read-only loading through the actual
shared loader. It can export all loaded rows to a new CSV to compare native-v1
against an old source build. `verify_hydrofusion_reference_et_cli.py` runs only
fabricated300-hour inputs through all forecast-family CLI/export/reload paths,
with semantic corruption rejection. These are synthetic development fits,
not the six proposed real research fits.

Build scripts quote source/generator paths containing spaces. Live source
snapshots must also avoid case-insensitive sync renames: Dropbox can rename
files differing only by case. Build in a private temporary directory and retain
source snapshots as case-safe archives in Dropbox with binary/source hashes.
The frozen-build helper now rejects missing/changed source snapshots after
compilation. Failed attempts remain preserved and are not acceptance evidence.

Next research proposal: six paired FFN fits (direct-log-Q vs persistence-residual,
seeds42/43/44), same50-epoch/32-tanh architecture/log-MSE objective and chronological
cutoffs as the completed first screen, on the corrected inputs. Do not add the
new activation/objective simultaneously if interpreting the input change. Use
validation-only advancement decisions. Any new loss/architecture experiment
and extension to remaining families need a separately bounded stage. No
research fits launch until the human approves the new concrete budget.
