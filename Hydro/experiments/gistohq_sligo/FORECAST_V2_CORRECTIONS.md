# Prospective forecast corrections (18 November 2026)

These options address verified defects/limitations from
`FORECAST_ERROR_DIAGNOSIS_2018.md`. They do not establish improved research
scores. No new real-data fitting or manuscript promotion is authorized here.
Existing outputs and default checkpoint recipes remain unchanged. These options
apply to the five additive causal-forecast families, not an automatic rewrite
of every historical FFN/LSTM/PINN run or a newly verified interactive UI control.

## Hourly physical reference

`physics_profile: causal_two_reservoir_mean_v2` opts into exact interval-mean
outflow for the physical-only and two physical-residual forecast families.
For a reservoir with constant hourly inflow r, initial storage S, recession k,
the integrated outflow is S(1-exp(-k)) + r[1-(1-exp(-k))/k]. The step is exactly
one hour; integrated depth divided by one hour matches the native hourly mean
runoff target. A small-k series prevents cancellation in the second term.
State propagation, forcing availability, causal origin, gap resets and initial
zero storage retain the original rules. The old endpoint-returning
`causal_two_reservoir_v1` functions are preserved for old bundle reproduction.
The profile choice is checkpoint-recipe bound, not an inference-time switch.

This is not a new runoff-generation model. Gain=1 still routes all
max(P-PET,0); runoff_coeff remains a fast/slow partition. There is no soil
infiltration/storage loss, no Q assimilation and no guarantee that the learned
hybrid closes a water budget. Averaging alone does not cure the measured
volume excess. Do not calibrate gain against test outcomes.

## Neural objective and FFN activation

`hydrofusion_objective: log_flow_mse_v2` opts into a fixed dimensionless mix:

    L = 0.5 MSE(standardized log target)
      + 0.5 mean(((reconstructed physical Q - observed Q) / sigma_train_Q)^2)

The physical reconstruction uses the same direct/residual reference, q_ref,
expm1 and final nonnegative clip as inference. It is differentiable and
computed in float64, with overflow rejection. sigma_train_Q is the population
standard deviation of eligible, unique, independently verified TRAIN targets
only (constant-series fallback 1 below 1e-12). It is frozen for validation and
saved as `training_flow_scale` in the checkpoint. Validation checkpoint
selection uses the same composite loss, never test performance. Physical MSE
includes squared mean bias; there is no additional fitted bias multiplier,
post-hoc rescaling or coefficient sweep. Equal weights are a prospective
engineering choice requiring validation, not an empirically established optimum.

`hydrofusion_ffn_activation: silu` opts into an unbounded FFN hidden activation
for direct/persistence-residual FFN and physical-residual FFN. This removes
the finite tanh-hidden-layer amplitude bound; it does not ensure accurate
peaks. LSTM and the physical-only family reject this FFN option. The existing
`activation: tanh` field remains the old forecast config requirement; the new
explicit field controls this opt-in FFN change only.

Default objective `log_mse_v1` and FFN activation `tanh` preserve the original
computation. New recipes use checkpoint envelope `hydrofusion-forecast-v2`
and a 16-value recipe (version, original parameters, profile/objective/head
flags); old v1 uses its original 13 values. Managed reload requires config,
format, recipe, model/scaler hashes and sidecar consistency. No silent upgrade.
New options are rejected for unrelated legacy engine approaches.

## Catchment alignment blocks further real-data fitting

The actual producer SiteSpec declares 23,754,600 m² and outlet
(38.97837618066522, -76.98873786162696). Metadata-only USGS Site Service for
station 01650800 lists drainage area 6.45 square miles = 16,705,423.3116672 m²
and gauge coordinates (38.9862222, -77.0048611). The package is 42.1969% larger
and its outlet is 1,644.19 m from the gauge. This does not identify the correct
basin polygon by itself, but it is concrete evidence against assuming exact
catchment/gauge alignment. The point meteorology was also requested at the
producer outlet. Existing conversion arithmetic was correct for its declared
area; that is different from hydrological representativeness.

Source: https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=01650800&siteOutput=expanded&siteStatus=all

Saved metadata, hashes and computed comparison are in Dropbox
`results/hydrofusion_error_diagnosis_2018/gauge_alignment_audit.json` and
`usgs_01650800_site_metadata.rdb`. `audit_hydrofusion_gauge_alignment.py`
reproduces the comparison without opening discharge assets. Its 1% area/100m
point differences are declared screening thresholds, not basin validation.
Even passing metadata comparisons never grants training readiness.

Next prerequisite: delineate/verify the upstream basin at the gauge; reconcile
area with USGS, confirm forcing/PET spatial coverage and temporal availability;
prepare a NEW, bounded presealed input package with provenance and support
checks. Do not silently replace the old area, relabel historical scores,
rescale existing outputs or inspect reserved observations. A corrected-input
research comparison needs an explicit new budget and a fresh prospective
matched design; the completed campaigns are not repeated automatically.

## Reproduction

Build HydroBatch with `build_hydrofusion_frozen.py` into a new private directory.
Link test mains/reload checker with `link_hydrofusion_check_from_batch_build.py`
against that exact frozen build. Run the torch-free mean test with ASan/UBSan,
the tensor objective test, and `verify_hydrofusion_forecast_v2.py` for seven
fabricated three-epoch configs plus v1 baseline comparisons. Use the saved
legacy comparison runner for the five original engine methods. Synthetic
fits are development tests, not new research experiments or evidence of skill.

A schema-checked **draft**, not a replacement package, is saved beside the audit:
`gauge_aligned_presealed_DRAFT_site_spec.yaml` (end2018-12-31 only) with its
`DRAFT_README`. GIStoOHQ requires explicit pipeline station_id01650800.
No acquisition/delineation was executed. Source/metadata matching and spatial
basin validation must precede any use of that draft.

## Verification receipt

Final private HydroBatch build and supplemental inference objects used the same
frozen headers/defines; all eleven modified production files were rehashed
against the compiled snapshot. The final build passes:

- 51 saved CLI/recipe checks: seven v2 variants, fresh-process reloads,
  seven v1 byte-exact baseline comparisons, option round-trips, semantic
  config/profile tampering and invalid-option rejection. Only fabricated
  three-epoch development jobs ran.
- Five original engine methods: 1,054 raw prediction rows byte-identical to
  the previously accepted baseline executable on the exact synthetic configs.
- Analytic hourly-mean/storage/causality tests under ASan+UBSan; tensor loss
  formula (direct and residual), finite gradients, overflow/domain, Train-only
  scale identities including extreme timestamps, and default/recipe tests.
- All 19 existing research bundles reload their managed archive/config/scaler/
  recipe successfully. This check does not repeat the previously completed
  native all-row inference diagnostic.
- 61 relevant Python regressions, including four metadata-only audit tests.

Saved evidence and executable/source identities:
`Dropbox/HydroPINN paper/audits/hydrofusion_forecast_v2_corrections_20261118`.
Baseline binary identity is pinned by its full source inventory and binary
hash, not merely its recorded HEAD (which does not identify uncommitted inputs).
The provisional runner initially rejected an output outside the checkout;
that harness path requirement was corrected before final verification. The
provisional build also preceded the final raw-timestamp overflow guard and is
not the accepted final executable. No optimized-build failure was encountered
in these isolated builds; this is not a claim that generic memory-safety
issues in unrelated code are impossible.

## Gauge review package prepared (18 November 2026)

`prepare_gauge_alignment_review.py` creates a separate non-runnable package
from the already verified2015–2018 native inputs and the saved upstream GIS
candidate. Raw discharge remains in m³/s, CSV bytes unchanged; no existing
observations, model outputs or normalization were modified. Original manifest
is preserved alongside the pending area/provenance. Timestamp/support checks
precede any interpretation of CSV values; reserved/end-crossing inputs fail.

Actual producer catalog confirms rainfall and PET are `provider_point` products
requested at the downstream outlet, outside the new upstream polygon. This
alone does not prove the coarse provider grid differs, but equivalent grid
support and upstream basin representativeness remain unverified. The products
must not be relabeled as basin averages or silently treated as gauge-aligned.

Saved package: `Dropbox/HydroPINN paper/results/hydrofusion_gauge_aligned_2018_review`.
Its schema `HydroFusionPreparedNative-PENDING-SPATIAL-REVIEW` is intentionally
unsupported by the real loader. Actual CLI rejection was verified before
fitting. Four new regressions cover strict/complete identities, sealed-source
rejection before CSV access, and output preservation. Re-running preparation
reproduced the saved files byte-identically. No new data acquisition or real
research fitting occurred. Next: establish upstream forcing/PET support, then
review/release an explicitly derived package before requesting a new fit budget.
