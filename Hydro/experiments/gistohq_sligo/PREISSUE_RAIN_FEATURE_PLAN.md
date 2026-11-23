Pre-issue rain-rate development contract

Status: prototype validation and archive inventory only. Not wired into training,
not an accepted native input, and no model improvement established.

The existing hourly rainfall accumulation and v3 committed physical state stay
unchanged. The proposed additional channel is instantaneous MRMS PrecipRate
(mm/h), valid exactly ten minutes before the hourly forecast origin. Its archive
HTTP Last-Modified proxy must be no later than origin. This proxy does not prove
historical operational end-user delivery. Actual rate values must be finite and
nonnegative, with at least75% positive-weight basin coverage. Invalid geometry,
missing files, late release and inadequate coverage produce an unavailable
sample, never an invented dry value or nearest-time substitution.

The one-hour forecast target remains origin+1hour. Timestamp/scope checks precede
weather value access. The prototype selector accepts development targets only
through x26303 and rejects the reserved period. Absolute UTC seconds prevent
fractional-hour information from being silently rounded into old integer-hour
release settings. Full inventories inspect weather metadata only, not Q.

Native integration must use a NEW versioned input/recipe and bind the rain
sidecar hash, geometry/weight hashes, valid/release timestamps, units, minimum
coverage and availability assumption in exported bundles. All fitted input
scalers remain training-only. Fresh-process reload must reconstruct features
from the same timestamped sidecar and reject swapped/late/future metadata.
Old v1/v2/v3/v4 archives and predictions must remain unchanged.

Every learned family and its matched control receives the SAME additional rain
channel and identical eligible timestamp support. A physical next-hour rainfall
assumption may hold the available instantaneous rate constant over the forecast
hour, but must be labelled a rainfall-persistence forecast, not observed future
rain. It may affect only the uncommitted forecast copy of the routing state;
it must never rewrite historical committed stores or guarantee corrected-output
water-budget closure. Separate controls are needed before attributing any gain
to physical context rather than information availability.

Before a research comparison: complete hourly archive availability and actual
basin-value/geometry inventory; preserve missingness; freeze shared support and
training-only scaling; test impulse alignment, release leakage, missingness,
state-copy isolation, native CLI export/reload and legacy byte parity. Freeze a
finite immutable comparison plan before fits. Use validation for advancement,
report NSE and squared Pearson R2 separately, and preserve failed hypotheses.
The existing diagnostic future-prediction shift is NONDEPLOYABLE and must never
be used as a model result, calibration input, or method of meeting score targets.
