# HydroFusion development adapters

These additive, Torch-free components do not enable package/CSV training. The
existing FFN synthetic-consent and real-source rejection guards remain unchanged.

## Native-support boundary

`adaptNativeHourlyRows` accepts already harmonized rows plus a separate,
row-identified native-support evidence vector. It does **not** read a package
from disk or recover qualifiers discarded by the legacy harmonizer. Its caller
must provide a presealed slice, declared units, actual package area, and an
expected source identity. The epoch is explicitly 2015-01-01 UTC. Every row and
evidence timestamp is checked before discharge/forcing values are accessed;
there is no accept-full-package-then-filter behavior.

Runoff is recomputed from m3/s using the supplied area in m2. Wrong/missing
units, invalid area, unknown source, duplicate/missing identities, nonhourly or
out-of-window timestamps fail closed. Native coverage is distinct from generic
harmonizer coverage. Interpolated, non-native, insufficiently covered or
non-finite discharge is masked unavailable; existing gap/history handling then
excludes affected samples. Conflicting native/interpolated evidence is rejected.
Forcing validity requires all six individual flags, aggregate validity, and
finite physical-domain values.

The source ID is a consistency check, **not authenticated proof**. Real package
qualifier extraction, source hashes, native coverage computation and linkage to
an approved presealed input manifest remain required upstream. Tests fabricate
structs shaped like harmonizer output; they do not constitute a disk-package
loader test or acceptance of any real observation. Caller-supplied flags are not
independently verified evidence merely because this adapter accepts them.

## Causal sequence boundary

`buildForecastSequences` reuses the common twenty-feature builder, masks and
persistence split assignment. One-step eligibility equals the FFN builder.
Longer sequences require complete consecutive eligible origins, and every step
origin plus the final target must lie in one partition. Feature histories may
use earlier causal observations as the existing FFN contract allows. The extra
sequence eligibility restriction must be applied to the FFN comparator too in
any future architecture comparison; existing FFN scores cannot be reused.

Each result retains origin/target identities, split, raw target and feature
steps. No padding or carried hidden state is introduced. Fitting q_ref/scalers
must use this final eligible training support, not blindly reuse scales from
an FFN arm with broader eligibility. Sequence assembly is not a trained LSTM,
new dispatch mode, checkpoint format or demonstrated improvement.

## Verification

```
bash Hydro/experiments/gistohq_sligo/test_hydrofusion_development_adapters.sh
```

Uses fabricated inputs only, normal optimization, warnings-as-errors,
AddressSanitizer and UBSan. Leak detection is disabled because the host prevents
LeakSanitizer operation under ptrace; no leak-clean claim is made.

Next: independently verify qualifier extraction from fabricated disk packages,
then integrate the sequence tensors with an opt-in LSTM model and test both
log-target modes, validation selection, serialization and fresh-process physical
Q reload. No real research training or five-method accuracy claim is authorized
by these checks.

## Development file reader and LSTM verification extension

`loadDevelopmentHourlyPackage` reads an explicit fourteen-column development
CSV with units in its exact header. It is not a GIStoOHQ manifest reader and
cannot establish the truth of caller-supplied qualifiers. No production source
guard is removed. Timestamp tokens for every row are checked against the
requested preseal hourly slice before any observation/forcing token is parsed.
The format has no quoted fields; source IDs containing commas are unsupported.
Full numeric-token consumption, finite values, literal 0/1 flags and exact
column counts are required. Duplicate support identities and unknown source
IDs are rejected by the common native-support adapter.

`prepareHydroFusionSequenceData` assembles [N,24,20] tensors for the synthetic
LSTM check. It recomputes q_ref from sequence-eligible Train targets once each,
then shares exactly the same support and inputs between direct-log-Q and
log-residual modes. The LSTM smoke check uses the existing HydroLSTM module,
Adam, training-only standardization and validation-selected weights. Its
self-contained Torch verification archive contains weights, scaler state,
mode/q_ref, model shape, synthetic input identities and expected physical Q.
This archive is a test fixture, NOT the production HydroArtifactLoader bundle
format. A separate executable process must reproduce physical Q exactly from
that archive; HydroBatch LSTM dispatch and managed-bundle inference remain a
subsequent integration stage.

Build this check from a frozen transitive source snapshot, then run with a
new output directory (never reuse an existing directory):

```
python3 Hydro/experiments/gistohq_sligo/build_hydrofusion_lstm_synthetic.py /tmp/new-hydrofusion-lstm-check
/tmp/new-hydrofusion-lstm-check /tmp/new-hydrofusion-lstm-results
```

The build retains copied headers, compiler command, source hashes, compiler
output and executable hash alongside the binary. It uses normal -O2 and no
optimization workaround. This tiny synthetic fit is development verification,
not a new research campaign or evidence of improved streamflow prediction.

Verified execution for this extension: fabricated CSV → strict file reader →
native-support adapter → sequence assembly → small real HydroLSTM fit →
validation-selected weights → verification archive → separate-process physical-Q
reload. Both target modes reproduced 36 test predictions exactly. Both selected
epoch 12 of 12; this run does not demonstrate selection of an earlier epoch.
Changing only validation/test discharge left Train tensors, q_ref and fitted
scaler offsets unchanged. Input discharge features use float32; original
origin discharge remains double precision for reconstruction. Identity checking
compares the feature conversion at float32 precision rather than incorrectly
requiring its double representation to equal the unrounded original.

File-reader tests passed conversion and interpolated-support masking plus nine
malformed/schema/provenance/timestamp rejection cases, including a fabricated
seal timestamp rejected before parsing its deliberately malformed observation.
The standalone adapter suites passed with ASan/UBSan (leak detection disabled);
the LibTorch LSTM check passed normal -O2 with warnings-as-errors. The latter is
not an ASan/UBSan or production managed-bundle acceptance claim.
