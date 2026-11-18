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
