# Independent Task4 close-out: synthetic HydroFusion CLI

Accepted scope: opt-in **synthetic-only FFN** fitting, checkpoint/scale export,
metadata-aware physical discharge inference, and repeatable CLI verification.
This is not a Sligo Creek accuracy result, a native-data loader, an implementation
for all five methods, or interactive GUI acceptance. No real-basin research
training, reserved observations, manuscript edits or push occurred.

## Historical crash diagnosis

The preserved failing ELF's `runIntegration` allocated its return object at
`rbp-0x15f0` and accessed `models` at `rbp-0x11b0` (offset 1088). Its compiled
loader initialized that map at offset 1120. The passing ELF's caller and loader
both use 1120. The pre-change source independently gives map offset 1088.
A debugger run of the preserved failing executable reproduced an invalid free
inside exporter map destruction. Thus incompatible aggregate layouts within the
failing executable are established; an optimization-only explanation is not.
The exact live-edit/stale-object build event was not independently reconstructed.
Build timestamp strings explain expected hash variation, not this diagnosis.

The correction is consistent, hash-verified copied build inputs and explicit
caller-size checks before exporter/loader aggregate access. No `-O0` or
`noinline` workaround is shipped. Final qmake commands end with `-O2`; the
standalone integration build also uses normal `-O2`. Generated wrappers run only
inside their isolated source tree. Optional cached objects require unchanged
headers/project/generator inputs and identical compile/link flags.

## Additional implementation corrections

- Validation selection now passes validation tensors explicitly to `train()`;
  merely filling its Test slot did not select a validation checkpoint.
- The configured seed is applied before model construction. The target mode
  and explicit synthetic consent are preserved in exported config.
- Explicit source/cutoff guards reject real package/CSV requests, absent
  synthetic consent, unknown modes, and missing/fractional/overflowing indices.
  Exact integral floats remain supported and absent legacy cutoffs retain -1.
- `HydroFusionInferenceSession` validates metadata/config/model/scalers and
  reconstructs physical runoff. Generic inference rejects the new transformed
  target by default. Physical feature width/domain/finiteness checks are tested.
- Metadata and origin/target map are now **inside** the managed artifact manifest.
  Completeness forbids unlisted files, not these files as a category. This fixes
  the earlier external-sidecar workaround. Both are protected against accidental
  modification; model/scaler hashes and semantic mode/bound checks still apply.
- Prediction `x` is target time, with an explicit one-hour origin/target map.
  Strict metadata parsing, finite-aware comparisons and overflow-safe identity
  checks reject corrupted fixtures rather than producing false parity passes.

## Verification

Saved independent evidence and source/binary hashes:
`/home/arash/Dropbox/HydroPINN paper/audits/codex_task4_takeover_20261118T210553Z/`
(`final_verification.json`, ABI report/disassembly/debugger log, frozen sources,
build logs/manifests, executables, synthetic configs/artifacts and test results).

- 10 real CLI groups pass: both target modes; package/CSV rejection;
  absent consent; unknown mode; missing/fractional/overflowing cutoff rejection;
  and the five original methods' synthetic jobs.
- Both CLI bundles independently reload in fresh processes: 59 predictions
  each, maximum measured absolute discharge difference **0**.
- Tampered metadata/map fail bundle integrity checks. A manifest-valid changed
  mode, swapped scaler, and NaN prediction are separately rejected.
- Same-seed predictions reproduce byte-for-byte; a different seed changes this
  nondegenerate fixture. Validation-selected epochs are 9 and 7 out of 30,
  matching the saved validation minima rather than the final epoch.
- The original five configurations' prediction files match retained e4c5409
  baseline outputs byte-for-byte: FFN/PINN/FFN-hybrid 220 rows each;
  LSTM/LSTM-hybrid 197 each. **Limitation:** the historical baseline executable
  hash was not retained. This is a retained-output regression, not a newly
  reconstructed source-to-binary provenance proof or universal behavior claim.
- Optimized integration tests pass including fresh children, metadata negative
  cases, ABI-size rejection, and a NaN expected-output regression. Legacy
  inference and exporter tests exit 0 against the consistent build objects.
- 19 log-target tests, 25 feature-builder tests, metadata roundtrip plus 20
  invalid cases pass. Eleven Python runner/config-loader test methods pass;
  loader checks run with UB and float-cast overflow sanitizers.

Existing third-party/Utilities warnings remain. No claim of general memory-safety
proof, complete profile coverage or operational forecasting validation is made.
The production C++/headers/project match the frozen final build; the updated
integration-test main is a separately hashed copy compiled against the same frozen
production headers/objects. A new real-data experiment still needs its own budget.

## Repeatable entry points

See `Hydro/models/HYDROFUSION_PROTOTYPE.md` and the fixture directory README.
Use NEW binary/output paths; runners never delete existing results.

```
python3 Hydro/experiments/gistohq_sligo/build_hydrofusion_frozen.py HydroBatch /tmp/new-HydroBatch
bash Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/run_hydrofusion_task4_cli_checks.sh /tmp/new-HydroBatch /tmp/new-cli-evidence
python3 Hydro/experiments/gistohq_sligo/link_hydrofusion_check_from_batch_build.py /tmp/new-HydroBatch /tmp/new-reload-check
/tmp/new-reload-check /tmp/new-cli-evidence/log_residual/hydrofusion_ffn_batch_smoke /tmp/new-cli-evidence/log_residual/hydrofusion_ffn_batch_smoke/hydrofusion_metadata.txt /tmp/new-cli-evidence/log_residual/hydrofusion_ffn_batch_smoke/hydrofusion_origin_target_map.csv ffn_hydrofusion
python3 Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/check_hydrofusion_bundle_integrity.py /tmp/new-reload-check /tmp/new-cli-evidence/log_residual/hydrofusion_ffn_batch_smoke /tmp/new-integrity-evidence
python3 Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks/check_hydrofusion_repeatability.py /tmp/new-HydroBatch /tmp/new-seed-evidence
python3 -m unittest discover -s Hydro/experiments/gistohq_sligo/hydrofusion_task4_cli_checks -p 'test_*.py' -v
```

The direct-log-Q bundle is similarly under
`direct_log_q/hydrofusion_ffn_batch_smoke_direct_log_q/`. Use its own managed
metadata/map paths. The link helper also supports legacy inference/exporter tests.
Standalone metadata test:

```
g++ -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined -fno-sanitize-recover=all Hydro/experiments/gistohq_sligo/test_hydrofusion_metadata.cpp -o /tmp/new-metadata-test
/tmp/new-metadata-test
```

Next: independently tested development-only native loading/qualifier mapping,
then the queued LSTM adapter and separately specified hybrid/baseline work.
A prospectively frozen matched comparison must precede any accuracy claim;
no further research-training campaign is authorized by this close-out.
