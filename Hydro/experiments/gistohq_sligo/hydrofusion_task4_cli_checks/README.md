# Synthetic CLI verification fixtures

These scripts exercise a supplied HydroBatch executable; they do not build one.
Use only the supplied fabricated-data configurations. The package/CSV rejection
fixtures deliberately name nonexistent paths. No real watershed data are required.

The HydroFusion jobs require an executable with the experimental `ffn_hydrofusion`
dispatch. Saving these fixtures does not accept that unfinished production change
by itself. The independent Task4 closure report records the aggregate-layout
diagnosis and subsequent frozen-source verification. A clean baseline without that
dispatch can still run `jobs_regression.batch` for the five original methods.

Run from any directory, using absolute executable paths:

```
./run_hydrofusion_task4_cli_checks.sh /absolute/path/to/HydroBatch
./compare_regression_against_baseline_build.sh /absolute/after/HydroBatch /absolute/baseline/HydroBatch
```

An omitted output directory creates a unique temporary evidence directory and
prints its location. An explicit output directory must not exist; scripts never
delete or reuse prior evidence. Executable SHA256 hashes are captured before
launch. Logs and outputs are retained after success or failure. These commands
perform small synthetic fits when supplied a real model executable.

The comparison requires nonempty prediction files with the expected header and
compares their bytes. It establishes agreement only for the supplied synthetic
configurations; it is not a semantic verifier of every prediction field, all
checkpoint paths, or predictive skill. Guard checks require failure with the
expected synthetic-only message; they do not establish real-data support.

Script-only regression tests use fake executables, with no model fitting:

```
python3 -m unittest discover -s /absolute/path/to/hydrofusion_task4_cli_checks -p test_runner_safety.py -v
```

They cover existing-output/input preservation, fatal hash failures, saved binary
hashes, matching/different/nonexistent data rows, and CLI success/guard orchestration.

Additional negative checks cover absent synthetic consent, unknown target mode,
missing/fractional/overflowing cutoffs. All integer boundaries are parsed before
conversion; old configurations may still omit them (the unset sentinel remains).
`check_hydrofusion_repeatability.py BINARY NEW_OUTPUT` performs three small
synthetic fits to check seed handling and validation checkpoint selection.
`test_cutoff_loader.py` compiles the real, Torch-free configuration loader under
undefined/float-cast overflow sanitizers and tests invalid numeric domains.

Current prototype acceptance/evidence is recorded in
`../TASK4_CLOSURE_20261118.md`. Forecast metadata/time maps now reside
INSIDE each managed experiment directory and are included in its manifest.
`check_hydrofusion_bundle_integrity.py VERIFIER SYNTHETIC_BUNDLE NEW_OUTPUT`
checks corrupted artifact copies without fitting any model or altering originals.
