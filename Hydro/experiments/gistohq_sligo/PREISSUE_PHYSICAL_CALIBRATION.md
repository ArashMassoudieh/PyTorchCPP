# Training-only physical calibration

`calibrate_preissue_physical.cpp` uses the production native loader and reservoir
routing functions without constructing a neural model. It checks the water
ledger and confirms that the held-rate forecast does not change committed stores.
It reads observations only from the requested split, with the training package
physically cropped before calibration. The support and weather inputs are pinned
by the runner's plan. These tools are specific to the current development fold,
not a general-purpose optimizer or an operational availability certificate.

Build from the repository root:

```sh
g++ -std=c++17 -O2 -fPIC -Wall -Wextra -Werror \
 Hydro/experiments/gistohq_sligo/calibrate_preissue_physical.cpp \
 Hydro/dataset/hydro_checksum.cpp -o /tmp/preissue_physical_calibration \
 $(pkg-config --cflags --libs Qt5Core) -lcrypto
```

`run_preissue_physical_calibration.py PILOT BINARY NEW_OUTPUT` freezes 27 routing
shapes and analytically fits one bounded gain per shape on training rows only.
It saves the selected configuration before the first validation run. Validation
is a single evaluation, with no automatic retuning or expansion. The runner
refuses an existing output directory. Do not rerun the closed 2026-11-24 stage.
A failed subprocess ends the stage; it is not retried.

Parameters are reservoir release rates (`storage_coeff`, `lambda_decay`), the
fast/slow partition fraction (`runoff_coeff`), and effective rainfall gain
(`forcing_gain`). The fraction is not a total runoff multiplier. Zero initial
storage and linear routing make the global gain fit exact for this model; the
separate C++ synthetic test verifies state and forecast scaling.

The 2026-11-24 result is **not accepted as an improved physical model**. Training
selection changed validation NSE from -0.457671 to +0.396480, but introduced
-50.940% volume bias and lost to the unchanged dry reference (NSE 0.448204).
Retain the dry reference; do not change production defaults or existing hybrid
checkpoints. Water conservation passing does not establish hydrological skill.
All validation is reused development evidence, not blind generalization.

Evidence is in Dropbox HydroPINN paper/results/
`hydrofusion_preissue_physical_train_calibration_20261124/`. The independent
verifier rechecks hashes, scalar metrics, all 27 training gains, selection,
chronological support and the validation gate. The real CLI and fresh native
reload verify that the selected setting uses the existing production path.
No test rows, catchment-area changes, neural fits or manuscript edits.
