# Full Tuning Pipeline Data-Source Invariants

The full tuning pipeline must use the data source selected in the HydroPINN GUI. It must never infer the source from a historical base JSON file.

## Source routing

`generate_unified_sweep.py` explicitly overwrites all source-specific fields in every generated configuration.

- **Synthetic**
  - `use_hydro_package=false`
  - `use_csv_data=false`
  - `hydro_package_path=""`
  - `csv_path=""`
  - selected `synthetic_profile`, sample count, `t_start`, and `t_end` are copied from the pipeline request.
- **CSV**
  - `use_csv_data=true`
  - `use_hydro_package=false`
  - Hydro-package path is cleared.
- **Hydro Package**
  - `use_hydro_package=true`
  - `use_csv_data=false`
  - CSV path is cleared.

The generated manifest records `data_source`, `synthetic_profile`, `synthetic_truth_k`, `hydro_package_path`, and `csv_path` so source leakage is visible before training.

## Controlled reduced-reservoir synthetic validation

The five-method Synthetic physics pipeline requires:

```text
Data source: Synthetic
Synthetic profile: reduced_reservoir
```

All five methods use the same generated forcing and runoff truth. The ground-truth coefficient is stored separately as:

```text
synthetic_reservoir_truth_k
```

The default controlled truth is `k_truth = 0.08`. This value is used only to generate the target hydrograph. Candidate/model reservoir coefficients continue to use `lambda_decay` / `storage_coeff` and may be swept independently. Therefore a model-k sweep no longer regenerates a different target series for every candidate.

## GUI full-pipeline behavior

The generated GUI gives stable object names to the source widgets. `full_tuning_pipeline.cpp` snapshots the current GUI values when the user starts the pipeline and asks for confirmation before Stage 1. The same snapshot is passed to every stage.

For a Synthetic five-method pipeline, a profile other than `reduced_reservoir` is rejected before generation. CSV physics runs also reject an incompatible column layout.

## Regression checks

Controlled Synthetic regression:

```bash
bash Hydro/experiments/gistohq_sligo/run_synthetic_method_regression.sh
```

This checks before training that all seven focused jobs:

1. are `data_source=synthetic`;
2. use `synthetic_profile=reduced_reservoir`;
3. have one fixed `synthetic_truth_k`;
4. contain no Hydro-package or CSV path/flag;
5. export successfully; and
6. produce different LSTM+PINN fitted metrics for different physics weights.

Sligo regression is independently pinned to Hydro Package input:

```bash
bash Hydro/experiments/gistohq_sligo/run_method_regression.sh
```

This explicit pin prevents future changes in defaults or GUI state from changing the Sligo regression source.
