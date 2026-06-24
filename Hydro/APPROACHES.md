# HydroPINN Approaches

HydroPINN exposes five local approaches for comparing data-driven and physics-informed hydrology workflows from the same UI. The approaches share the same data-source, training, plotting, and performance-assessment tabs where possible, but differ in model family and how physics residuals are used.

## Approach order in the UI

1. **FFN**
2. **FFN + PINN**
3. **LSTM**
4. **LSTM + PINN**
5. **PINN**

The standalone **PINN** is intentionally listed as the fifth approach so it is visually separated from the paired FFN/LSTM baseline-plus-physics variants.

## 1. FFN

**Purpose:** A feed-forward neural-network baseline for supervised regression.

**Backend:** `FFNWrapper`.

**Training signal:** Data loss only, using the selected target series.

**Inputs:** Uses the configured Hydro inputs. If **Use time-lagged FFN inputs** is enabled, lag groups from **Input lag steps** are expanded into FFN features.

**Use when:** You want a fast, supervised baseline before adding recurrent sequence memory or physics constraints.

## 2. FFN + PINN

**Purpose:** A feed-forward supervised model augmented with a physics residual.

**Backend:** `FFNPINNWrapper`.

**Training signal:** Combined supervised data loss and PINN physics loss. The relative strengths are controlled by **Data weight** and **Physics weight**.

**Inputs:** Can use the same FFN lag configuration as the FFN baseline. Physics profiles may also require forcing or storage-like inputs depending on the selected profile.

**Use when:** You want to keep the FFN baseline structure while regularizing it with hydrology-inspired physics.

## 3. LSTM

**Purpose:** A recurrent sequence baseline that learns temporal dynamics directly from sequences.

**Backend:** `LSTMWrapper`.

**Training signal:** Data loss only.

**Inputs:** Uses the LibTorch LSTM backend sequence handling. FFN lag-step settings are ignored because the LSTM keeps sequence memory internally.

**Use when:** You expect temporal context and sequence memory to matter more than explicit FFN lag engineering.

## 4. LSTM + PINN

**Purpose:** A recurrent sequence model augmented with a physics residual.

**Backend:** `LSTMPINNWrapper`.

**Training signal:** Combined supervised data loss and PINN physics loss.

**Inputs:** Uses the LSTM sequence backend plus the selected PINN physics profile and weights.

**Use when:** You want both recurrent sequence memory and physics-informed regularization.

## 5. PINN

**Purpose:** A standalone physics-first PINN comparison point.

**Backend:** The feed-forward PINN backend (`FFNPINNWrapper`) is reused with FFN time-lagged inputs disabled.

**Training signal:** Physics-only loss in the current runner. The UI dispatch sets `data_weight = 0` and ensures `physics_weight >= 1` for this approach.

**Inputs:** Uses physics-coordinate style inputs rather than FFN lagged features. The chosen physics profile determines which input columns are most useful.

**Use when:** You want to test how far the selected physics residual can explain the target behavior without supervised data-loss fitting.

## PINN physics profiles

The PINN-capable approaches (**FFN + PINN**, **LSTM + PINN**, and **PINN**) share the same physics-profile controls:

- **`exp_decay`**: Exponential-decay residual, useful for pure decay dynamics.
- **`linear_reservoir`**: Forcing-driven residual for reservoir-like hydrologic response.
- **`cstr_first_order`**: First-order forcing/reservoir analogue.
- **`water_balance`**: Rainfall-runoff mass-balance style residual using rainfall, evapotranspiration, runoff, and storage terms.

For forcing or water-balance profiles, CSV data or synthetic profiles with multi-feature inputs are recommended.

## Practical comparison workflow

1. Train **FFN** to establish a supervised baseline.
2. Train **FFN + PINN** to see whether physics regularization improves the FFN baseline.
3. Train **LSTM** to evaluate sequence-memory benefits.
4. Train **LSTM + PINN** to evaluate sequence memory plus physics regularization.
5. Train **PINN** to measure the standalone physics-first fit.
6. Use the Plot and Performance tabs to compare target-vs-predicted curves, 1:1 plots, Taylor diagrams, residuals, error CDFs, and summary metrics across all five approaches.

## Notes and limitations

- GA lag optimization applies only to **FFN** and **FFN + PINN**, because LSTM approaches use sequence memory internally and standalone **PINN** ignores FFN lagged inputs.
- Standalone **PINN** is a physics-first diagnostic approach in this UI, not a replacement for supervised calibration when measured target data is available.
- The approach names in this document are UI names; backend class names may differ.
