# HydroPINN Approaches

HydroPINN exposes five local approaches for comparing data-driven and physics-informed hydrology workflows from the same UI. This document summarizes the model intent, inputs, losses, governing equations, diagnostics, and recommended comparison workflow.

## Notation

Let:

- \(t_i\) be the timestamp or normalized simulation coordinate for sample \(i\).
- \(\mathbf{x}_i \in \mathbb{R}^{d}\) be the input feature vector at sample \(i\).
- \(y_i\) be the observed target value.
- \(\hat{y}_i = f_\theta(\mathbf{x}_i)\) be the model prediction with trainable parameters \(\theta\).
- \(N\) be the number of training samples.
- \(\mathcal{L}_{data}\) be the supervised data loss.
- \(\mathcal{L}_{phys}\) be the physics residual loss.

The common supervised loss is mean squared error:

\[
\mathcal{L}_{data}
= \frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i - y_i\right)^2.
\]

PINN-capable approaches combine supervised and physics losses as:

\[
\mathcal{L}_{total}
= w_{data}\,\mathcal{L}_{data}
+ w_{phys}\,\mathcal{L}_{phys},
\]

where `Data weight` controls \(w_{data}\) and `Physics weight` controls \(w_{phys}\). The standalone **PINN** approach currently uses physics-first training by setting \(w_{data}=0\) during dispatch.


## Watershed synthetic generators and water-balance PINNs

HydroPINN includes hydrology-oriented synthetic profiles for exercising PINN
losses before field data are connected:

- `watershed_balance`: the primary watershed stress test with effective
  precipitation, evapotranspiration, temperature, soil storage, groundwater
  storage, impervious fraction, and runoff. The generator combines storm pulses,
  snowmelt contribution, infiltration, soil-to-groundwater recharge, baseflow,
  lateral flow, and impervious quickflow.
- `rainfall_runoff`: the smaller event-scale rainfall, evapotranspiration,
  temperature, soil storage, and runoff baseline.

For `water_balance`, the PINN residual reads the first mass-balance columns as
\(P\), \(ET\), predicted runoff \(Q_\theta\), and storage \(S\):

\[
r_\theta(t) = P(t) - ET(t) - Q_\theta(t) - \frac{dS}{dt}.
\]

This keeps the residual general enough for rainfall-runoff and broader watershed
experiments while allowing additional watershed descriptors to remain available
as supervised features.

## Approach order in the UI

1. **FFN**
2. **FFN + PINN**
3. **LSTM**
4. **LSTM + PINN**
5. **PINN**

The standalone **PINN** is intentionally listed as the fifth approach so it is visually separated from the paired FFN/LSTM baseline-plus-physics variants.

## Summary table

| Approach | Backend | Main idea | Training objective | Lag settings |
| --- | --- | --- | --- | --- |
| **FFN** | `FFNWrapper` | Feed-forward supervised baseline | \(\mathcal{L}_{data}\) | Optional time-lagged FFN inputs |
| **FFN + PINN** | `FFNPINNWrapper` | FFN with physics regularization | \(w_{data}\mathcal{L}_{data}+w_{phys}\mathcal{L}_{phys}\) | Optional time-lagged FFN inputs |
| **LSTM** | `LSTMWrapper` | Recurrent supervised sequence model | \(\mathcal{L}_{data}\) | Ignored; LSTM carries sequence memory |
| **LSTM + PINN** | `LSTMPINNWrapper` | LSTM with physics regularization | \(w_{data}\mathcal{L}_{data}+w_{phys}\mathcal{L}_{phys}\) | Ignored; LSTM carries sequence memory |
| **PINN** | `FFNPINNWrapper` | Physics-first standalone PINN | \(\mathcal{L}_{phys}\) | Ignored; uses physics-coordinate inputs |

## 1. FFN

### Purpose

**FFN** is the simplest supervised baseline. It estimates a direct nonlinear mapping from input features to the target:

\[
\hat{y}_i = f_\theta(\mathbf{x}_i).
\]

### Model form

For a feed-forward network with \(L\) hidden layers:

\[
\mathbf{h}^{(0)} = \mathbf{x},
\]

\[
\mathbf{h}^{(\ell)} = \phi\left(\mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right),
\quad \ell=1,\ldots,L,
\]

\[
\hat{y} = \mathbf{W}^{(L+1)}\mathbf{h}^{(L)} + b^{(L+1)}.
\]

Here \(\phi\) is the configured activation function, such as `relu`, `tanh`, or `sigmoid`.

### Loss

\[
\mathcal{L}_{FFN} = \mathcal{L}_{data}.
\]

### Inputs and lags

If time-lagged FFN inputs are enabled, lag groups are expanded into features. For a scalar input series \(u(t)\) and lags \(\tau_1,\ldots,\tau_k\):

\[
\mathbf{x}_i = \left[u(t_i-\tau_1), u(t_i-\tau_2), \ldots, u(t_i-\tau_k)\right].
\]

For multiple input variables, the feature vector concatenates each variable's configured lag group.

### Use when

Use **FFN** to establish a fast supervised baseline before adding recurrence or physics constraints.

## 2. FFN + PINN

### Purpose

**FFN + PINN** keeps the FFN architecture but adds a physics residual penalty. It answers: does a physics-informed constraint improve a supervised FFN baseline?

### Model form

\[
\hat{y}_i = f_\theta(\mathbf{x}_i),
\]

with the same FFN hidden-layer structure as the baseline.

### Loss

\[
\mathcal{L}_{FFN+PINN}
= w_{data}\,\mathcal{L}_{data}
+ w_{phys}\,\mathcal{L}_{phys}.
\]

The physics residual depends on the selected PINN physics profile. For a generic residual \(r_\theta(t_i, \mathbf{x}_i)\):

\[
\mathcal{L}_{phys}
= \frac{1}{N}\sum_{i=1}^{N} r_\theta(t_i, \mathbf{x}_i)^2.
\]

### Inputs and lags

**FFN + PINN** can use the same lagged features as **FFN**. For forcing or water-balance physics profiles, the feature set should include the required forcing or storage variables.

### Use when

Use **FFN + PINN** when the FFN baseline is useful, but you want hydrology-inspired regularization to improve physical plausibility or extrapolation.

## 3. LSTM

### Purpose

**LSTM** is a recurrent supervised baseline. It learns temporal dynamics from ordered sequences rather than explicit FFN lag expansion.

### Model form

For sequence input \(\mathbf{x}_1,\ldots,\mathbf{x}_T\), an LSTM updates hidden and cell states:

\[
\mathbf{i}_t = \sigma\left(\mathbf{W}_i\mathbf{x}_t + \mathbf{U}_i\mathbf{h}_{t-1} + \mathbf{b}_i\right),
\]

\[
\mathbf{f}_t = \sigma\left(\mathbf{W}_f\mathbf{x}_t + \mathbf{U}_f\mathbf{h}_{t-1} + \mathbf{b}_f\right),
\]

\[
\mathbf{o}_t = \sigma\left(\mathbf{W}_o\mathbf{x}_t + \mathbf{U}_o\mathbf{h}_{t-1} + \mathbf{b}_o\right),
\]

\[
\tilde{\mathbf{c}}_t = \tanh\left(\mathbf{W}_c\mathbf{x}_t + \mathbf{U}_c\mathbf{h}_{t-1} + \mathbf{b}_c\right),
\]

\[
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t,
\]

\[
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t),
\]

\[
\hat{y}_t = g_\theta(\mathbf{h}_t).
\]

### Loss

\[
\mathcal{L}_{LSTM} = \mathcal{L}_{data}.
\]

### Inputs and lags

FFN lag-step settings are ignored because the LSTM sequence window carries temporal memory internally.

### Use when

Use **LSTM** when sequence memory and temporal ordering are expected to matter more than manually engineered lag features.

## 4. LSTM + PINN

### Purpose

**LSTM + PINN** combines recurrent sequence modeling with a physics residual penalty.

### Model form

\[
\hat{y}_t = g_\theta(\mathbf{h}_t),
\]

where \(\mathbf{h}_t\) is produced by the LSTM recurrence.

### Loss

\[
\mathcal{L}_{LSTM+PINN}
= w_{data}\,\mathcal{L}_{data}
+ w_{phys}\,\mathcal{L}_{phys}.
\]

### Inputs and lags

Like **LSTM**, it uses sequence windows internally and ignores FFN lag-step settings. Physics profiles still require appropriate forcing or storage inputs when applicable.

### Use when

Use **LSTM + PINN** when you want sequence memory and physics-informed regularization together.

## 5. PINN

### Purpose

**PINN** is a standalone physics-first comparison point. It asks: how much behavior can the chosen physics residual explain without supervised data-loss fitting?

### Backend and dispatch behavior

The current runner reuses the feed-forward PINN backend (`FFNPINNWrapper`) and dispatches standalone **PINN** with:

\[
w_{data} = 0,
\]

\[
w_{phys} \ge 1.
\]

It also disables FFN time-lagged inputs for this approach.

### Loss

\[
\mathcal{L}_{PINN} = \mathcal{L}_{phys}.
\]

### Inputs

Standalone **PINN** uses physics-coordinate style inputs. The selected physics profile determines which input columns are meaningful. For example, an exponential-decay profile mainly needs time, while water-balance profiles need rainfall, evapotranspiration, runoff, and storage-related information.

### Use when

Use **PINN** as a physics-first diagnostic. It is useful for measuring whether the selected residual captures the target behavior on its own, but it is not a replacement for supervised calibration when measured targets are available.

## PINN physics profiles and equations

The PINN-capable approaches are **FFN + PINN**, **LSTM + PINN**, and **PINN**. They share the same physics-profile controls.

### Exponential decay: `exp_decay`

The exponential-decay ODE is:

\[
\frac{dy}{dt} + \lambda y = 0.
\]

The model predicts \(\hat{y}(t)\), and automatic differentiation estimates \(d\hat{y}/dt\). The residual is:

\[
r_{decay}(t) = \frac{d\hat{y}}{dt} + \lambda \hat{y}.
\]

The physics loss is:

\[
\mathcal{L}_{phys,decay}
= \frac{1}{N}\sum_{i=1}^{N}\left(\frac{d\hat{y}_i}{dt} + \lambda \hat{y}_i\right)^2.
\]

### Forcing-driven reservoir: `linear_reservoir`

A simple forcing-driven reservoir analogue can be written as:

\[
\frac{dy}{dt} + \lambda y - g u = 0,
\]

where:

- \(u\) is a forcing input,
- \(g\) is the forcing gain,
- \(\lambda\) is the decay or drainage coefficient.

The residual is:

\[
r_{forcing}(t) = \frac{d\hat{y}}{dt} + \lambda \hat{y} - g u.
\]

The physics loss is:

\[
\mathcal{L}_{phys,forcing}
= \frac{1}{N}\sum_{i=1}^{N}\left(\frac{d\hat{y}_i}{dt} + \lambda \hat{y}_i - g u_i\right)^2.
\]

### First-order CSTR analogue: `cstr_first_order`

The CSTR-style profile uses the same first-order forcing form as a reservoir/treatment analogue:

\[
\frac{dy}{dt} + \lambda y - g u = 0.
\]

It is useful when the process behaves like a first-order input-response system.

### Water balance: `water_balance`

A simplified rainfall-runoff water-balance equation is:

\[
P - ET - Q - \frac{dS}{dt} = 0,
\]

where:

- \(P\) is rainfall or precipitation,
- \(ET\) is evapotranspiration,
- \(Q\) is runoff/discharge,
- \(S\) is storage.

HydroPINN uses model prediction as runoff:

\[
Q \approx \hat{y}.
\]

The residual is:

\[
r_{wb}(t) = P(t) - ET(t) - \hat{y}(t) - \frac{dS}{dt}.
\]

The physics loss is:

\[
\mathcal{L}_{phys,wb}
= \frac{1}{N}\sum_{i=1}^{N}\left(P_i - ET_i - \hat{y}_i - \frac{dS_i}{dt}\right)^2.
\]

For finite-difference storage dynamics, the derivative is approximated as:

\[
\frac{dS_i}{dt} \approx \frac{S_i - S_{i-1}}{\Delta t}.
\]

## GA lag optimization scope

GA lag optimization applies to **FFN** and **FFN + PINN** because those approaches can use explicit lagged FFN feature groups. It does not apply to:

- **LSTM**, because sequence memory is handled by the recurrent state.
- **LSTM + PINN**, for the same reason.
- **PINN**, because standalone PINN disables FFN time-lagged inputs and uses physics-coordinate inputs.

A candidate lag specification uses semicolon-separated feature groups and comma-separated lags within a group, for example:

```text
1,2;1;1,3
```

This means:

- feature group 1 uses lags 1 and 2,
- feature group 2 uses lag 1,
- feature group 3 uses lags 1 and 3.

## Evaluation metrics

HydroPINN plots and summaries use the configured run result metrics. Common regression metrics are:

### MSE

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i-y_i\right)^2.
\]

### RMSE

\[
RMSE = \sqrt{MSE}.
\]

### MAE

\[
MAE = \frac{1}{N}\sum_{i=1}^{N}\left|\hat{y}_i-y_i\right|.
\]

### Coefficient of determination

\[
R^2 = 1 - \frac{\sum_{i=1}^{N}\left(y_i-\hat{y}_i\right)^2}{\sum_{i=1}^{N}\left(y_i-\bar{y}\right)^2}.
\]

where:

\[
\bar{y}=\frac{1}{N}\sum_{i=1}^{N}y_i.
\]

## Practical comparison workflow

1. Train **FFN** to establish a supervised feed-forward baseline.
2. Train **FFN + PINN** to measure whether physics regularization improves the FFN baseline.
3. Train **LSTM** to evaluate sequence-memory benefits.
4. Train **LSTM + PINN** to evaluate sequence memory plus physics regularization.
5. Train **PINN** to measure the standalone physics-first fit.
6. Compare all five approaches in the Plot and Performance tabs:
   - target-vs-predicted curves,
   - 1:1 target/predicted scatter,
   - Taylor diagram,
   - approach subplots,
   - residual-vs-time plot,
   - absolute-error CDF,
   - summary metrics.

## Interpretation guidance

- If **FFN** performs well and **FFN + PINN** improves stability or extrapolation, the physics residual is likely helpful.
- If **LSTM** outperforms **FFN**, temporal memory is likely important.
- If **LSTM + PINN** improves over **LSTM**, physics regularization is helpful even when the recurrent model already captures sequence structure.
- If standalone **PINN** performs poorly, the selected physics residual alone is not sufficient to reproduce the observed target.
- If standalone **PINN** performs well, the chosen residual may strongly explain the system behavior, but supervised calibration should still be considered for measured data.

## Notes and limitations

- The approach names in this document are UI names; backend class names may differ.
- Standalone **PINN** currently uses physics-only loss in the HydroPINN runner.
- The forcing and water-balance profiles work best with CSV data or synthetic profiles that provide the required multi-feature inputs.
- Physics-informed losses can be sensitive to feature scaling, time normalization, loss weights, and the selected physics profile.
