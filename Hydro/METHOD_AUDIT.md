# Hydro method audit

This audit evaluates the five Hydro methods **before further hyperparameter sweeps**. The goal is to separate optimization problems from method/formulation problems.

## Current status

| Method | Status | Main conclusion |
|---|---|---|
| FFN | **Sound baseline** | Supervised data path, chronological split, train-only scaling, validation checkpointing, inverse scaling, and held-out metrics are internally consistent. |
| LSTM | **Sound baseline** | Sequence construction and held-out evaluation are internally consistent. GIStoOHQ sequences contain forcing variables only, so preceding forcing history across a split boundary is predictor context rather than target leakage. |
| FFN + PINN | **Corrected for GIStoOHQ** | Uses one joint data/physics Adam update and finite-difference runoff dynamics along the forcing trajectory. No precomputed storage state is used. |
| LSTM + PINN | **Corrected for GIStoOHQ** | Data and physics terms now participate in the same sequential mini-batch optimizer update, so `physics_weight` is a genuine tradeoff parameter. |
| PINN | **Corrected reduced formulation** | Uses physics-only runoff dynamics plus one observed initial-condition anchor, which is required to identify the first-order forced ODE. |

## 1. Supervised baselines

Current GIStoOHQ supervised inputs are

\[
[P,\,T,\,RH,\,wind,\,solar,\,PET].
\]

### FFN

When lagging is enabled, lagged inputs are constructed before the chronological split. Input and target scalers are fitted using only the training subset. Validation is used for checkpoint selection, and predictions are inverse-transformed before physical-unit metrics are evaluated.

**Decision:** retain FFN unchanged as the feed-forward data-driven baseline.

### LSTM

The LSTM builds forcing sequences first and predicts the target associated with each sequence endpoint. The chronological split is then performed on sequences. Input and target scaling is fitted only on training sequences, and the validation-selected checkpoint is restored before test evaluation.

GIStoOHQ LSTM inputs contain meteorological/hydrologic forcings but not previous observed discharge. Therefore a validation/test sequence may legitimately include forcing history from immediately before the split; this is available predictor history, not target leakage.

**Decision:** retain LSTM unchanged as the recurrent data-driven baseline.

## 2. Removed circular-storage formulation

The previous GIStoOHQ physics adapter constructed

\[
S_t = \max\left[0,\;S_{t-1}+\Delta t\left(P_t-PET_t-kS_{t-1}\right)\right]
\]

and then enforced

\[
r_t=P_t-PET_t-Q_t-\frac{S_t-S_{t-1}}{\Delta t}.
\]

Ignoring clipping, substitution gives approximately

\[
r_t \approx kS_{t-1}-Q_t,
\]

so the physics term mainly forced

\[
Q_t\approx kS_{t-1}.
\]

That was algebraically circular because the storage trajectory had already been generated from the same reservoir assumption.

This storage construction has now been removed for GIStoOHQ physics modes.

## 3. Current GIStoOHQ physics forcing layout

Physics-informed GIStoOHQ runs now use

\[
[t,\,P_{eff},\,P,\,PET,\,T,\,RH,\,wind,\,solar],
\]

where

\[
P_{eff}=\max(P-PET,0).
\]

No observed discharge is used to construct an input state and no synthetic storage trajectory is supplied to the models. The longest contiguous hourly segment is retained so finite-difference derivatives never cross a data gap.

The reduced conceptual reservoir is

\[
\frac{dQ}{dt}=k(P_{eff}-Q),
\]

with residual

\[
r_Q=\frac{dQ}{dt}-k(P_{eff}-Q).
\]

This is deliberately a **conceptual physics regularizer**, not a claim of exact watershed physics.

## 4. FFN + PINN

For GIStoOHQ, the corrected FFN+PINN uses ordered mini-batches and a finite-difference total derivative of the predicted runoff along the forcing trajectory. This avoids treating \(\partial Q/\partial t\) at fixed forcing as the physical trajectory derivative.

After the warm-up period, each optimizer step uses

\[
\mathcal L = w_d\,\mathrm{MSE}(Q_{pred},Q_{obs})
            +w_p\left[\mathrm{MSE}(r_Q,0)+0.05\,\mathrm{MSE}(\max(-Q_{pred},0),0)\right].
\]

The data and physics gradients are combined before a single Adam update, so `physics_weight` has a real relative meaning.

The legacy FFN-PINN backend remains available for non-GIStoOHQ profiles.

## 5. LSTM + PINN

The previous LSTM+PINN applied physics in a second optimizer step after supervised mini-batches. With Adam, scaling an isolated physics gradient by a positive scalar was largely normalized by the optimizer moments, which explained the nearly identical results across different physics weights.

The corrected GIStoOHQ LSTM+PINN now uses ordered sequence mini-batches. For each mini-batch, runoff predictions, data loss, finite-difference runoff dynamics, and the non-negativity penalty are computed together. One combined loss is backpropagated and one Adam step is performed.

Thus

\[
\mathcal L = w_d\mathcal L_{data}+w_p\mathcal L_{physics}
\]

is now implemented as an actual joint objective.

## 6. Standalone PINN

A forced first-order ODE does not have a unique trajectory without one initial/boundary condition. The corrected standalone GIStoOHQ PINN therefore does **not** use a full-series discharge data loss. It uses:

1. the runoff-reservoir physics residual over the training forcing trajectory;
2. one observed runoff value at the start of the training period as the initial-condition anchor; and
3. a small non-negativity penalty.

Its objective is

\[
\mathcal L = w_p\,\mathrm{MSE}(r_Q,0)
            +\mathrm{MSE}(Q(t_0),Q_{obs}(t_0))
            +0.05\,\mathrm{MSE}(\max(-Q,0),0).
\]

The remaining training-period runoff observations do not enter the PINN optimization objective. They are used only later for evaluation/diagnostics.

This should be described as a **physics-driven model with an observed initial condition**, not as completely data-free.

## 7. What to test next

Do not resume broad hyperparameter sweeps yet. First perform a small five-method diagnostic run and verify method behavior:

1. FFN and LSTM should reproduce their established supervised baselines within deterministic expectations.
2. FFN+PINN metrics should differ from the old circular-storage results.
3. LSTM+PINN runs with materially different `physics_weight` values must now produce different fitted models/metrics.
4. Standalone PINN should no longer reproduce the pathological old `Q≈kS` behavior and should satisfy its initial condition.
5. Physics residuals should be inspected together with NSE/KGE/PBIAS; low pointwise RMSE alone is not sufficient.

Only after these checks pass should optimizer, architecture, or physics-weight sweeps resume.
