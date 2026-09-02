# Hydro method audit

This audit intentionally evaluates the five Hydro methods **before further hyperparameter sweeps**. The goal is to separate optimization problems from method/formulation problems.

## Status

| Method | Current status | Main conclusion |
|---|---|---|
| FFN | **Sound baseline** | Supervised data path, chronological split, train-only scaling, validation checkpointing, inverse scaling, and held-out metrics are internally consistent. |
| LSTM | **Sound baseline** | Sequence construction and held-out evaluation are internally consistent. For GIStoOHQ, sequences contain forcing variables only, so use of preceding forcing history across a split boundary is legitimate forecast context rather than target leakage. |
| FFN + PINN | **Implementation works; physics formulation needs revision** | Data and physics losses are combined in the same optimization step, so `physics_weight` is meaningful. However, the current GIStoOHQ latent-storage construction makes the water-balance residual collapse toward a prescribed linear reservoir. |
| LSTM + PINN | **Needs code correction and physics revision** | Physics is currently applied in a separate Adam step after supervised mini-batches. Scaling that isolated gradient by `physics_weight` is largely normalized by Adam, explaining nearly identical results across physics weights. The shared latent-storage formulation also needs revision. |
| PINN | **Formulation needs redesign** | With no discharge data loss, the current water-balance constraint is not an independent rainfall-runoff model; it primarily reproduces the assumed latent linear-reservoir relationship and does not identify observed streamflow dynamics. |

## 1. FFN

Current GIStoOHQ supervised inputs are the verified six forcings

\[
[P,\,T,\,RH,\,wind,\,solar,\,PET].
\]

When lagging is enabled, lagged inputs are constructed before the chronological split. Input and target scalers are fitted using only the training subset. Validation is used for checkpoint selection, and predictions are inverse-transformed before physical-unit metrics are evaluated.

**Decision:** retain FFN as the primary feed-forward data-driven baseline. No method-level correction is presently required.

## 2. LSTM

The LSTM builds forcing sequences first and predicts the target associated with each sequence endpoint. The chronological split is then performed on sequences. Input and target scaling is fitted only on training sequences, and the validation-selected checkpoint is restored before test evaluation.

GIStoOHQ LSTM inputs contain meteorological/hydrologic forcings but not previous observed discharge. Therefore a validation/test sequence may legitimately include forcing history from immediately before the split; this is available predictor history, not leakage of the target.

**Decision:** retain LSTM as the primary recurrent data-driven baseline. No method-level correction is presently required.

## 3. Current GIStoOHQ physics formulation

For physics-informed runs, the current package adapter constructs a latent storage state using

\[
S_t = \max\left[0,\;S_{t-1}+\Delta t\left(P_t-PET_t-kS_{t-1}\right)\right].
\]

The PINN water-balance residual is then evaluated as

\[
r_t=P_t-PET_t-Q_t-\frac{S_t-S_{t-1}}{\Delta t}.
\]

Ignoring the clipping operation for the moment, substitution of the storage update gives

\[
r_t \approx kS_{t-1}-Q_t.
\]

Thus the physics term does **not** provide an independent catchment water-balance constraint. It pushes the learned runoff toward the preselected linear-reservoir relationship

\[
Q_t\approx kS_{t-1}.
\]

This explains why changing `k` strongly changes physics-informed behavior and why a model can obtain a small pointwise error while still giving poor NSE/KGE/bias: the physics term is enforcing a simplified prescribed response that may not match the observed watershed dynamics.

**Required redesign:** do not precompute a storage trajectory with the same constitutive relation later enforced by the PINN. The physics constraint must contain independent information.

## 4. FFN + PINN

The FFN PINN backend currently forms one combined objective during a training step:

\[
\mathcal L = w_d\mathcal L_{data}+w_p\mathcal L_{physics}.
\]

After the warm-up period, both terms contribute to the same gradient before the Adam update. Therefore `physics_weight` has a real relative meaning in this implementation.

The remaining problem is the latent-storage formulation described above, not loss-weight propagation.

**Decision:** retain the FFN+PINN architecture, but replace the GIStoOHQ physics residual before further tuning.

## 5. LSTM + PINN

The current recurrent backend performs supervised mini-batch updates first. It then performs a second, separate full-sequence optimizer step:

1. zero gradients;
2. evaluate physics loss;
3. call `(physics_weight * physics_loss).backward()`;
4. call `Adam.step()`.

Because the physics gradient is isolated in its own Adam update, multiplication by a positive scalar is largely removed by Adam's moment normalization. Consequently values such as `0.001`, `0.005`, `0.01`, `0.025`, `0.05`, and `0.1` can yield effectively the same parameter update direction/magnitude and therefore nearly identical results.

This is a method implementation issue, not evidence that physics weight is irrelevant.

**Required correction:** after warm-up, form data and physics terms in a **single differentiable objective** and perform one optimizer update for that objective. For conservation residuals that require chronology, use ordered sequence batches (or a full ordered training sequence) rather than a separate weighted Adam step.

## 6. Standalone PINN

The current standalone PINN sets

\[
w_d=0,\qquad w_p=1.
\]

With the current precomputed storage, the model is therefore trained primarily to satisfy

\[
Q\approx kS.
\]

This is closer to fitting the output of a prescribed conceptual reservoir than solving an independently constrained rainfall-runoff inverse problem. It also lacks a clear independent initial/boundary-state treatment for observed streamflow dynamics.

**Required redesign:** a standalone rainfall-runoff PINN needs a well-posed physics problem. A practical reduced formulation should include an independent state/evolution equation and an initial/boundary condition. If discharge data are used only to specify an initial condition, that should be stated explicitly rather than described as fully data-free.

## Recommended method redesign

For the next implementation revision, use a reduced reservoir equation that acts directly on predicted runoff without first generating storage from the same equation. One candidate is

\[
\frac{dQ}{dt}=k\left(P-PET-Q\right),
\]

which follows from a linear reservoir only after eliminating storage. Its residual is

\[
r_Q=\frac{dQ}{dt}-k\left(P-PET-Q\right).
\]

This is still simplified and must be presented as a **conceptual regularizer**, not exact watershed physics, but it avoids the current algebraic circularity. A more complete later formulation can introduce a learned latent storage state and jointly enforce mass balance plus a storage-discharge relation.

For hybrid models:

\[
\mathcal L=w_d\,\mathrm{MSE}(Q_{pred},Q_{obs})+w_p\,\mathrm{MSE}(r_Q,0).
\]

For a standalone PINN, the physics objective must additionally include an initial/boundary condition so the solution is identifiable.

## Next code changes

1. Leave FFN and LSTM method implementations unchanged.
2. Remove dependence of the GIStoOHQ physics residual on the precomputed `S_latent` trajectory.
3. Replace the current residual with an independent runoff-evolution residual (initially the reduced-reservoir form above).
4. Change LSTM+PINN so data and physics losses participate in the same optimizer update.
5. Add an explicit initial-condition loss for standalone PINN.
6. Only after these method corrections, rerun a small diagnostic comparison of the five methods; do **not** resume broad sweeps until each method reacts correctly to its physics parameters.
