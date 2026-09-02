# Hydro method audit

This audit evaluates the five Hydro methods **before further hyperparameter sweeps**. The goal is to separate optimization problems from method/formulation problems.

## Current status

| Method | Status | Main conclusion |
|---|---|---|
| FFN | **Sound baseline** | Supervised split/scaling/checkpoint/evaluation path is internally consistent. |
| LSTM | **Sound baseline** | Sequence construction and held-out evaluation are internally consistent. |
| FFN + PINN | **Corrected, cross-source** | Joint data/physics update with finite-difference runoff dynamics; shared by Synthetic, CSV, and Hydro package inputs. |
| LSTM + PINN | **Corrected, cross-source** | Data and physics terms participate in the same sequential mini-batch Adam update for all reduced-reservoir data sources. |
| PINN | **Corrected, cross-source** | Physics-driven runoff dynamics plus one initial-condition anchor; shared by Synthetic, CSV, and Hydro package inputs. |

## 1. Supervised baselines

FFN and LSTM remain unchanged. Their data-source-specific input builders continue to define the supervised predictor set, and normalization is fitted only on the training subset.

## 2. Removed circular-storage formulation

The previous GIStoOHQ physics adapter constructed

\[
S_t = \max\left[0,\;S_{t-1}+\Delta t(P_t-PET_t-kS_{t-1})\right]
\]

and then enforced

\[
r_t=P_t-PET_t-Q_t-\frac{S_t-S_{t-1}}{\Delta t}.
\]

Ignoring clipping, substitution gives approximately

\[
r_t \approx kS_{t-1}-Q_t,
\]

so the physics term mainly forced \(Q\approx kS\). That storage reconstruction has been removed from the reduced-reservoir physics path.

## 3. Unified reduced-reservoir physics contract

The corrected physics-informed path uses

\[
\frac{dQ}{dt}=k(P_{eff}-Q),
\qquad
P_{eff}=\max(P-PET,0),
\]

with residual

\[
r_Q=\frac{dQ}{dt}-k(P_{eff}-Q).
\]

The common tensor contract is

\[
[t,\,P_{eff},\,\text{optional forcing features}],
\]

and no storage state is reconstructed from the same governing equation.

This reduced equation is a **conceptual physics regularizer**, not exact watershed physics.

### Synthetic

Reduced-reservoir physics runs use a dedicated synthetic verification series generated directly from the same forced ODE. This provides a controlled unit-test case in which the governing process is known.

The older `watershed_balance` and `rainfall_runoff` synthetic processes are retained as separate known-state hydrologic test cases. In particular, `physics_profile="water_balance"` can still be used when a synthetic case independently provides storage. These known-state tests are intentionally distinct from the reduced-reservoir formulation.

### CSV

Reduced-reservoir CSV physics no longer manufactures forcing from the target runoff. That former fallback was target leakage.

The explicit CSV physics contract is:

- column 0: physical time;
- column 1: precipitation;
- column 2: PET;
- `csv_y_column`: runoff target/diagnostic series;
- optional remaining non-target columns: explanatory forcings.

The loader derives \(P_{eff}=\max(P-PET,0)\). CSV physics therefore requires `csv_x_column=0` and `csv_y_column>=3`.

### Hydro packages / GIStoOHQ

GIStoOHQ reduced-reservoir inputs use

\[
[t,\,P_{eff},\,P,\,PET,\,T,\,RH,\,wind,\,solar].
\]

The longest contiguous hourly segment is retained so finite-difference derivatives never cross a data gap. Generic Hydro packages with explicit P/PET are transformed to the same forcing contract without constructing storage.

## 4. FFN + PINN

For all reduced-reservoir data sources, FFN+PINN uses ordered mini-batches and a finite-difference total derivative along the forcing trajectory. After warm-up,

\[
\mathcal L = w_d\,\mathrm{MSE}(Q_{pred},Q_{obs})
            +w_p\left[\mathrm{MSE}(r_Q,0)+0.05\,\mathrm{MSE}(\max(-Q_{pred},0),0)\right].
\]

Data and physics gradients are combined before one Adam update, so `physics_weight` is a genuine tradeoff parameter.

Known-state or legacy physics profiles continue through the legacy FFN-PINN backend rather than being silently converted.

## 5. LSTM + PINN

For all reduced-reservoir data sources, LSTM+PINN uses ordered sequence mini-batches. Prediction, data loss, finite-difference runoff dynamics, and non-negativity are computed in the same batch and backpropagated through one combined objective:

\[
\mathcal L = w_d\mathcal L_{data}+w_p\mathcal L_{physics}.
\]

The former separate physics-only Adam step is no longer used for the reduced-reservoir path.

## 6. Standalone PINN

A first-order forced ODE requires one initial/boundary condition. The standalone reduced-reservoir PINN therefore uses:

1. the runoff-reservoir residual over the training forcing trajectory;
2. one runoff value at the beginning of the training period as the initial-condition anchor; and
3. a small non-negativity penalty.

Its objective is

\[
\mathcal L = w_p\,\mathrm{MSE}(r_Q,0)
            +\mathrm{MSE}(Q(t_0),Q_0)
            +0.05\,\mathrm{MSE}(\max(-Q,0),0).
\]

For real/CSV data, \(Q_0\) is the first observed runoff value. For the controlled synthetic reduced-reservoir test, it is the known synthetic initial condition. Remaining runoff observations do not enter the standalone PINN optimization objective; they are used for evaluation.

## 7. Data-source matrix

| Data source | FFN/LSTM | Reduced FFN+PINN | Reduced LSTM+PINN | Reduced PINN | Known-state water balance |
|---|---|---|---|---|---|
| Synthetic | yes | yes | yes | yes | yes |
| CSV | yes | yes, explicit P/PET required | yes, explicit P/PET required | yes, explicit P/PET required | only if an explicit compatible state layout is provided |
| Hydro package | yes | yes | yes | yes | yes when storage is independently supplied |
| GIStoOHQ | yes | yes | yes | yes | not used because observed storage is unavailable |

## 8. What to test next

Do not resume broad sweeps yet. First verify method behavior:

1. supervised FFN/LSTM still reproduce established baselines;
2. the reduced synthetic test is recovered by FFN+PINN, LSTM+PINN, and PINN;
3. materially different `physics_weight` values now change hybrid fitted models;
4. CSV physics rejects time+target-only files instead of deriving forcing from runoff;
5. GIStoOHQ physics results differ from the former circular-storage results;
6. evaluate NSE/KGE/PBIAS and residual behavior together rather than ranking on RMSE alone.

Only after these checks pass should broad tuning resume.
