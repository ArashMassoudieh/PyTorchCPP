# PINN reduced-reservoir physics rationale

## Current equation

The `linear_reservoir` profile uses

\[
\frac{dQ}{dt}=k\left(I^*-Q\right),\qquad k=\frac{1}{K},
\]

where `Q` is runoff expressed as catchment-equivalent depth per unit time and
`I*` is the simplified forcing currently defined as

\[
I^*=\max(P-PET,0).
\]

For GIStoOHQ packages, precipitation, PET, and observed runoff are all converted
to consistent hourly catchment-depth units before this residual is evaluated.

## Hydrologic basis

This ODE is the classical single linear-reservoir equation. Starting from
continuity,

\[
\frac{dS}{dt}=I-Q,
\]

and the linear storage-discharge relation

\[
S=KQ,
\]

substitution gives

\[
\frac{dQ}{dt}=\frac{1}{K}(I-Q).
\]

The USGS SCALP documentation writes this governing equation explicitly and uses
linear reservoirs in series for routing. Linear-reservoir and Nash-cascade
concepts are also long-established rainfall-runoff/routing models.

## Important limitation for the current Sligo experiment

The equation is physically legitimate as a **routing/storage model if `I` is the
actual reservoir inflow**. In the current reduced experiment, however,
`I*=max(P-PET,0)` is only a simplified net atmospheric input. It does not model
interception, infiltration, soil-moisture limitation, groundwater partitioning,
preferential flow, spatial heterogeneity, or separate fast/slow reservoirs.
Therefore the current profile must be described in the paper as a **reduced
linear-reservoir physical regularizer/benchmark**, not a complete rainfall-runoff
process model and not a full catchment water-balance closure.

This interpretation is consistent with the final Sligo diagnostics: models can
satisfy the reduced ODE closely while retaining substantial runoff-volume bias.
That behavior indicates structural mismatch between the reduced constraint and
real catchment dynamics, rather than an algebraic error in the linear-reservoir
ODE itself.

## Relation to current physics-informed hydrology literature

Recent hydrology literature generally favors richer conservation-aware or
process-based differentiable structures for real catchments:

- Frame et al. (2023), *Hydrological Processes*, DOI: 10.1002/hyp.14847,
  evaluated strict mass-conservation constraints in rainfall-runoff ML and found
  that strict closure can reduce predictive skill when forcing/target data are
  inconsistent or biased.
- Feng et al. (2022), *Water Resources Research*, DOI: 10.1029/2022WR032404,
  used differentiable HBV-type process models with explicit soil, groundwater,
  evapotranspiration, runoff-generation, and routing components and approached
  LSTM-level streamflow performance.
- Wang et al. (2024), *Water Resources Research*, DOI: 10.1029/2023WR036461,
  introduced the Mass-Conserving Perceptron for conservative geoscientific
  input-state-output dynamics.
- Bhasme et al. (2022), *Journal of Hydrology*, DOI:
  10.1016/j.jhydrol.2022.128618, coupled ML with a conceptual hydrologic model
  and assessed physical consistency through water-balance analysis.
- Chen et al. (2026), *Advances in Water Resources*, DOI:
  10.1016/j.advwatres.2026.105330, embedded a differentiable mass-conserving
  Nash instantaneous-unit-hydrograph routing module in a hybrid flood-forecast
  network.

## Paper decision

For the current five-method comparison, retain the single linear-reservoir ODE
as the deliberately reduced physics benchmark because:

1. the controlled synthetic experiment validates its implementation and
   recovers the known `k` with standalone PINN;
2. its governing equation is a standard linear-reservoir equation;
3. the real Sligo results provide a useful diagnostic of what happens when an
   intentionally simplified physical prior is imposed on a real watershed.

Do **not** interpret the Sligo-selected `k` values as calibrated physical
catchment recession constants. If the project is extended beyond this benchmark,
the next physics model should use a differentiable multi-store water-balance
backbone (for example HBV-like soil + fast/slow groundwater reservoirs, or a
mass-conserving storage network) rather than simply adding more weight to the
current single-reservoir residual.
