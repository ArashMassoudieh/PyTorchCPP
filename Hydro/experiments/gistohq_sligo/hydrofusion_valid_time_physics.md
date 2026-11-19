# Causal physical timing v3

Opt in with `physics_profile: causal_valid_time_dry_v3` on physical-only,
FFN-physical or LSTM-physical forecast modes. Legacy v1/v2 profiles and recipe
bytes remain unchanged. v3 uses a separately identified
`hydrofusion-forecast-v3` archive and bound recipe; old models cannot silently
be interpreted using the new reference.

The native adapter preserves rainfall interval end and declared release hour
on each HourlyRow without changing existing numerical features. At origin o,
rain ending at t=o-delay describes [t-1,t), not rain occurring at o.
Only released observations enter committed reservoir state. Each newly
released interval advances that historical state exactly once. State is then
copied, advanced to o with zero assumed unobserved effective rain, and hourly
mean outflow over [o,o+1) predicted with the same explicit dry assumption.
Dry projection is never committed: later releases supply actual historical
forcing rather than advancing the projected state a second time.

This is a dry-gap policy, not proof that unobserved hours were dry or a rainfall
nowcast. With 24-hour latency it will miss storms whose rainfall is unavailable.
The fix removes false late storm injection; it cannot create missing predictive
information or guarantee higher skill. The PET estimate available at release
is used as the effective-rain loss estimate, not asserted to be realized ET
for that historical interval. No future PET/current-day totals, observed-Q
initialization, Q assimilation or fitted runoff coefficient are introduced.

Requirements: presealed origin AND target; explicit valid/release identities;
release equals origin; delay0..168h; hourly valid-time progression within
contiguous available segments; finite nonnegative rain/PET. Missing available
forcing or origin gaps reset historical state. Changed delay within a segment
is rejected. Raw-row inference callers must preserve trusted adapter provenance.

The same profile implementation is used for training and inference. Existing
model/scaler/config/native-manifest bindings remain enforced. No research fits
are authorized by this implementation; old neural weights cannot be reused
under a changed reference as a new scientific result.

Regression coverage: delayed impulse analytical response, future/prefix
invariance, independence of observed Q, missing/reordered identities,
nonfinite forcing, gaps, reserved targets, zero-delay next-hour semantics,
ledger closure, old v1/v2 regressions and native adapter metadata propagation.
The independent audit records actual executed production checks and limits.
