# Goal-driven batch controller

The CLI and **Batch > Goal-driven training…** share `controller.py`. This first
version executes a finite, reviewed candidate inventory; it does not invent new
architectures, grow a budget, guarantee a requested score or edit a manuscript.

## Request and execution

A request JSON declares `binary`, `verifier`, `goals`, `limits`, `rule`, and `jobs`.
Every job has `id`, `candidate`, `family`, `fold`, `seed`, `config`, and `support`.
Supported families are `ffn_causal_forecast`, `lstm_hydrofusion`,
`ffn_physical_forecast`, `lstm_physical_forecast`, `pinn_causal_forecast`.
Config files use the existing HydroBatch schema. They must opt into development
only execution with a native package physically cropped at validation end.
A support CSV has `split,target_x,observed,persistence`, with training and
validation only. Matching folds across all candidates/families require identical
support; history candidates must have common support established beforehand.

Goals are separate finite numbers in [0,1], for example
`{"nse": 0.7, "r_squared": 0.9}`. R² means squared Pearson correlation, not NSE.

`limits` explicitly declares epochs, wall_seconds_per_fit, rss_bytes, threads,
gpu (false), artifact_bytes_per_fit and max_fits. The total inventory cannot
exceed max_fits. Thus total fit-time is bounded by max_fits × per-fit wall cap;
verifications have a separate one-hour timeout each. Execution is sequential.
The kernel enforces memory/CPU caps; a watchdog enforces wall and artifact caps.

The fixed `rule` contains min_nse_gain against persistence, max_mae_ratio,
max_high_rmse_ratio, max_abs_bias_increase (percentage points) and
required_fraction of declared fold/seed pairs. Example:

```json
{"min_nse_gain":0.02,"max_mae_ratio":1.02,"max_high_rmse_ratio":1.02,
 "max_abs_bias_increase":2,"required_fraction":1.0}
```

A candidate qualifies only after all its declared pairs complete, both mean
metric goals pass, and the required fraction pass both goals and every guard.
The overall goal is met when each requested family has a qualified candidate.
The physical reference is reported honestly if its goal cannot be reached.
No lucky best seed replaces the declared aggregation.

```bash
python3 Hydro/goal_batch/controller.py prepare /new/run --request request.json
python3 Hydro/goal_batch/controller.py run /new/run
python3 Hydro/goal_batch/controller.py status /new/run
python3 Hydro/goal_batch/controller.py cancel /new/run
python3 Hydro/goal_batch/controller.py revise-goals /new/run --nse .8 --r-squared .9
```

Preparation freezes configs, dispatch batches, native/weather/support/source and
binary hashes. Live source changes invalidate resume: do not edit an active
controller. The plan retains paths to immutable inputs; keep those files.
Goals can be revised only while the controller is idle, creating an append-only
hash chain without modifying the original plan or resetting fit charges.

Run reuses only verified completed jobs, rehashes artifacts and recomputes scores.
An exclusive file lock prevents concurrent controllers. A claimed incomplete or
failed job is never automatically repeated. Cancellation during a fit kills its
process group and leaves its fit charge intact. A pre-fit cancellation consumes
no fit; remove the CANCEL marker explicitly to continue unstarted inventory.
Closing the Qt dialog requests cancellation; application exit also has a Linux
parent-death guard for an active training child.

Terminal states distinguish GOAL_MET, INVENTORY_EXHAUSTED, FIT_BUDGET_EXHAUSTED,
CANCELLED, INTERRUPTED and FAILED. Unreached goals are not success messages.
`report.json` includes separate family/candidate scores and all fold/seed scores,
persistence, high-flow errors and bias; `state.json` records charges and status.
No terminal holdout evaluation or manuscript promotion occurs in this version.

## Verification

```bash
python3 -m unittest discover -s Hydro/goal_batch -p test_controller.py
```

Tests include fabricated orchestration across all five methods, strict goals,
limits, support/test/seal rejection, corruption, exclusive launch, interruption,
repeat prevention and real watchdog cancellation. These are not engine-skill tests.
`prepare_smoke_request.py` constructs a separate real-engine two-epoch smoke
request from the existing explicitly fabricated 220-hour fixture, with no
research observations. It checks only integration, never scientific skill.

## Current limits

The calendar-study wizard now builds a finite inventory from a supported native
package and pre-issue weather sidecar. Choose methods, one year, baseline width32
or comparison widths32/64, epoch cap and goals; it derives chronological support
with the actual feature preflight tool and writes a launchable request. The
physical reference is included once, not retrained for each width. The GUI
prepares only; use Goal-driven training to review/launch the generated request.
This wizard currently uses the fixed2015origin, one seed42, first9months TRAIN
and remaining months validation. CLI also supports a two-year prefix with the
same first9months TRAIN. It does not certify coverage from date labels alone.
Ten-year selection is visibly blocked before any observation access: corrected
input/weather coverage and reserved-data permission remain unresolved. Engine,
native verifier and feature-preflight executables must be selected; installation
packaging/discovery is not complete. Automatic calibration proposals, terminal
holdout promotion, plotting and adaptive/futility pruning are not implemented.
Existing original/frozen campaigns and reserved observations remain untouched.


## Calendar builder CLI

```bash
python3 Hydro/goal_batch/study_builder.py /new/study \
  --manifest /verified/native_input_manifest.json --weather /verified/preissue_rain.csv \
  --binary /tools/HydroBatch --verifier /tools/native_reload --probe /tools/fold_preflight \
  --years 1 --epochs 200 --widths 32,64 --nse .7 --r-squared .8
```

This creates cropped copies, common support, configs, a finite request and a
preflight record. It does not launch fitting or acquire missing inputs. Existing
packages and closed campaigns are preserved. Rainfall release timestamps remain
retrospective archive proxies, not proof of operational availability.
