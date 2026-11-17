#!/usr/bin/env python3
"""High-flow/tail diagnostics for the Stage B matched-control-vs-multiscale
comparison (improvement_loop_20261023), per CLAUDE_SUPERVISOR_TASKS.md's
2026-11-16 action checklist item 2's "not yet done" list: "tail/extreme-flow
RMSE". Folds in the high/low-flow split idea from Tan et al. 2026
(Multi-Objective Hyperparameter Optimization..., DOI 10.3390/
hydrology13080218) per CODEX_SUPERVISION_LOG.md's 2026-11-16 review note --
their paper's "low" category is the remaining 90% of flows, NOT a low-flow
tail, and their SHAP explains hyperparameters, not met-input attributions;
neither of those is reused here, only the high/remaining split concept.

Threshold definition (frozen, not retrospective): Q90 is computed from each
fold's own TRAINING-split observed discharge ONLY (never test or
validation), via the same fail-closed row-reading path as the rest of this
program. This threshold is fixed before any test-split row is read, so
"high-flow" is not defined by looking at the data being scored.

Reuses compute_stageb_matched_support_metrics.py's single verified matched-
row path (get_native_matched_rows) for the actual test-split comparison --
every number here is built on exactly the same matched-support definition
as the published stage_b_matched_support_metrics_test.json. Confirms the
training-split observed discharge is identical across all 3 seeds within a
fold before computing one threshold for all three (checked, not assumed).

For each fold/seed pair, partitions the native-matched TEST rows into
high-flow (observed >= Q90) and remaining (observed < Q90), scores RMSE/NSE
on each partition separately, and verifies SSE(high) + SSE(remaining) ==
SSE(total) to within floating-point tolerance -- an exact arithmetic
identity for a correct, non-overlapping, exhaustive partition; any
violation would indicate a real bug (double-counted or dropped rows), not a
modeling choice.

Reads only already-exported artifacts; starts no training; never reads an
x>=77016 row (training split is always well before any fold's seal).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import compute_stageb_matched_support_metrics as base


def _quantile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolation-between-closest-ranks quantile (same convention
    as numpy's default 'linear' method), spelled out explicitly here so the
    frozen Q90 threshold is reproducible without depending on numpy."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _training_observed_by_x(control_dir: Path, train_end_x: float, *, label: str) -> Dict[float, float]:
    """Training-split observed discharge, read via the same fail-closed
    path as everything else in this program, bounded by the ALREADY-
    VERIFIED train_end_x (obtained by the caller from get_native_matched_rows'
    sidecar verification, not re-derived here)."""
    rows = base.read_split_rows_fail_closed(
        control_dir / "predictions.csv", "lstm_pinn", split="train",
        interval_start_x=-1.0, interval_end_x=train_end_x,
    )
    if not rows:
        raise base.MatchedSupportError(f"{label}: no training rows found to define a Q90 threshold")
    return {x: v[0] for x, v in rows.items()}


def _sse(observed: List[float], predicted: List[float]) -> float:
    return sum((o - p) ** 2 for o, p in zip(observed, predicted))


def _partition_metrics(observed: List[float], predicted: List[float]) -> dict:
    if not observed:
        return {"n": 0}
    result = {"n": len(observed), "sse": _sse(observed, predicted)}
    try:
        result["rmse"] = base.rmse(observed, predicted)
    except base.MatchedSupportError:
        pass
    try:
        result["nse"] = base.nse(observed, predicted)
    except base.MatchedSupportError as exc:
        result["nse_undefined_reason"] = str(exc)
    return result


def high_flow_tail_diagnostics(
    year: str, d: int, split: str, frozen_manifest_hashes: Dict[str, str], native_support: set,
) -> dict:
    seed_rows = {}
    train_end_x: Optional[float] = None
    for seed in base.SEEDS:
        control_dir = base.STAGE_B_OUTPUT / f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}"
        multiscale_dir = base.STAGE_B_OUTPUT / f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}"
        r = base.get_native_matched_rows(
            control_dir, multiscale_dir, split=split, expected_d=d, expected_seed=int(seed),
            frozen_manifest_hashes=frozen_manifest_hashes, native_support=native_support,
            label=f"{year}/s{seed}", expected_year=year,
        )
        seed_rows[seed] = r
        if train_end_x is None:
            train_end_x = r["bounds"]["train_end_x"]
        elif train_end_x != r["bounds"]["train_end_x"]:
            raise base.MatchedSupportError(
                f"fold {year}: train_end_x differs between seeds ({train_end_x} vs "
                f"{r['bounds']['train_end_x']}) -- a single frozen Q90 threshold cannot be shared"
            )

    # Confirm training-split observed discharge is identical across all 3
    # seeds before computing one threshold for all of them -- checked, not
    # assumed (same discipline as the matched-x cross-seed check).
    training_by_seed = {}
    for seed in base.SEEDS:
        control_dir = base.STAGE_B_OUTPUT / f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}"
        training_by_seed[seed] = _training_observed_by_x(control_dir, train_end_x, label=f"{year}/s{seed}")
    first_seed = base.SEEDS[0]
    for seed in base.SEEDS[1:]:
        if training_by_seed[seed] != training_by_seed[first_seed]:
            raise base.MatchedSupportError(
                f"fold {year}: training-split observed discharge differs between seed {first_seed} "
                f"and seed {seed} -- a single frozen Q90 threshold cannot be shared across seeds "
                "whose training targets genuinely differ"
            )
    training_observed = sorted(training_by_seed[first_seed].values())
    q90 = _quantile(training_observed, 0.90)

    per_seed = {}
    for seed in base.SEEDS:
        r = seed_rows[seed]
        matched_x = r["matched_x"]
        obs, pred_c, pred_m = r["obs"], r["control_pred"], r["multiscale_pred"]

        high_idx = [i for i, o in enumerate(obs) if o >= q90]
        remaining_idx = [i for i, o in enumerate(obs) if o < q90]
        if len(high_idx) + len(remaining_idx) != len(obs):
            raise base.MatchedSupportError(f"{year}/s{seed}: high+remaining partition does not cover all rows")

        entry = {"n_total": len(obs), "n_high": len(high_idx), "n_remaining": len(remaining_idx)}
        for arm, pred in (("control", pred_c), ("multiscale", pred_m)):
            obs_high = [obs[i] for i in high_idx]
            pred_high = [pred[i] for i in high_idx]
            obs_rem = [obs[i] for i in remaining_idx]
            pred_rem = [pred[i] for i in remaining_idx]
            sse_total = _sse(obs, pred)
            sse_high = _sse(obs_high, pred_high)
            sse_rem = _sse(obs_rem, pred_rem)
            if not math.isclose(sse_high + sse_rem, sse_total, rel_tol=1e-9, abs_tol=1e-9):
                raise base.MatchedSupportError(
                    f"{year}/s{seed}/{arm}: SSE(high)+SSE(remaining)={sse_high + sse_rem} != "
                    f"SSE(total)={sse_total} -- partition arithmetic is broken"
                )
            entry[arm] = {
                "high": _partition_metrics(obs_high, pred_high),
                "remaining": _partition_metrics(obs_rem, pred_rem),
                "total_sse": sse_total,
                "sse_partition_check_passed": True,
            }
        per_seed[seed] = entry

    return {
        "fold": year,
        "D": d,
        "train_end_x": train_end_x,
        "n_training_rows_for_q90": len(training_observed),
        "q90_threshold_mm_h": q90,
        "matched_x_sha256": seed_rows[first_seed]["matched_x_sha256"],
        "per_seed": per_seed,
    }


def main(split: str = "test", out_path: Optional[Path] = None, *, allow_overwrite: bool = False) -> int:
    if split not in base.ALLOWED_SPLITS:
        raise base.MatchedSupportError(f"split {split!r} is not one of {base.ALLOWED_SPLITS!r}")

    frozen_manifest_hashes = base._parse_frozen_manifest_hashes(base.FROZEN_MANIFEST_HASHES)
    native_support = base._load_native_discharge_support()

    folds = [
        high_flow_tail_diagnostics(year, d, split, frozen_manifest_hashes, native_support)
        for year, d in base.RESOLVED_D.items()
    ]

    if out_path is None:
        out_path = Path(
            "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/"
            f"multiscale_history_length_selection/stage_b_highflow_tail_diagnostics_{split}.json"
        )
    if out_path.exists() and not allow_overwrite:
        raise base.MatchedSupportError(
            f"Refusing to overwrite existing output file {out_path} -- pass an explicit new path, "
            "or allow_overwrite=True if regenerating it deliberately"
        )

    report = {
        "split": split,
        "cli_invocation": f"{Path(__file__).name} {split}",
        "methodology": {
            "q90_definition": (
                "90th percentile (linear interpolation between closest ranks) of each fold's own "
                "TRAINING-split observed discharge only, never test/validation; frozen before any "
                "test-split row is read; confirmed identical across all 3 seeds before use"
            ),
            "high_flow": "observed >= Q90 (mm/h)",
            "remaining": "observed < Q90 (mm/h)",
            "sse_partition_check": "SSE(high) + SSE(remaining) == SSE(total), verified per seed per arm",
            "tan_et_al_2026_relation": (
                "folds in the high/remaining split concept from Tan et al. 2026 "
                "(10.3390/hydrology13080218); their own 'low' category is the remaining 90% of "
                "flows, not a low-flow tail, and their SHAP explains hyperparameters not "
                "meteorological-input attributions -- neither of those is reused here"
            ),
        },
        "folds": folds,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test"))
