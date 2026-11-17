#!/usr/bin/env python3
"""Per-year and cross-seed stability diagnostics for the Stage B
matched-control-vs-multiscale comparison (improvement_loop_20261023),
per CLAUDE_SUPERVISOR_TASKS.md's 2026-11-16 action checklist item 2's
"not yet done" list: "per-year breakdown within each 6-year test window,
cross-seed stability statistics". Reads only already-exported artifacts
(predictions.csv, boundary_provenance.json sidecars, dataset_manifest.json);
starts no training and touches no x>=77016 row.

Reuses compute_stageb_matched_support_metrics.py's single verified
matched-row path (get_native_matched_rows) rather than re-implementing
sidecar/manifest/identity verification or the native-support restriction --
every number here is built on exactly the same matched-support definition
as the published stage_b_matched_support_metrics_{split}.json.

Per-year binning: x (an integer hour-index) is converted to a calendar year
via x_to_year(), using STUDY_START_UTC = 2015-01-01T00:00:00Z as x=0 --
confirmed identical across all 18 Stage B job directories' own
dataset_manifest.json, and cross-checked against each fold's own
test_end_x landing exactly on that fold's name-year (see
compute_stageb_matched_support_metrics.STUDY_START_UTC's docstring for the
exact evidence). A calendar year's matched-x set is confirmed identical
across all 3 seeds within a fold before any per-year metric is computed
(structural: fold bounds/native-support/row-eligibility do not depend on
seed) -- if that assumption is ever violated by real data, this raises
rather than silently computing something not comparable across seeds.

Per-year bins can be small or degenerate (e.g. a year with a short partial
window, or a bin whose summed observed discharge is ~0); metrics that would
raise MatchedSupportError (zero NSE/KGE denominator, zero PBIAS denominator)
are recorded as {"undefined": true, "reason": ...} for that bin instead of
aborting the whole report, since an undefined per-year metric is an expected
possibility, not evidence of a bug.
"""
from __future__ import annotations

import datetime
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

import compute_stageb_matched_support_metrics as base

_STUDY_START = datetime.datetime.fromisoformat(base.STUDY_START_UTC.replace("Z", "+00:00"))

_METRIC_KEYS = (
    "nse", "kge", "rmse_mm_h", "rmse_m3_s", "mae_mm_h", "mae_m3_s",
    "pbias_signed_percent", "pbias_absolute_percent",
)


def x_to_year(x: float) -> int:
    """x is an integer hour-index from STUDY_START_UTC (x=0); see module
    docstring for the independent evidence this mapping is correct."""
    return (_STUDY_START + datetime.timedelta(hours=x)).year


def _safe_metrics(obs: List[float], pred: List[float]) -> dict:
    """Same metrics as all_metrics(), but a degenerate bin (zero NSE/KGE
    variance, zero PBIAS denominator -- both real possibilities for a small
    per-year slice) is disclosed as undefined rather than raising and
    aborting the whole report."""
    try:
        metrics = base.all_metrics(obs, pred)
        metrics["undefined"] = False
        return metrics
    except base.MatchedSupportError as exc:
        return {"undefined": True, "reason": str(exc), "n": len(obs)}


def per_year_diagnostics(
    year: str, d: int, split: str, frozen_manifest_hashes: Dict[str, str], native_support: set,
) -> dict:
    seed_rows = {}
    for seed in base.SEEDS:
        control_dir = base.STAGE_B_OUTPUT / f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}"
        multiscale_dir = base.STAGE_B_OUTPUT / f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}"
        seed_rows[seed] = base.get_native_matched_rows(
            control_dir, multiscale_dir, split=split, expected_d=d, expected_seed=int(seed),
            frozen_manifest_hashes=frozen_manifest_hashes, native_support=native_support,
            label=f"{year}/s{seed}", expected_year=year,
        )

    first_seed = base.SEEDS[0]
    matched_x = seed_rows[first_seed]["matched_x"]
    for seed in base.SEEDS[1:]:
        if seed_rows[seed]["matched_x"] != matched_x:
            raise base.MatchedSupportError(
                f"fold {year}: matched_x differs between seed {first_seed} and seed {seed} -- "
                "per-year/cross-seed diagnostics assume an identical matched-x set across seeds "
                "within a fold (fold bounds/native-support/row-eligibility are seed-independent); "
                "this assumption just broke on real data and must be investigated, not silently "
                "worked around by scoring each seed on its own different set"
            )

    index_of = {x: i for i, x in enumerate(matched_x)}
    calendar_years = sorted(set(x_to_year(x) for x in matched_x))

    per_year = {}
    for cal_year in calendar_years:
        year_x = [x for x in matched_x if x_to_year(x) == cal_year]
        indices = [index_of[x] for x in year_x]
        per_seed = {}
        for seed in base.SEEDS:
            r = seed_rows[seed]
            obs = [r["obs"][i] for i in indices]
            pred_c = [r["control_pred"][i] for i in indices]
            pred_m = [r["multiscale_pred"][i] for i in indices]
            per_seed[seed] = {
                "n_matched": len(year_x),
                "control": _safe_metrics(obs, pred_c),
                "multiscale": _safe_metrics(obs, pred_m),
            }
        per_year[str(cal_year)] = per_seed

    cross_seed = {}
    for arm, pred_key in (("control", "control_pred"), ("multiscale", "multiscale_pred")):
        arm_metrics_by_seed = {
            seed: base.all_metrics(seed_rows[seed]["obs"], seed_rows[seed][pred_key])
            for seed in base.SEEDS
        }
        cross_seed[arm] = {
            key: {
                "per_seed": {seed: arm_metrics_by_seed[seed][key] for seed in base.SEEDS},
                "mean": statistics.mean(arm_metrics_by_seed[seed][key] for seed in base.SEEDS),
                "stdev": statistics.stdev(arm_metrics_by_seed[seed][key] for seed in base.SEEDS),
            }
            for key in _METRIC_KEYS
        }
    delta_by_seed = {
        seed: {
            key: (
                base.all_metrics(seed_rows[seed]["obs"], seed_rows[seed]["multiscale_pred"])[key]
                - base.all_metrics(seed_rows[seed]["obs"], seed_rows[seed]["control_pred"])[key]
            )
            for key in _METRIC_KEYS
        }
        for seed in base.SEEDS
    }
    cross_seed["delta_multiscale_minus_control"] = {
        key: {
            "per_seed": {seed: delta_by_seed[seed][key] for seed in base.SEEDS},
            "mean": statistics.mean(delta_by_seed[seed][key] for seed in base.SEEDS),
            "stdev": statistics.stdev(delta_by_seed[seed][key] for seed in base.SEEDS),
        }
        for key in _METRIC_KEYS
    }

    return {
        "fold": year,
        "D": d,
        "n_matched_total": len(matched_x),
        "calendar_years_present": [str(y) for y in calendar_years],
        "matched_x_sha256": seed_rows[first_seed]["matched_x_sha256"],
        "per_year": per_year,
        "cross_seed_stability": cross_seed,
    }


def main(split: str = "test", out_path: Optional[Path] = None, *, allow_overwrite: bool = False) -> int:
    frozen_manifest_hashes = base._parse_frozen_manifest_hashes(base.FROZEN_MANIFEST_HASHES)
    native_support = base._load_native_discharge_support()

    folds = [
        per_year_diagnostics(year, d, split, frozen_manifest_hashes, native_support)
        for year, d in base.RESOLVED_D.items()
    ]

    if out_path is None:
        out_path = Path(
            "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/"
            f"multiscale_history_length_selection/stage_b_year_and_seed_diagnostics_{split}.json"
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
            "x_to_calendar_year": (
                f"x is an integer hour-index from study_start={base.STUDY_START_UTC} "
                "(confirmed identical across all 18 Stage B job dirs' dataset_manifest.json); "
                "calendar year = (study_start + x hours).year"
            ),
            "matched_x_identical_across_seeds_within_fold": (
                "asserted at runtime, not assumed -- raises MatchedSupportError if violated"
            ),
            "undefined_metric_handling": (
                "a per-year bin whose NSE/KGE denominator or PBIAS denominator is zero is "
                "recorded as {'undefined': true, 'reason': ...} rather than aborting the report"
            ),
            "cross_seed_stability": "mean/stdev (n=3 seeds) of each metric per arm, and of the "
            "per-seed delta (multiscale - control), computed on the fold's full native-matched "
            "test support (not per-year)",
        },
        "catchment_area_km2": base.CATCHMENT_AREA_KM2,
        "folds": folds,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test"))
