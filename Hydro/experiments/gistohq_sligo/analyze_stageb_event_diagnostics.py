#!/usr/bin/env python3
"""Event diagnostics for the Stage B matched-control-vs-multiscale comparison
(improvement_loop_20261023), per CLAUDE_SUPERVISOR_TASKS.md's 2026-11-16
action checklist item 2's last remaining piece: "event peak-timing/magnitude
error". Per the user's 2026-11-17 direction: events are defined ONCE from
observed discharge and the resulting fixed windows are applied IDENTICALLY
to both arms -- never detected per-arm, which would make the comparison
apples-to-oranges. Reports peak magnitude, peak timing, event volume, and
event-window error, explicitly including missed peaks (disclosed as a
percentage, not merely a binary flag).

Event-detection methodology is adapted from the existing precedent in this
program (analyze_failure_atlas.py's event_cores/expanded_events/
event_metrics, used for the regional-pretraining event decomposition --
reports 51/75/76), with one change: the detection threshold is the SAME
frozen, training-derived Q90 threshold already computed and tested in
analyze_stageb_highflow_tail_diagnostics.py (report 96), not a new
arbitrary quantile -- this ties the two diagnostics to one shared,
already-verified definition of "high flow" rather than introducing a
second, inconsistent one.

Because events are built from the native-matched x set (not every hour in
the raw predictions.csv -- some hours are excluded as non-native per
report 94/checkpoint66), an event WINDOW is an x-range; the rows actually
scored inside it are whichever native-matched x values fall in that range,
and the count is disclosed per event rather than assumed to be contiguous.

Reuses compute_stageb_matched_support_metrics.py's single verified matched-
row path (get_native_matched_rows). Reads only already-exported artifacts;
starts no training; never reads an x>=77016 row.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import compute_stageb_matched_support_metrics as base
from analyze_stageb_highflow_tail_diagnostics import _quantile, _training_observed_by_x

EVENT_MERGE_GAP_HOURS = 24
EVENT_PAD_BEFORE_HOURS = 24
EVENT_PAD_AFTER_HOURS = 48
PEAK_MATCH_RADIUS_HOURS = 12
# A predicted peak at least this far (percent) below the observed peak,
# within the matching radius, is reported as a missed peak. Disclosed
# alongside the raw percentage, not used to hide it.
MISSED_PEAK_ERROR_PERCENT_THRESHOLD = -50.0


def event_cores(observed_by_x: Dict[float, float], threshold: float, merge_gap: float) -> List[Tuple[float, float]]:
    xs = sorted(x for x, v in observed_by_x.items() if v >= threshold)
    if not xs:
        return []
    groups: List[List[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= merge_gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [(g[0], g[-1]) for g in groups]


def expanded_events(cores: List[Tuple[float, float]], before: float, after: float) -> List[Tuple[float, float]]:
    expanded: List[List[float]] = []
    for start, end in cores:
        candidate = [start - before, end + after]
        if expanded and candidate[0] <= expanded[-1][1]:
            expanded[-1][1] = max(expanded[-1][1], candidate[1])
        else:
            expanded.append(candidate)
    return [(s, e) for s, e in expanded]


def event_arm_metrics(xs: List[float], obs: List[float], pred: List[float], *, peak_match_radius_hours: float) -> dict:
    """xs/obs/pred must already be restricted to one event window and
    sorted by x. Peak timing/magnitude are matched locally (the observed
    peak's position, then the best-matching predicted local max within
    peak_match_radius_hours of it) so a preceding, unrelated pulse inside a
    wide padded window cannot create a meaningless multi-day timing error --
    same discipline as analyze_failure_atlas.py's event_metrics."""
    n = len(xs)
    peak_obs_i = max(range(n), key=lambda i: obs[i])
    candidates = [i for i in range(n) if abs(xs[i] - xs[peak_obs_i]) <= peak_match_radius_hours]
    peak_pred_i = max(candidates, key=lambda i: pred[i])
    observed_peak = obs[peak_obs_i]
    predicted_peak = pred[peak_pred_i]
    peak_magnitude_error_percent = (
        100.0 * (predicted_peak - observed_peak) / observed_peak if observed_peak != 0 else None
    )
    missed_peak = (
        peak_magnitude_error_percent is not None
        and peak_magnitude_error_percent <= MISSED_PEAK_ERROR_PERCENT_THRESHOLD
    )
    sse = sum((o - p) ** 2 for o, p in zip(obs, pred))
    observed_volume = sum(obs)
    predicted_volume = sum(pred)
    event_volume_error_percent = (
        100.0 * (predicted_volume - observed_volume) / observed_volume if observed_volume != 0 else None
    )
    return {
        "n": n,
        "rmse": math.sqrt(sse / n) if n else None,
        "observed_peak": observed_peak,
        "predicted_peak": predicted_peak,
        "peak_magnitude_error_percent": peak_magnitude_error_percent,
        "peak_timing_error_hours": xs[peak_pred_i] - xs[peak_obs_i],
        "missed_peak": missed_peak,
        "observed_event_volume": observed_volume,
        "predicted_event_volume": predicted_volume,
        "event_volume_error_percent": event_volume_error_percent,
    }


def event_diagnostics(
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
            raise base.MatchedSupportError(f"fold {year}: train_end_x differs between seeds")

    first_seed = base.SEEDS[0]
    matched_x = seed_rows[first_seed]["matched_x"]
    for seed in base.SEEDS[1:]:
        if seed_rows[seed]["matched_x"] != matched_x:
            raise base.MatchedSupportError(
                f"fold {year}: matched_x differs between seed {first_seed} and seed {seed} -- event "
                "windows must be defined on an identical matched-x set across seeds"
            )

    # Same frozen, training-derived Q90 threshold as report 96 -- not a new,
    # independently-chosen quantile.
    training_by_seed = {}
    for seed in base.SEEDS:
        control_dir = base.STAGE_B_OUTPUT / f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}"
        training_by_seed[seed] = _training_observed_by_x(control_dir, train_end_x, label=f"{year}/s{seed}")
    for seed in base.SEEDS[1:]:
        if training_by_seed[seed] != training_by_seed[first_seed]:
            raise base.MatchedSupportError(
                f"fold {year}: training-split observed discharge differs between seed {first_seed} "
                f"and seed {seed}"
            )
    q90 = _quantile(sorted(training_by_seed[first_seed].values()), 0.90)

    # Events defined ONCE from observed discharge on the native-matched x
    # set -- observed values are shared/verified-identical between arms by
    # construction (get_native_matched_rows' own observed-agreement check),
    # so the resulting windows apply identically to control and multiscale.
    observed_by_x = dict(zip(matched_x, seed_rows[first_seed]["obs"]))
    cores = event_cores(observed_by_x, q90, EVENT_MERGE_GAP_HOURS)
    windows = expanded_events(cores, EVENT_PAD_BEFORE_HOURS, EVENT_PAD_AFTER_HOURS)

    index_of = {x: i for i, x in enumerate(matched_x)}
    events = []
    for event_no, (win_start, win_end) in enumerate(windows, start=1):
        window_x = [x for x in matched_x if win_start <= x <= win_end]
        if not window_x:
            continue
        indices = [index_of[x] for x in window_x]
        per_seed = {}
        for seed in base.SEEDS:
            r = seed_rows[seed]
            obs = [r["obs"][i] for i in indices]
            pred_c = [r["control_pred"][i] for i in indices]
            pred_m = [r["multiscale_pred"][i] for i in indices]
            per_seed[seed] = {
                "control": event_arm_metrics(window_x, obs, pred_c, peak_match_radius_hours=PEAK_MATCH_RADIUS_HOURS),
                "multiscale": event_arm_metrics(window_x, obs, pred_m, peak_match_radius_hours=PEAK_MATCH_RADIUS_HOURS),
            }
        events.append({
            "event_no": event_no,
            "window_start_x": win_start,
            "window_end_x": win_end,
            "n_matched_in_window": len(window_x),
            "per_seed": per_seed,
        })

    return {
        "fold": year,
        "D": d,
        "q90_threshold_mm_h": q90,
        "n_events": len(events),
        "matched_x_sha256": seed_rows[first_seed]["matched_x_sha256"],
        "events": events,
    }


def main(split: str = "test", out_path: Optional[Path] = None, *, allow_overwrite: bool = False) -> int:
    if split not in base.ALLOWED_SPLITS:
        raise base.MatchedSupportError(f"split {split!r} is not one of {base.ALLOWED_SPLITS!r}")

    frozen_manifest_hashes = base._parse_frozen_manifest_hashes(base.FROZEN_MANIFEST_HASHES)
    native_support = base._load_native_discharge_support()

    folds = [
        event_diagnostics(year, d, split, frozen_manifest_hashes, native_support)
        for year, d in base.RESOLVED_D.items()
    ]

    if out_path is None:
        out_path = Path(
            "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/"
            f"multiscale_history_length_selection/stage_b_event_diagnostics_{split}.json"
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
            "event_definition": (
                "contiguous (merge gap <=24h) runs of native-matched observed discharge >= the "
                "fold's frozen training-derived Q90 threshold (same threshold as report 96), "
                "padded 24h before / 48h after, defined ONCE from observed discharge and applied "
                "identically to both arms"
            ),
            "peak_matching": (
                f"observed peak located first; predicted peak is the local max within "
                f"{PEAK_MATCH_RADIUS_HOURS}h of the observed peak's x, not the window-wide max"
            ),
            "missed_peak_definition": (
                f"peak_magnitude_error_percent <= {MISSED_PEAK_ERROR_PERCENT_THRESHOLD} "
                "(predicted peak at least 50% below observed peak); the raw percentage is always "
                "reported alongside the boolean flag"
            ),
            "window_is_an_x_range": (
                "events are windows over x, not a guaranteed-contiguous hourly run -- the actual "
                "scored rows are whichever native-matched x values fall in the window, with "
                "n_matched_in_window disclosed per event"
            ),
        },
        "folds": folds,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test"))
