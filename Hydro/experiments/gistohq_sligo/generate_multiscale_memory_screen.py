#!/usr/bin/env python3
"""Builds the frozen manifest for the multiscale-memory screen
(improvement_loop_20261023 reports 66-68). NOT launched by this script or
by running it -- it only writes config/batch files to disk.

Three fixed development folds, matching this program's own established
precedent (report 07's finalist comparison): 2018, 2020, 2023, each an
expanding-window fold whose train/validation/test_end_x boundaries and
hydro_package_path are copied LITERALLY from the already-frozen
sixyear_corrected_forcing_{year}_zero_s42 config for that year -- not
re-derived -- so the new screen's chronological split is provably identical
to this program's existing, already-reviewed split for that fold.

Two phases, budgeted separately (report 68):
  1. Selection (9 jobs): 3 folds x 3 daily-history-length candidates
     {14, 30, 60} x seed 42 only. Scored on each fold's OWN inner/
     validation-split NSE, never on its test split or the other seeds --
     the output of this phase (one winning D per fold) is a required
     INPUT to phase 2's multiscale-arm configs, which this script does
     NOT yet have (selection has not been run) and therefore does not
     fabricate. Phase 2's multiscale-arm configs are written with
     multiscale_daily_history_days = -1, a value experiment_loader.cpp
     itself rejects (">= 0" required) -- a deliberate, enforced "not
     runnable yet" sentinel, not an oversight.
  2. Comparison (18 jobs): 2 arms x 3 folds x 3 seeds.
     - Control: retrained fresh under the CURRENT executable (not reusing
       the original frozen baseline's checkpoint) so both arms in this
       paired comparison share one executable build -- a new experiment_id
       (suffix _multiscale_control_retrain), never overwriting or editing
       any frozen original job directory. multiscale_matched_control=True
       and the SAME per-fold D sentinel as its paired multiscale job below
       are both set (improvement_loop_20261023 report 72, fixing a gap
       found on user review: an earlier version of this script left both
       unset, so a regenerated control config would have trained/scored
       on every window instead of the eligible subset the multiscale arm
       requires, defeating report 71's matched-control design).
     - Multiscale: same split/package, approach name "lstm_multiscale",
       daily-history fields added; D is the phase-1 sentinel above.

Architecture (24h hourly LSTM, hidden=32, 1 layer; daily branch hidden=4,
2 features) and capacity-control decision (no third arm; delta +448 params
/ +8.7%, disclosed not controlled for) are both fixed per report 67/68 --
not re-derived here.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path

ORIGINAL_RUNS = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "improvement_loop_20261023/six_year_physics_runs_corrected_forcing"
)
OUT_DIR = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "improvement_loop_20261023/multiscale_memory_screen_manifests"
)
FOLD_YEARS = ["2018", "2020", "2023"]
SELECTION_D_CANDIDATES = [14, 30, 60]
COMPARISON_SEEDS = [42, 123, 27182]
DAILY_HIDDEN_SIZE = 4
SEAL_START_X = 77016
UNRESOLVED_D_SENTINEL = -1  # rejected by experiment_loader.cpp -- not runnable until replaced


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fold_template(year: str) -> dict:
    path = ORIGINAL_RUNS / f"sixyear_corrected_forcing_{year}_zero_s42" / "experiment_config.json"
    template = json.loads(path.read_text())
    if template["test_end_x"] >= SEAL_START_X:
        raise RuntimeError(f"refusing: {year} test_end_x reaches the reserved-period seal")
    return template


def write_config(out_dir: Path, name: str, cfg: dict, approach: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    if out_path.exists():
        raise RuntimeError(f"refusing to overwrite existing config {out_path}")
    out_path.write_text(json.dumps(cfg, indent=2) + "\n")
    (out_dir / f"{name}.batch").write_text(f"{approach} {out_path}\n")
    return out_path


def main() -> None:
    manifest = {"selection_jobs": [], "comparison_control_jobs": [], "comparison_multiscale_jobs": []}

    # Phase 1: selection, 9 jobs.
    selection_dir = OUT_DIR / "phase1_selection"
    for year in FOLD_YEARS:
        template = load_fold_template(year)
        for d in SELECTION_D_CANDIDATES:
            name = f"multiscale_selection_{year}_D{d}_s42"
            cfg = dict(template)
            cfg.update(
                experiment_id=name,
                random_seed=42,
                multiscale_daily_history_days=d,
                multiscale_daily_hidden_size=DAILY_HIDDEN_SIZE,
            )
            path = write_config(selection_dir, name, cfg, "lstm_multiscale")
            manifest["selection_jobs"].append({
                "fold_year": year, "daily_history_days": d, "seed": 42,
                "config_path": str(path), "config_sha256": sha(path),
                "scored_on": "training-internal/validation-split NSE only (never test split)",
            })

    # Phase 2a: comparison control arm, 9 jobs -- retrained fresh, new experiment_id.
    # multiscale_matched_control=True + the same unresolved D sentinel as the
    # paired multiscale job (set together, below) -- both must be replaced
    # with the same resolved per-fold D before either arm is runnable
    # (report 72). Checkpoint format/approach are unchanged ("lstm_pinn" /
    # torch-module-v1): the matched-control arm is still a plain HydroLSTM.
    control_dir = OUT_DIR / "phase2_comparison" / "control"
    for year in FOLD_YEARS:
        template = load_fold_template(year)
        for seed in COMPARISON_SEEDS:
            name = f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}"
            cfg = dict(template)
            cfg.update(
                experiment_id=name,
                random_seed=seed,
                multiscale_daily_history_days=UNRESOLVED_D_SENTINEL,
                multiscale_daily_hidden_size=DAILY_HIDDEN_SIZE,
                multiscale_matched_control=True,
            )
            path = write_config(control_dir, name, cfg, "lstm_pinn")
            manifest["comparison_control_jobs"].append({
                "fold_year": year, "seed": seed, "config_path": str(path), "config_sha256": sha(path),
                "multiscale_matched_control": True,
                "multiscale_daily_history_days": "UNRESOLVED -- set from phase 1's winning D for this "
                    "fold_year, IDENTICAL to the paired multiscale job's own D, before use",
            })

    # Phase 2b: comparison multiscale arm, 9 jobs -- D left as an unresolved,
    # enforced-invalid sentinel pending phase 1's actual results.
    multiscale_dir = OUT_DIR / "phase2_comparison" / "multiscale"
    for year in FOLD_YEARS:
        template = load_fold_template(year)
        for seed in COMPARISON_SEEDS:
            name = f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}"
            cfg = dict(template)
            cfg.update(
                experiment_id=name,
                random_seed=seed,
                multiscale_daily_history_days=UNRESOLVED_D_SENTINEL,
                multiscale_daily_hidden_size=DAILY_HIDDEN_SIZE,
            )
            path = write_config(multiscale_dir, name, cfg, "lstm_multiscale")
            manifest["comparison_multiscale_jobs"].append({
                "fold_year": year, "seed": seed, "config_path": str(path), "config_sha256": sha(path),
                "multiscale_daily_history_days": "UNRESOLVED -- set from phase 1's winning D for this fold_year before use",
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total = len(manifest["selection_jobs"]) + len(manifest["comparison_control_jobs"]) + len(manifest["comparison_multiscale_jobs"])
    print(f"Wrote {total} config/batch pairs ({len(manifest['selection_jobs'])} selection + "
          f"{len(manifest['comparison_control_jobs'])} control + {len(manifest['comparison_multiscale_jobs'])} multiscale) "
          f"and manifest.json to {OUT_DIR}")
    print("No job executed. Phase 2's multiscale-arm D fields are an enforced-invalid sentinel until phase 1 runs.")


if __name__ == "__main__":
    main()
