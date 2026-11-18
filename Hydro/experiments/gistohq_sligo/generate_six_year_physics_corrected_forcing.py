#!/usr/bin/env python3
"""Builds the 36 six-year physics confirmation configs for the
INTERVAL-CORRECTED forcing package.

Method: copy the literal, already-run original-forcing config for each
(year, arm, seed=42) as the base template -- not a re-derivation from
the annual manifest -- so every architecture/hyperparameter field
(epochs, lstm_sequence_length, lambda_decay, lstm_dropout,
antecedent_index_halflife_hours, use_mrms_spatial_cv_feature, the
train/validation/test_end_x bounds, etc.) is preserved byte-for-byte.
Then override ONLY:
  - experiment_id
  - hydro_package_path (-> interval-corrected package)
  - random_seed (for the seed=123/27182 variants; 42's seed is unchanged)
  - the 7 recalibrated urban_* physics fields + pinn_routing_lag_hours,
    taken from this run's own corrected-forcing calibration
    (six_year_physics_calibration_corrected_forcing/selected_params.json),
    NOT reused from the original-forcing calibration.
  - urban_detention_fraction stays 0 for both arms, exactly as the
    original generator set it (the "no-detention" convention is
    preserved, not recalibrated).

A programmatic diff against the literal original seed=42 config is
written for every (year, arm) pair, asserting no other field changed.
Original run directories (six_year_physics_runs/) are only read, never
written.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path

ORIGINAL_RUNS = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "improvement_loop_20261023/six_year_physics_runs"
)
CALIBRATION_CORRECTED = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "improvement_loop_20261023/six_year_physics_calibration_corrected_forcing/selected_params.json"
)
CORRECTED_PACKAGE_PATH = (
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_interval_corrected_devperiod/hydropinn"
)
OUT_DIR = Path(__file__).resolve().parent / "improvement_loop_20261023" / "six_year_physics_configs_corrected_forcing"
DIFF_OUT = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "improvement_loop_20261023/component_diagnostics/six_year_physics_corrected_forcing_config_diff.json"
)
YEARS = ["2018", "2019", "2020", "2021", "2022", "2023"]
ARMS = ["zero", "urban_no_detention"]
SEEDS = [42, 123, 27182]
SEAL_START_X = 77016

EXPECTED_CHANGED_FIELDS = {
    "experiment_id", "hydro_package_path", "random_seed",
    "urban_quick_k", "urban_slow_k", "urban_detention_k", "urban_channel_k",
    "urban_quick_fraction", "urban_runoff_coefficient", "pinn_routing_lag_hours",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    params = json.loads(CALIBRATION_CORRECTED.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_diffs = {}
    jobs = []

    for year in YEARS:
        p = params[year]["managed"]
        for arm in ARMS:
            orig_s42_path = (ORIGINAL_RUNS / f"sixyear_{year}_{arm}_s42" /
                              f"sixyear_{year}_{arm}_s42" / "experiment_config.json")
            orig_template = json.loads(orig_s42_path.read_text())
            if orig_template["test_end_x"] >= SEAL_START_X:
                raise RuntimeError(f"refusing: {year}/{arm} test_end_x reaches seal")

            for seed in SEEDS:
                name = f"sixyear_corrected_forcing_{year}_{arm}_s{seed}"
                cfg = dict(orig_template)
                cfg.update(
                    experiment_id=name,
                    hydro_package_path=CORRECTED_PACKAGE_PATH,
                    random_seed=seed,
                    urban_quick_k=p["quick_k"],
                    urban_slow_k=p["slow_k"],
                    urban_detention_k=p["detention_k"],
                    urban_channel_k=p["channel_k"],
                    urban_quick_fraction=p["quick_fraction"],
                    urban_detention_fraction=0.0,
                    urban_runoff_coefficient=p["runoff_coefficient"],
                    pinn_routing_lag_hours=float(p["lag_steps"]),
                )

                out_path = OUT_DIR / f"{name}.json"
                if out_path.exists():
                    raise RuntimeError(f"refusing to overwrite existing config {out_path}")
                out_path.write_text(json.dumps(cfg, indent=2) + "\n")
                batch_path = out_path.with_suffix(".batch")
                batch_path.write_text(f"lstm_pinn {out_path}\n")

                # Programmatic diff vs the literal original seed=42 artifact for this year/arm
                diff = {}
                all_keys = set(orig_template) | set(cfg)
                for k in sorted(all_keys):
                    if orig_template.get(k) != cfg.get(k):
                        diff[k] = {"original": orig_template.get(k), "corrected_forcing": cfg.get(k)}
                unexpected = set(diff) - EXPECTED_CHANGED_FIELDS
                if unexpected:
                    raise RuntimeError(f"{name}: unexpected field changes vs original template: {unexpected}")
                all_diffs[name] = diff

                jobs.append({
                    "id": name, "year": year, "arm": arm, "seed": seed,
                    "config": str(out_path), "config_sha256": sha(out_path),
                    "batch": str(batch_path),
                    "original_template_path": str(orig_s42_path),
                    "original_template_sha256": sha(orig_s42_path),
                })

    manifest_path = OUT_DIR / "run_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite existing manifest {manifest_path}")
    manifest_path.write_text(json.dumps({
        "purpose": "Six-year expanding-window common-scale physics evaluation, interval-corrected MRMS forcing",
        "selection_source": str(CALIBRATION_CORRECTED),
        "selection_sha256": sha(CALIBRATION_CORRECTED),
        "package_path": CORRECTED_PACKAGE_PATH,
        "epochs": 150,
        "seeds": SEEDS,
        "jobs": jobs,
    }, indent=2) + "\n")

    DIFF_OUT.parent.mkdir(parents=True, exist_ok=True)
    DIFF_OUT.write_text(json.dumps({
        "expected_changed_fields": sorted(EXPECTED_CHANGED_FIELDS),
        "per_job_diff_vs_original_seed42_template": all_diffs,
    }, indent=2, sort_keys=True))

    print(f"Wrote {len(jobs)} configs to {OUT_DIR}")
    print(f"Wrote diff record to {DIFF_OUT}")
    # Print one example diff for a quick sanity read
    example = next(iter(all_diffs))
    print(f"\nExample diff ({example}):")
    print(json.dumps(all_diffs[example], indent=2))


if __name__ == "__main__":
    main()
