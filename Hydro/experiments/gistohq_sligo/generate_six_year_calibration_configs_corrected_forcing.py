#!/usr/bin/env python3
"""Builds the six validation-only calibration configs for the
INTERVAL-CORRECTED forcing package, mirroring the original-forcing
calibration config shape exactly (report 22/run_six_year_calibration.py),
with two changes only: hydro_package_path points to the corrected
package, and output goes to a separate, non-overlapping directory tree
so the original-forcing calibration (six_year_physics_calibration/,
immutable, already used for the published six-year decision in reports
24/25) is never touched.

Per-year train/validation/test_end_x bounds are reused unmodified from
the same annual template source generate_six_year_physics.py used
(decade_20261022/annual_configs_v2, lstm_24h model, seed 42), with
test_end_x truncated to validation_end_x+1 for the calibration step
only (validation-only selection; no test data read), exactly as report
22 describes for the original-forcing calibration.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path

base = Path(__file__).resolve().parent
ANNUAL_TEMPLATE_SOURCE = base / "decade_20261022/annual_configs_v2/run_manifest.json"
CORRECTED_PACKAGE_PATH = (
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_interval_corrected_devperiod/hydropinn"
)
OUT_DIR = base / "improvement_loop_20261023/six_year_physics_calibration_configs_corrected_forcing"
SEAL_START_X = 77016


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    annual = json.loads(ANNUAL_TEMPLATE_SOURCE.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for j in annual["jobs"]:
        if j.get("model") != "lstm_24h" or j.get("seed") != 42:
            continue
        year = j["year"]
        template = json.loads(Path(j["config"]).read_text())
        validation_end_x = template["validation_end_x"]
        calib_test_end_x = validation_end_x + 1
        if calib_test_end_x >= SEAL_START_X:
            raise RuntimeError(f"refusing: year {year} calibration test_end_x {calib_test_end_x} reaches seal")

        cfg = dict(template)
        cfg.update(
            experiment_id=f"sixyear_calib_corrected_forcing_{year}",
            hydro_package_path=CORRECTED_PACKAGE_PATH,
            test_end_x=calib_test_end_x,
        )
        out_path = OUT_DIR / f"{year}.json"
        if out_path.exists():
            raise RuntimeError(f"refusing to overwrite existing config {out_path}")
        out_path.write_text(json.dumps(cfg, indent=2) + "\n")

        jobs.append({"year": year, "config": str(out_path), "sha256": sha(out_path)})
        print(f"{year}: train_end_x={cfg['train_end_x']} validation_end_x={validation_end_x} "
              f"calib_test_end_x={calib_test_end_x}")

    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite existing manifest {manifest_path}")
    manifest_path.write_text(json.dumps({
        "purpose": "Validation-only six-year physics calibration, interval-corrected MRMS forcing",
        "package_path": CORRECTED_PACKAGE_PATH,
        "annual_template_source": str(ANNUAL_TEMPLATE_SOURCE),
        "annual_template_source_sha256": sha(ANNUAL_TEMPLATE_SOURCE),
        "jobs": jobs,
    }, indent=2) + "\n")
    print(f"\nWrote {len(jobs)} configs + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
