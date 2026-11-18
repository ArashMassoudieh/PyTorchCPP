#!/usr/bin/env python3
"""Builds the FULL (non-devperiod) interval-corrected package, for the
one-shot holdout evaluation authorized per reports 45/46. Same logic,
same checks, same -1h shift as build_mrms_interval_corrected_package.py
(report 37) -- repointed at the full source package
(sligocreekdemo_mrms_corrected, study_end 2024-12-31T23:00:00Z) instead
of the devperiod one.

Confirmed before writing this script: the full package's own
temporal_4.csv (MRMS) has native coverage only through
2023-10-14T23:00:00Z -- identical to the devperiod package. After the
-1h shift, the last MRMS row becomes 2023-10-14T22:00:00Z, still before
the seal (2023-10-15T00:00:00Z). The seal-exclusion check below is kept
unweakened from the original script (not because it is expected to
fire, but because nothing is relaxed on the belief that it won't): if
this expectation were ever wrong, the script refuses rather than
silently writing a reserved-period MRMS row.

temporal_1.csv (NASA POWER), temporal_2.csv (PET), temporal_3.csv
(USGS discharge) are copied byte-identical and hash-verified -- these
already extend to 2024-12-31/2025-01-01 in the source package and are
untouched by the shift, exactly as temporal_1-3 were untouched in the
devperiod build.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCE_PACKAGE = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_corrected/hydropinn"
)
DEST_PACKAGE_ROOT = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_interval_corrected_full"
)
DEST_HYDROPINN = DEST_PACKAGE_ROOT / "hydropinn"
SEAL_START = datetime(2023, 10, 15, tzinfo=timezone.utc)
SHIFT = timedelta(hours=-1)
MRMS_VARIABLE_NAME = "PRECIP_MRMS_BASIN_AVG"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    if DEST_PACKAGE_ROOT.exists():
        raise RuntimeError(f"destination already exists, refusing to overwrite: {DEST_PACKAGE_ROOT}")

    source_manifest = json.loads((SOURCE_PACKAGE / "manifest.json").read_text())
    study_start = parse_ts(source_manifest["study_start"])

    src_temporal_4 = SOURCE_PACKAGE / "observations" / "temporal_4.csv"
    with open(src_temporal_4, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        src_rows = list(reader)

    last_original_ts = max(parse_ts(r["timestamp_utc"]) for r in src_rows)
    print(f"Source MRMS last native row: {fmt_ts(last_original_ts)} "
          f"-> after -1h shift: {fmt_ts(last_original_ts + SHIFT)} "
          f"(seal is {fmt_ts(SEAL_START)}; "
          f"{'BEFORE seal, as expected' if last_original_ts + SHIFT < SEAL_START else 'AT/AFTER SEAL -- UNEXPECTED'})")

    corrected_rows = []
    sidecar_rows = []
    seen_corrected_ts: set[str] = set()
    n_duplicate_after_shift = 0
    n_before_study_start = 0
    n_at_or_after_seal = 0

    for row in src_rows:
        if row["variable"] != MRMS_VARIABLE_NAME:
            raise ValueError(f"unexpected variable in temporal_4.csv: {row['variable']!r}")
        orig_dt = parse_ts(row["timestamp_utc"])
        corrected_dt = orig_dt + SHIFT
        corrected_ts = fmt_ts(corrected_dt)

        if corrected_dt >= SEAL_START:
            n_at_or_after_seal += 1
            continue
        if corrected_dt < study_start:
            n_before_study_start += 1
            continue
        if corrected_ts in seen_corrected_ts:
            n_duplicate_after_shift += 1
            continue
        seen_corrected_ts.add(corrected_ts)

        new_row = dict(row)
        new_row["timestamp_utc"] = corrected_ts
        corrected_rows.append(new_row)
        sidecar_rows.append({
            "original_timestamp_utc": row["timestamp_utc"],
            "corrected_timestamp_utc": corrected_ts,
            "variable": row["variable"],
            "value": row["value"],
            "native_unit": row["native_unit"],
            "provider_qualifiers": row["provider_qualifiers"],
        })

    if n_at_or_after_seal > 0:
        raise RuntimeError(
            f"REFUSING TO PROCEED: {n_at_or_after_seal} corrected MRMS row(s) would fall at/after "
            "the sealed holdout boundary. Expectation (no native MRMS past 2023-10-14) was violated -- "
            "stop and investigate before building anything."
        )

    (DEST_HYDROPINN / "observations").mkdir(parents=True)

    for name in ("temporal_1.csv", "temporal_2.csv", "temporal_3.csv"):
        src = SOURCE_PACKAGE / "observations" / name
        dst = DEST_HYDROPINN / "observations" / name
        shutil.copyfile(src, dst)
        src_hash, dst_hash = sha256_file(src), sha256_file(dst)
        if src_hash != dst_hash:
            raise RuntimeError(f"copy verification failed for {name}: {src_hash} != {dst_hash}")

    dst_temporal_4 = DEST_HYDROPINN / "observations" / "temporal_4.csv"
    with open(dst_temporal_4, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in sorted(corrected_rows, key=lambda r: r["timestamp_utc"]):
            writer.writerow(row)

    sidecar_path = DEST_PACKAGE_ROOT / "interval_correction_sidecar.csv"
    with open(sidecar_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sidecar_rows[0].keys()))
        writer.writeheader()
        for row in sorted(sidecar_rows, key=lambda r: r["original_timestamp_utc"]):
            writer.writerow(row)

    shutil.copyfile(SOURCE_PACKAGE / "variables.json", DEST_HYDROPINN / "variables.json")

    new_manifest = dict(source_manifest)
    new_manifest["site_id"] = "sligocreekdemo_mrms_interval_corrected_full"
    new_manifest["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_manifest["qc_gate"] = "not_run"
    new_manifest["provenance_note"] = (
        "FULL (non-devperiod) interval-corrected package, built for the one-shot reserved-period "
        "holdout evaluation authorized per reports 45/46 of "
        "framework_decade_20261022/improvement_loop_20261023. Same -1h MRMS relabeling as "
        "sligocreekdemo_mrms_interval_corrected_devperiod (report 37), applied to the full source "
        "package (sligocreekdemo_mrms_corrected) instead of the devperiod-bounded one. "
        "IMPORTANT, checked not assumed: the source package's own MRMS (temporal_4.csv) has native "
        "coverage only through 2023-10-14T23:00:00Z -- identical to the devperiod package -- so this "
        "correction has NO forcing-source presence in the reserved period itself (2023-10-15 onward); "
        "every reserved-period hour's precipitation forcing is NASA POWER PRECTOTCORR fallback, "
        "identical whether MRMS is 'corrected' or not. temporal_1/2/3.csv are byte-identical, "
        "sha256-verified copies of the source package (NASA POWER forcing, PET, USGS discharge), "
        "extending to 2024-12-31/2025-01-01, unchanged. Original MRMS timestamps/values preserved in "
        "interval_correction_sidecar.csv at this package's root. Build script: "
        "Hydro/experiments/gistohq_sligo/build_mrms_interval_corrected_full_package.py. Full check "
        "results: build_report.json at this package's root."
    )
    (DEST_HYDROPINN / "manifest.json").write_text(json.dumps(new_manifest, indent=2, sort_keys=True))

    build_report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_package": str(SOURCE_PACKAGE),
        "dest_package": str(DEST_HYDROPINN),
        "shift_applied_hours": -1,
        "mrms_last_native_row_original": fmt_ts(last_original_ts),
        "mrms_last_native_row_after_shift": fmt_ts(last_original_ts + SHIFT),
        "mrms_row_counts": {
            "source_rows": len(src_rows),
            "corrected_rows_written": len(corrected_rows),
            "dropped_duplicate_after_shift": n_duplicate_after_shift,
            "dropped_before_study_start": n_before_study_start,
            "dropped_at_or_after_seal": n_at_or_after_seal,
        },
        "checks": {
            "no_duplicate_corrected_timestamps": n_duplicate_after_shift == 0,
            "no_row_before_study_start": n_before_study_start == 0,
            "no_row_at_or_after_seal": n_at_or_after_seal == 0,
            "temporal_1_2_3_byte_identical_to_source": True,
            "mrms_has_no_reserved_period_presence": last_original_ts + SHIFT < SEAL_START,
        },
        "source_file_hashes": {
            name: sha256_file(SOURCE_PACKAGE / "observations" / name)
            for name in ("temporal_1.csv", "temporal_2.csv", "temporal_3.csv", "temporal_4.csv")
        },
        "dest_file_hashes": {
            name: sha256_file(DEST_HYDROPINN / "observations" / name)
            for name in ("temporal_1.csv", "temporal_2.csv", "temporal_3.csv", "temporal_4.csv")
        },
        "sidecar_path": str(sidecar_path),
        "sidecar_row_count": len(sidecar_rows),
    }
    (DEST_PACKAGE_ROOT / "build_report.json").write_text(json.dumps(build_report, indent=2, sort_keys=True))

    print(json.dumps(build_report["mrms_row_counts"], indent=2))
    print(json.dumps(build_report["checks"], indent=2))
    print(f"\nBuilt {DEST_HYDROPINN}")


if __name__ == "__main__":
    main()
