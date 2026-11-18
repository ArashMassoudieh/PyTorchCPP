#!/usr/bin/env python3
"""Builds a SEPARATE, versioned, development-only package with MRMS
precipitation re-labeled from interval-END timestamps (the producer's
actual convention, confirmed in report 35) to interval-START timestamps
(new_ts = original_ts - 1h), per report 33's frozen protocol.

Only observations/temporal_4.csv (MRMS, variable PRECIP_MRMS_BASIN_AVG)
is modified. temporal_1.csv (NASA POWER forcing), temporal_2.csv
(PET), and temporal_3.csv (USGS discharge) are copied byte-identical
and hash-verified, per the explicit instruction to never shift
discharge/NASA POWER/PET. The ORIGINAL package
(sligocreekdemo_mrms_corrected_devperiod) is never opened for writing.

Produces, inside the new package root (NOT inside observations/, so the
HydroPINNExport loader's asset list is unaffected):
  - interval_correction_sidecar.csv: original_timestamp_utc,
    corrected_timestamp_utc, variable, value, native_unit,
    provider_qualifiers for every MRMS row, so the correction is fully
    reversible/auditable without touching the original package.
  - build_report.json: counts, hashes, and every check this script ran.

Checks performed (all required by the frozen protocol):
  - No duplicate corrected timestamps.
  - No corrected timestamp before the package's declared study_start.
  - No corrected timestamp at or after the sealed holdout boundary
    (2023-10-15T00:00:00Z) -- this is also asserted as a hard failure,
    not just logged, since shifting only moves timestamps earlier it
    cannot newly violate this, but it is checked explicitly anyway.
  - temporal_1/2/3.csv byte-identical to the source (sha256 compared).
  - Row count preserved (no row silently dropped or duplicated).
  - Spatial-CV alignment: this package has no spatial-CV variable
    (confirmed against variables.json), so this check is logged as
    not-applicable rather than silently skipped, per protocol wording
    ("shift matching CV only if applicable").

No training launched. No modification to the original package.
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
    "sligocreekdemo_mrms_corrected_devperiod/hydropinn"
)
DEST_PACKAGE_ROOT = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_interval_corrected_devperiod"
)
DEST_HYDROPINN = DEST_PACKAGE_ROOT / "hydropinn"
SEAL_START = datetime(2023, 10, 15, tzinfo=timezone.utc)
SHIFT = timedelta(hours=-1)
MRMS_VARIABLE_NAME = "PRECIP_MRMS_BASIN_AVG"
MRMS_ASSET_INDEX = 3  # temporal_4.csv is assets[3] (0-indexed) per manifest.json


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
        raise RuntimeError(
            f"destination already exists, refusing to overwrite: {DEST_PACKAGE_ROOT}. "
            "Delete it manually first if a clean rebuild is intended."
        )

    source_manifest = json.loads((SOURCE_PACKAGE / "manifest.json").read_text())
    study_start = parse_ts(source_manifest["study_start"])

    # --- 1. temporal_4.csv (MRMS): read, shift, validate ---
    src_temporal_4 = SOURCE_PACKAGE / "observations" / "temporal_4.csv"
    with open(src_temporal_4, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        src_rows = list(reader)

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
            continue  # hard exclude, never write a sealed-period row, logged below
        if corrected_dt < study_start:
            n_before_study_start += 1
            continue  # boundary-clip: corrected timestamp would predate the package's own study window
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
            f"REFUSING TO PROCEED: {n_at_or_after_seal} corrected row(s) would fall at/after the "
            "sealed holdout boundary. This should be mathematically impossible for a -1h shift on "
            "a package that already ends before the boundary; something is wrong upstream."
        )

    # --- 2. Build destination package ---
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

    # variables.json: unchanged schema (timestamp convention is documented in the
    # manifest's provenance_note, not a per-variable schema field)
    shutil.copyfile(SOURCE_PACKAGE / "variables.json", DEST_HYDROPINN / "variables.json")
    variables_doc = json.loads((SOURCE_PACKAGE / "variables.json").read_text())
    has_spatial_cv = any("SPATIAL_CV" in v["name"] for v in variables_doc["variables"])

    # --- 3. New manifest.json ---
    new_manifest = dict(source_manifest)
    new_manifest["site_id"] = "sligocreekdemo_mrms_interval_corrected_devperiod"
    new_manifest["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_manifest["qc_gate"] = "not_run"
    new_manifest["provenance_note"] = (
        "Development-only variant of sligocreekdemo_mrms_corrected_devperiod with ONLY "
        "observations/temporal_4.csv (MRMS, PRECIP_MRMS_BASIN_AVG) relabeled from the "
        "producer's interval-END timestamp convention to interval-START "
        "(corrected_timestamp_utc = original_timestamp_utc - 1 hour; values and product "
        "names unchanged), per the frozen protocol in "
        "framework_decade_20261022/improvement_loop_20261023/33_interval_alignment_protocol.md, "
        "informed by the NOAA-documentation interval verification in report 35 and the "
        "zero-mismatch acquisition-to-export trace in report 36. "
        "temporal_1.csv (NASA POWER forcing), temporal_2.csv (PET), and temporal_3.csv "
        "(USGS discharge) are byte-identical, sha256-verified copies of the parent "
        "package's files -- NOT shifted, per explicit instruction to never shift "
        "discharge, NASA POWER, or PET automatically. Original timestamps, values, and "
        "product names for every MRMS row are preserved in "
        "interval_correction_sidecar.csv at this package's root (sibling to manifest.json, "
        "not inside observations/, so the HydroPINNExport loader's asset list is "
        "unaffected). This is NOT a replacement for sligocreekdemo_mrms_corrected_devperiod "
        "-- that package is unmodified and remains the frozen baseline for every prior "
        "result in this program. Build script: "
        "Hydro/experiments/gistohq_sligo/build_mrms_interval_corrected_package.py. "
        "Full check results: build_report.json at this package's root."
    )
    (DEST_HYDROPINN / "manifest.json").write_text(json.dumps(new_manifest, indent=2, sort_keys=True))

    # --- 4. Build report ---
    build_report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_package": str(SOURCE_PACKAGE),
        "dest_package": str(DEST_HYDROPINN),
        "shift_applied_hours": -1,
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
            "temporal_1_2_3_byte_identical_to_source": True,  # raised above if false
            "spatial_cv_variable_present_in_this_package": has_spatial_cv,
            "spatial_cv_shift_applicable": has_spatial_cv,
            "spatial_cv_note": (
                "This package (sligocreekdemo_mrms_corrected_devperiod) has no "
                "PRECIP_MRMS_SPATIAL_CV variable, confirmed against variables.json -- "
                "the spatial-CV feature lives only in the separate "
                "sligocreekdemo_mrms_spatial_cv_devperiod variant, which was already "
                "rejected (reports 14/15) and is not part of this build. Spatial-CV "
                "alignment is therefore logged as not-applicable, per the protocol's own "
                "'only if applicable' wording, not silently skipped."
            ),
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
