#!/usr/bin/env python3
"""Second, independent catalog-to-export trace: reads the ALREADY-REGISTERED
basin-mean values from the project's real MRMS asset catalog
(results/nse08_recovery_20261015/mrms_catalog.json + its object store),
rather than recomputing from raw GRIB2 (that is
audit_mrms_catalog_export_trace.py's job). This directly replicates
report 32's original method ("raw_catalog_value" vs "exported_value")
and extends it to the MultiSensor Pass2 era, using the real production
catalog rather than an ad-hoc script (none was preserved from report 32).

No network access, no package modification, no training. Does not
touch any timestamp on or after 2023-10-15T00:00:00Z.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

CACHE_ROOT = Path("/home/arash/Dropbox/HydroPINN paper/results/nse08_recovery_20261015")
CATALOG_PATH = CACHE_ROOT / "mrms_catalog.json"
OBJECT_STORE = CACHE_ROOT / "mrms_cache" / "objects" / "sha256"
PACKAGE = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_corrected_devperiod/hydropinn"
)
TEMPORAL_4 = PACKAGE / "observations" / "temporal_4.csv"
SEAL_START = datetime(2023, 10, 15, tzinfo=timezone.utc)

# Same representative days as audit_mrms_catalog_export_trace.py, so the
# two independent methods (recompute-from-GRIB vs read-from-catalog) are
# directly comparable for the same hours.
REPRESENTATIVE_DAYS = [
    ("2020-06-04", "GaugeCorr era"),
    ("2020-09-09", "GaugeCorr era"),
    ("2021-08-09", "MultiSensor Pass2 era"),
    ("2022-07-02", "MultiSensor Pass2 era"),
]


def load_exported_values() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(TEMPORAL_4, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variable"] != "PRECIP_MRMS_BASIN_AVG":
                continue
            out[row["timestamp_utc"]] = {"value": float(row["value"]), "product": row["provider_qualifiers"]}
    return out


def load_catalog_rows_for_day(day: str) -> dict[str, dict]:
    """Returns timestamp_utc -> {value, product} for every catalog asset
    whose temporal_coverage includes any hour of `day`."""
    catalog = json.loads(CATALOG_PATH.read_text())
    day_start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    day_end_excl = day_start.replace(hour=23) if True else None
    rows_out: dict[str, dict] = {}
    for asset in catalog["assets"]:
        cov_start = datetime.strptime(asset["temporal_coverage"]["start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        cov_end = datetime.strptime(asset["temporal_coverage"]["end"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if cov_end < day_start or cov_start > day_start.replace(hour=23):
            continue
        digest = asset["content_digest"]
        blob_path = OBJECT_STORE / digest[:2] / digest[2:]
        if not blob_path.exists():
            raise FileNotFoundError(f"catalog references missing object: {blob_path}")
        payload = json.loads(blob_path.read_text())
        for row in payload["rows"]:
            ts = row["timestamp_utc"]
            if ts.startswith(day):
                rows_out[ts] = {"value": row["value"], "product": row["product"], "asset_id": asset["asset_id"]}
    return rows_out


def main() -> None:
    exported = load_exported_values()
    results = []
    n_compared = 0
    n_mismatch = 0

    for day, era_label in REPRESENTATIVE_DAYS:
        day_start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if day_start >= SEAL_START:
            raise RuntimeError(f"refusing to process a day at/after the sealed holdout boundary: {day}")
        catalog_rows = load_catalog_rows_for_day(day)
        if not catalog_rows:
            results.append({"day": day, "era": era_label, "error": "no catalog rows found for this day"})
            continue
        for ts in sorted(catalog_rows):
            cat = catalog_rows[ts]
            exp = exported.get(ts)
            n_compared += 1
            row = {
                "era": era_label,
                "timestamp_utc": ts,
                "catalog_value": cat["value"],
                "catalog_product": cat["product"],
                "catalog_asset_id": cat["asset_id"],
                "exported_value": exp["value"] if exp else None,
                "exported_product": exp["product"] if exp else None,
                "present_in_export": exp is not None,
            }
            if exp is not None:
                row["abs_difference"] = abs(cat["value"] - exp["value"])
                row["equal_within_1e-9"] = row["abs_difference"] < 1e-9
                if not row["equal_within_1e-9"]:
                    n_mismatch += 1
            else:
                row["equal_within_1e-9"] = None
                n_mismatch += 1
            results.append(row)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "Reads already-registered basin-mean rows directly from the production "
                  "AssetCatalog (mrms_catalog.json) + its content-addressed object store, "
                  "compared against the package's exported temporal_4.csv at the same "
                  "timestamp. This is the same method report 32 used for 2020-09-10 "
                  "(GaugeCorr), extended here to additional GaugeCorr days and to the "
                  "MultiSensor Pass2 era, using the real catalog rather than an "
                  "unrecovered ad-hoc script.",
        "catalog_path": str(CATALOG_PATH),
        "n_hours_compared": n_compared,
        "n_mismatches_or_missing": n_mismatch,
        "rows": results,
    }
    out_path = Path("/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/component_diagnostics") / "mrms_catalog_lookup_trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({n_compared} hours compared, {n_mismatch} mismatches/missing)")


if __name__ == "__main__":
    main()
