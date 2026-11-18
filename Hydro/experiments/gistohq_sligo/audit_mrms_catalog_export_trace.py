#!/usr/bin/env python3
"""Reusable catalog-to-export provenance audit for MRMS basin-averaged
precipitation.

For each requested local GRIB2 file, independently recomputes the
area-weighted basin-mean precipitation using the project's own
production functions (ohqbuilder.watershed_data.mrms:
basin_cell_intersection_weights_m2 and area_weighted_basin_average --
not a reimplemented pipeline), then compares that recomputed value to
the value actually exported in the package's temporal_4.csv at the
same timestamp.

This extends report 32's existing GaugeCorr-only trace (one day,
2020-09-10) to a reusable script covering BOTH product families
(GaugeCorr pre-2020-10-14, MultiSensor Pass2 post-2020-10-14) on
representative wet days, per improvement_loop_20261023 report 35's
remaining item 2.

No network access, no package modification, no training. Reads only
already-downloaded local GRIB2 files under raw_data/mrms_flood_N/ and
the existing package's temporal_4.csv. Does not touch any timestamp
on or after 2023-10-15T00:00:00Z (the sealed holdout boundary) --
every date used here is hardcoded well before that boundary.

Usage:
    python3 audit_mrms_catalog_export_trace.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

GISTOOHQ_ROOT = Path("/mnt/3rd900/Projects/GIStoOHQ")
sys.path.insert(0, str(GISTOOHQ_ROOT))

from ohqbuilder.watershed_data.mrms import (  # noqa: E402
    area_weighted_basin_average,
    basin_cell_intersection_weights_m2,
)

RAW_DATA = Path("/home/arash/Dropbox/HydroPINN paper/results/nse08_recovery_20261015/raw_data")
BASIN_GEOJSON = RAW_DATA / "nldi_basin_01650800.geojson"
PACKAGE = Path(
    "/mnt/3rd900/Projects/GIStoOHQ/examples/SligoCreek/outputs/"
    "sligocreekdemo_mrms_corrected_devperiod/hydropinn"
)
TEMPORAL_4 = PACKAGE / "observations" / "temporal_4.csv"
SEAL_START = datetime(2023, 10, 15, tzinfo=timezone.utc)

# Representative wet days: one dominant-storm day per directory, two
# directories per product era (report 32 already independently checked
# 2020-09-10/GaugeCorr for one full day; these are additional,
# non-duplicate days, chosen from directories whose GRIB2 filenames are
# all one product -- confirmed by listing each directory before writing
# this script).
REPRESENTATIVE_DAYS = [
    ("mrms_flood_1", "20200604", "GaugeCorr era"),
    ("mrms_flood_5", "20200909", "GaugeCorr era"),
    ("mrms_flood_6", "20210809", "MultiSensor Pass2 era"),
    ("mrms_flood_10", "20220702", "MultiSensor Pass2 era"),
]


def load_basin_geometry():
    from shapely.geometry import shape

    raw = json.loads(BASIN_GEOJSON.read_text())
    geom = raw["features"][0]["geometry"] if "features" in raw else raw
    return shape(geom)


def load_exported_values() -> dict[str, dict]:
    """timestamp_utc -> {value, provider_qualifiers} from temporal_4.csv."""
    out: dict[str, dict] = {}
    with open(TEMPORAL_4, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variable"] != "PRECIP_MRMS_BASIN_AVG":
                continue
            out[row["timestamp_utc"]] = {
                "value": float(row["value"]),
                "product": row["provider_qualifiers"],
            }
    return out


FNAME_RE = re.compile(r"^(?P<product>.+?)_00\.00_(?P<date>\d{8})-(?P<time>\d{6})\.grib2$")


def parse_grib_filename(path: Path) -> tuple[str, datetime]:
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unrecognized MRMS GRIB2 filename: {path.name}")
    dt = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return m.group("product"), dt


def recompute_hour(grib_path: Path, basin_geometry, weights_cache: dict) -> dict:
    import xarray as xr

    ds = xr.open_dataset(str(grib_path), engine="cfgrib", backend_kwargs={"indexpath": ""})
    varname = list(ds.data_vars)[0]
    da = ds[varname]

    if "positions" not in weights_cache:
        lats = [float(v) for v in da.latitude.values]
        lons_360 = [float(v) for v in da.longitude.values]
        cell_lats, cell_lons, weights = basin_cell_intersection_weights_m2(basin_geometry, lats, lons_360)
        lat_index = {v: i for i, v in enumerate(lats)}
        lon_index = {v: i for i, v in enumerate(lons_360)}
        positions = [(lat_index[lat], lon_index[lon + 360.0]) for lat, lon in zip(cell_lats, cell_lons)]
        weights_cache.update(cell_lats=cell_lats, cell_lons=cell_lons, weights=weights, positions=positions)

    grid = da.values
    values = [float(grid[i, j]) for i, j in weights_cache["positions"]]
    stats = area_weighted_basin_average(
        weights_cache["cell_lats"], weights_cache["cell_lons"], values, weights_cache["weights"]
    )
    return stats


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    basin_geometry = load_basin_geometry()
    exported = load_exported_values()
    weights_cache: dict = {}
    results = []
    n_compared = 0
    n_mismatch = 0

    for dirname, datestr, era_label in REPRESENTATIVE_DAYS:
        day_dir = RAW_DATA / dirname
        files = sorted(p for p in day_dir.glob(f"*_{datestr}-*.grib2"))
        if not files:
            results.append({"dir": dirname, "date": datestr, "era": era_label, "error": "no matching files found"})
            continue
        for grib_path in files:
            product, dt = parse_grib_filename(grib_path)
            if dt >= SEAL_START:
                raise RuntimeError(f"refusing to process a file at/after the sealed holdout boundary: {grib_path}")
            ts_key = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            stats = recompute_hour(grib_path, basin_geometry, weights_cache)
            recomputed_value = stats["mean"]
            exp = exported.get(ts_key)
            n_compared += 1
            row = {
                "era": era_label,
                "source_file": str(grib_path),
                "source_file_sha256": sha256_file(grib_path),
                "product_from_filename": product,
                "timestamp_utc": ts_key,
                "recomputed_basin_mean_mm_per_hour": recomputed_value,
                "exported_value_mm_per_hour": exp["value"] if exp else None,
                "exported_provider_qualifiers": exp["product"] if exp else None,
                "present_in_export": exp is not None,
            }
            if exp is not None and recomputed_value is not None:
                row["abs_difference"] = abs(recomputed_value - exp["value"])
                row["equal_within_1e-6"] = row["abs_difference"] < 1e-6
                if not row["equal_within_1e-6"]:
                    n_mismatch += 1
            else:
                row["equal_within_1e-6"] = None
                if exp is None:
                    n_mismatch += 1
            results.append(row)
            print(f"{ts_key} [{product}] recomputed={recomputed_value} exported={row['exported_value_mm_per_hour']} "
                  f"match={row['equal_within_1e-6']}")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "package": str(PACKAGE),
        "basin_geojson": str(BASIN_GEOJSON),
        "basin_geojson_sha256": sha256_file(BASIN_GEOJSON),
        "n_hours_compared": n_compared,
        "n_mismatches_or_missing": n_mismatch,
        "method": "Recomputed area-weighted basin mean from local GRIB2 via "
                  "ohqbuilder.watershed_data.mrms.area_weighted_basin_average + "
                  "basin_cell_intersection_weights_m2 (production functions, not reimplemented), "
                  "compared at identical filename timestamp against the package's own exported "
                  "temporal_4.csv row. No shift applied in this comparison -- this tests for "
                  "acquisition/export pipeline correctness (no accidental re-timestamping or "
                  "corruption), not the external NOAA interval-labeling convention (that is "
                  "report 35's separate, already-resolved question).",
        "rows": results,
    }
    out_path = Path("/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/component_diagnostics") / "mrms_catalog_export_trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {out_path} ({n_compared} hours compared, {n_mismatch} mismatches/missing)")


if __name__ == "__main__":
    main()
