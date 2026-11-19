#!/usr/bin/env python3
"""Audit station/catchment metadata only; never open observation assets.

Matching metadata is not proof of a correct upstream basin. This audit never
changes areas, rescales discharge, acquires observations or authorizes training.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import yaml


def audit(spec, site_text, station):
    lines = [line for line in site_text.splitlines() if line and not line.startswith('#')]
    if len(lines) < 3:
        raise ValueError('Missing USGS site metadata')
    rows = list(csv.DictReader(io.StringIO('\n'.join([lines[0]] + lines[2:])), delimiter='\t'))
    rows = [r for r in rows if r['site_no'] == station]
    if len(rows) != 1:
        raise ValueError('Station metadata must have exactly one matching record')
    r = rows[0]
    gauge_area = float(r['drain_area_va']) * 1609.344**2
    package_area = float(spec['catchment']['area_m2'])
    outlet = spec['geometry']['outlet']
    lat, lon = float(r['dec_lat_va']), float(r['dec_long_va'])
    values = (gauge_area, package_area, lat, lon, outlet['latitude'], outlet['longitude'])
    if not all(math.isfinite(v) for v in values) or min(gauge_area, package_area) <= 0:
        raise ValueError('Invalid area/coordinate metadata')
    if not all(-90 <= v <= 90 for v in (lat, outlet['latitude'])) or not all(-180 <= v <= 180 for v in (lon, outlet['longitude'])):
        raise ValueError('Coordinates outside geographic domain')
    a, b = math.radians(lat), math.radians(outlet['latitude'])
    dlon = math.radians(lon-outlet['longitude'])
    h = math.sin((a-b)/2)**2 + math.cos(a)*math.cos(b)*math.sin(dlon/2)**2
    distance = 6371008.8 * 2 * math.asin(math.sqrt(min(1., h)))
    return {'station_id': station, 'gauge_area_m2': gauge_area, 'package_area_m2': package_area,
            'package_area_excess_percent': 100*(package_area/gauge_area-1),
            'gauge_latitude': lat, 'gauge_longitude': lon, 'outlet_distance_m': distance,
            'new_training_ready': False,
            'reason': 'Upstream gauge-aligned basin and forcing coverage require verification; no automatic area replacement.',
            'metadata_discrepancy': abs(package_area/gauge_area-1) > .01 or distance > 100,
            'screening_limits': {'relative_area': .01, 'outlet_m': 100},
            'limitations': 'Drainage area/point comparison only. No basin polygon validation or observations read.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site_spec'); p.add_argument('usgs_site_rdb'); p.add_argument('output')
    p.add_argument('--station', default='01650800')
    a = p.parse_args()
    paths = [Path(a.site_spec), Path(a.usgs_site_rdb)]
    result = audit(yaml.safe_load(paths[0].read_text()), paths[1].read_text(), a.station)
    result['source_sha256'] = {str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    result['metadata_url'] = 'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites='+a.station+'&siteOutput=expanded&siteStatus=all'
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
