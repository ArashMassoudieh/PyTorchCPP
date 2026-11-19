#!/usr/bin/env python3
"""Prepare a NEW, deliberately non-runnable gauge-alignment review package.

No acquisition, normalization, fitting or edits to original packages. Consumer
schema is intentionally unsupported until basin/forcing review is complete.
"""
import argparse
import csv
import re
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_identity_csv(path, first, end):
    count=0
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or reader.fieldnames[0]!='epoch_seconds':
            raise ValueError('Missing interval identities')
        for count,row in enumerate(reader,1):
            token=row['epoch_seconds']
            if count>end-first or not re.fullmatch(r'\d+',token) or int(token)!=1420070400+(first+count-1)*3600:
                raise ValueError('CSV timestamp/support mismatch; values not interpreted')
    if count!=end-first:
        raise ValueError('CSV support incomplete')


def prepare(source, gis_audit, catalog, output):
    source, gis_audit, catalog, output = map(Path, (source, gis_audit, catalog, output))
    if output.exists():
        raise FileExistsError(output)
    original = source/'native_input_manifest.json'
    manifest = json.loads(original.read_text())
    # Establish exact presealed bounds BEFORE touching the native CSV.
    first, end = manifest['first_interval_x'], manifest['end_interval_x_exclusive']
    if (manifest['schema'] != 'HydroFusionPreparedNative-v1' or
            type(first) is not int or type(end) is not int or not 0 <= first < end < 77016 or
            manifest['row_count'] != end-first or manifest['interval_csv'] != 'intervals.csv'):
        raise ValueError('Exact bounded presealed source required')
    for day in manifest['daily_pet']:
        if not 1420070400 <= day['day_start_epoch_seconds'] < 1420070400+end*3600:
            raise ValueError('PET outside presealed source bounds')
    csv_path = source/'intervals.csv'
    if csv_path.is_symlink() or digest(csv_path) != manifest['interval_csv_sha256']:
        raise ValueError('Native interval hash/path mismatch')
    validate_identity_csv(csv_path, first, end)
    audit = json.loads((gis_audit/'audit.json').read_text())
    if not audit['gauge_is_routed_upstream_subset'] or audit['results']['reviewed_outlet']['saved_boundary_iou'] != 1:
        raise ValueError('Verified nested-basin evidence required')
    for path, expected in audit['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('GIS audit input changed: '+path)
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union
    geometry_path = gis_audit/'gauge_candidate.geojson'
    basin = unary_union([shape(f['geometry']) for f in json.loads(geometry_path.read_text())['features']])
    forcing = []
    for asset in json.loads(catalog.read_text())['assets']:
        if asset.get('provider') == 'nasa-power' and asset.get('processing_status') == 'native':
            forcing.append({'asset_id':asset['asset_id'], 'product':asset['product'],
                            'spatial_support':asset['spatial_support'], 'longitude':asset['longitude'],
                            'latitude':asset['latitude'], 'requested_point_inside_gauge_basin':
                            basin.covers(Point(asset['longitude'],asset['latitude'])),
                            'upstream_representativeness':'unverified'})
    if {item['product'] for item in forcing} != {'pet-et','historical-meteorology'}:
        raise ValueError('Both forcing metadata sources required')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(csv_path,output/'intervals.csv')
    shutil.copy2(original,output/'original_native_input_manifest.json')
    shutil.copy2(geometry_path,output/'gauge_candidate.geojson')
    if digest(output/'intervals.csv') != manifest['interval_csv_sha256']:
        raise RuntimeError('Copy failed identity check')
    pending = dict(manifest, schema='HydroFusionPreparedNative-PENDING-SPATIAL-REVIEW',
                   catchment_area_m2=audit['results']['gauge']['area_km2']*1e6,
                   catchment_area_source='Existing GIStoOHQ GRASS upstream-gauge candidate; review pending',
                   training_authorized=False, research_ready=False,
                   original_catchment_area_m2=manifest['catchment_area_m2'],
                   forcing_spatial_review=forcing,
                   remaining_gates=['Review upstream basin/urban drainage','Verify or replace upstream forcing/PET',
                                    'Approve released input provenance/support','Obtain separate research fitting budget'],
                   derivation_sha256={str(p.resolve()):digest(p) for p in [original, gis_audit/'audit.json',geometry_path,catalog]})
    (output/'native_input_manifest.json').write_text(json.dumps(pending,indent=2,allow_nan=False)+'\n')
    (output/'README.md').write_text('# Gauge-alignment review package — NOT runnable\n\n'
        '2015–2018 raw Q remains in m³/s; CSV byte-identical to original. No discharge rescaling performed. '
        'GIS upstream area is a pending normalization candidate, not an accepted replacement. '
        'Original producer manifest/audit provenance remains historical; no new producer acquisition is implied. '
        'Existing POWER rainfall/PET requested downstream, outside this basin; same provider cell or basin '
        'representativeness has NOT been established. The intentionally unsupported schema prevents training '
        'before review. Do not bypass it by renaming the schema. Preserve old packages/results.\n')
    return pending


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','gis_audit','catalog','output']:p.add_argument(name,type=Path)
    a=p.parse_args();prepare(a.source,a.gis_audit,a.catalog,a.output)
    print('Saved gauge-alignment review package; model loading intentionally blocked.')
