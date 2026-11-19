#!/usr/bin/env python3
"""Join a verified GIStoOHQ basin-forcing export to bounded native gauge inputs.

Preserves native Q/auxiliary weather strings; no fitting, scaling or imputation.
New native v2 carries explicit reference-ET semantics and provider-day timing.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime

from prepare_gauge_alignment_review import validate_identity_csv
from prepare_hydrofusion_causal_package import CSV_HEADER
from audit_hydrofusion_producer_package import read_json

EPOCH=1420070400


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(token):
    if not isinstance(token,str) or not re.fullmatch(r'[0-9]+',token):
        raise ValueError('Canonical integer timestamp/flag required')
    return int(token)


def finite(token):
    value=float(token)
    if not math.isfinite(value) or value<0:
        raise ValueError('Finite nonnegative forcing required')
    return value


def station_binding(base, weather_dir, producer_catalog, original_export_manifest, station_metadata, expected_station=""):
    """Bind Q's declared source chain to the new basin's explicit USGS station."""
    import yaml
    from audit_hydrofusion_gauge_alignment import audit
    if any(p is None for p in (producer_catalog, original_export_manifest, station_metadata)):
        raise ValueError('Explicit native station-provenance metadata required')
    producer_catalog, original_export_manifest, station_metadata = map(Path,(producer_catalog,original_export_manifest,station_metadata))
    exported=read_json(original_export_manifest.read_bytes())
    if (sha(original_export_manifest)!=base['producer_manifest_sha256'] or
            exported['source_package_id']!=base['source_id']):
        raise ValueError('Original native producer binding mismatch')
    catalog=read_json(producer_catalog.read_bytes())
    assets={a['asset_id']:a for a in catalog['assets']}
    if len(assets)!=len(catalog['assets']):raise ValueError('Duplicate catalog identities')
    derived=[assets[a['asset_id']] for a in exported['assets'] if a['asset_id'] in assets and assets[a['asset_id']].get('provider')=='usgs']
    if len(derived)!=1:raise ValueError('Exactly one source discharge asset required')
    parents=derived[0].get('parent_asset_ids',[])
    if len(parents)!=1 or parents[0] not in assets:raise ValueError('Native discharge parent is missing')
    parent=assets[parents[0]]
    station=parent.get('station_id')
    if (parent.get('provider')!='usgs' or parent.get('processing_status')!='native' or
            not isinstance(station,str) or not re.fullmatch(r'[0-9]{8,15}',station) or
            parent.get('request_parameters',{}).get('sites')!=station):
        raise ValueError('Native USGS station identity is not explicit')
    site=yaml.safe_load((weather_dir/'site_spec.yaml').read_text())
    site_station=site.get('sources',{}).get('discharge',{}).get('station_id','')
    if (expected_station and site_station and expected_station!=site_station) or (expected_station or site_station)!=station:
        raise ValueError('Forcing basin and discharge station identities disagree')
    review=audit(site,station_metadata.read_text(),station)
    if review['metadata_discrepancy']:raise ValueError('Gauge area/outlet metadata requires review before input release')
    return {'station_id':station,'native_asset_id':parent['asset_id'],'derived_asset_id':derived[0]['asset_id'],
            'station_area_m2':review['gauge_area_m2'],'outlet_distance_m':review['outlet_distance_m'],
            'producer_catalog_sha256':sha(producer_catalog),'original_export_manifest_sha256':sha(original_export_manifest),
            'station_metadata_sha256':sha(station_metadata),
            'limitation':'Metadata/source-chain identity is checked; does not independently validate rating curves or urban drainage.'}


def prepare(native_manifest, forcing_directory, output, *, producer_catalog=None, original_export_manifest=None, station_metadata=None):
    native_manifest, forcing_directory, output=map(Path,(native_manifest,forcing_directory,output))
    if output.exists():raise FileExistsError(output)
    base=read_json(native_manifest.read_bytes())
    first,end=base['first_interval_x'],base['end_interval_x_exclusive']
    if (base['schema']!='HydroFusionPreparedNative-v1' or type(first) is not int or type(end) is not int or
            not 0<=first<end<77016 or base['row_count']!=end-first or base['interval_csv']!='intervals.csv'):
        raise ValueError('Wholly presealed native-v1 source required before CSV access')
    fm=forcing_directory/'manifest.json'
    weather=read_json(fm.read_bytes())
    if weather['schema']!='GaugeBasinForcing-v1' or weather['et_day_start_hour_utc']!=7:
        raise ValueError('Versioned GIStoOHQ reference-ET export required')
    allowed={'rainfall.csv','reference_et.csv','spatial_weights.json','basin.geojson','site_spec.yaml','request.json','raw_sources.json'}
    if set(weather['files']) not in (allowed,allowed|{'source_basin'}):
        raise ValueError('Unexpected producer asset paths')
    for name,digest in weather['files'].items():
        path=forcing_directory/name
        if path.is_symlink() or sha(path)!=digest:raise ValueError('Producer asset hash/path mismatch: '+name)
    if 'source_basin' in weather['files'] and sha(forcing_directory/'source_basin')!=weather['basin_sha256']:
        raise ValueError('Original basin source identity mismatch')
    start_epoch,end_epoch=EPOCH+first*3600,EPOCH+end*3600
    if (int(datetime.fromisoformat(weather['start_utc']).timestamp())!=start_epoch or
            int(datetime.fromisoformat(weather['end_exclusive_utc']).timestamp())!=end_epoch):
        raise ValueError('Producer/native chronological supports differ')
    area=weather['catchment_area_m2']
    if isinstance(area,bool) or not isinstance(area,(int,float)) or not math.isfinite(area) or area<=0:
        raise ValueError('Explicit positive producer basin area required')
    if weather.get('et_semantics')!='ASCE short-grass reference evapotranspiration':
        raise ValueError('Reference ET semantics missing')
    from shapely.geometry import shape
    from shapely.ops import unary_union,transform
    from pyproj import Transformer
    geometry=read_json((forcing_directory/'basin.geojson').read_bytes())
    basin=unary_union([shape(f['geometry']) for f in geometry['features']])
    # Equal-area verification; minor UTM/reference projection differences are disclosed.
    basin_area=transform(Transformer.from_crs(4326,5070,always_xy=True).transform,basin).area
    if basin.is_empty or not basin.is_valid or not math.isfinite(basin_area) or basin_area<=0 or abs(area/basin_area-1)>.001:
        raise ValueError('Producer normalization area and supplied geometry disagree')
    binding=station_binding(base,forcing_directory,producer_catalog,original_export_manifest,station_metadata,weather.get("station_id",""))
    import yaml
    site=yaml.safe_load((forcing_directory/'site_spec.yaml').read_text())
    if abs(float(site['catchment']['area_m2'])-area)>1e-6:
        raise ValueError('Producer manifest and basin SiteSpec area disagree')
    native_csv=native_manifest.parent/'intervals.csv'
    if native_csv.is_symlink() or sha(native_csv)!=base['interval_csv_sha256']:
        raise ValueError('Original native interval hash/path mismatch')
    validate_identity_csv(native_csv,first,end)
    rain={}
    with (forcing_directory/'rainfall.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['interval_start_epoch_seconds','rain_mm','valid','spatial_coverage']:
            raise ValueError('Unexpected rainfall schema')
        for n,row in enumerate(reader):
            t=integer(row['interval_start_epoch_seconds'])
            if t!=start_epoch+n*3600 or t>=end_epoch:raise ValueError('Rainfall time identity mismatch')
            if row['valid'] not in {'0','1'}:raise ValueError('Invalid rainfall validity mask')
            value,coverage=finite(row['rain_mm']),finite(row['spatial_coverage'])
            valid=row['valid']=='1'
            if coverage>1+1e-10 or valid and coverage<1-1e-10:
                raise ValueError('Partial rain coverage cannot be marked valid')
            rain[t]=(value,valid)
    if len(rain)!=end-first:raise ValueError('Incomplete rainfall support')
    daily=[]
    with (forcing_directory/'reference_et.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['day_start_epoch_seconds','reference_et_mm_day','valid','spatial_coverage']:
            raise ValueError('Unexpected reference ET schema')
        previous=None
        for row in reader:
            t=integer(row['day_start_epoch_seconds'])
            if not start_epoch<=t<=end_epoch-86400 or t%86400!=7*3600 or previous is not None and t!=previous+86400:
                raise ValueError('Reference ET provider-day identity mismatch')
            previous=t
            if row['valid'] not in {'0','1'}:raise ValueError('Invalid reference ET mask')
            value,coverage=finite(row['reference_et_mm_day']),finite(row['spatial_coverage'])
            valid=row['valid']=='1'
            if coverage>1+1e-10 or valid and coverage<1-1e-10:raise ValueError('Partial ET coverage marked valid')
            daily.append({'day_start_epoch_seconds':t,'reference_et_mm_day':value,'valid':valid})
    expected_days=list(range(start_epoch+7*3600,end_epoch-86400+1,86400))
    if [d['day_start_epoch_seconds'] for d in daily]!=expected_days:
        raise ValueError('Incomplete provider-day ET support')
    provenance={'station_binding':binding,'preparer_sha256':sha(__file__),'original_manifest_sha256':sha(native_manifest),'original_csv_sha256':sha(native_csv),
                'gistohq_forcing_manifest_sha256':sha(fm),'producer_files':weather['files']}
    source_id='sha256:'+hashlib.sha256(json.dumps(provenance,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.corrected-gauge-',dir=output.parent))
    try:
        with native_csv.open(newline='') as stream,(stage/'intervals.csv').open('w',newline='') as dest:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=CSV_HEADER:raise ValueError('Original native CSV schema changed')
            writer=csv.DictWriter(dest,fieldnames=CSV_HEADER,lineterminator='\n');writer.writeheader()
            for row in reader:
                t=integer(row['epoch_seconds']);value,valid=rain[t]
                if row['forcing_valid'] not in {'0','1'}:raise ValueError('Invalid original forcing mask')
                row.update(p_mm_h=format(value,'.17g'),pet_mm_h='0',
                           forcing_valid=str(int(row['forcing_valid']=='1' and valid)),source_id=source_id)
                writer.writerow(row)
        manifest=dict(base,schema='HydroFusionPreparedNative-v2',source_id=source_id,
                      interval_csv_sha256=sha(stage/'intervals.csv'),catchment_area_m2=area,
                      catchment_area_source=weather['catchment_area_source'],daily_pet=daily,
                      daily_pet_utc_start_hour=7,pet_input_semantics='short_grass_reference_et',
                      derivation=provenance,research_training_authorized=False,
                      input_review_scope='Topographic gauge-domain retrospective comparison; urban drainage not independently verified',
                      forcing_spatial_support={'rainfall':'exact supplied gauge-polygon area weights; AORC',
                                               'reference_et':'exact supplied gauge-polygon area weights; gridMET',
                                               'other_weather':'Original coarse POWER provider-point covariates retained unchanged'},
                      limitations=['New retrospective analysis inputs; no proof of operational provider latency.',
                                   'Original Q remains m3/s; conversion uses the explicit gauge-domain area.',
                                   'No imputation; all source missing/quality masks retained.',
                                   'New fitting requires an explicit budget; completed old campaigns are preserved.'])
        manifest['historical_producer_fields']={key:manifest.pop(key) for key in list(manifest) if key.startswith('producer_')}
        (stage/'native_input_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        shutil.copy2(fm,stage/'gistohq_forcing_manifest.json')
        shutil.copy2(forcing_directory/'basin.geojson',stage/'gauge_basin.geojson')
        shutil.copy2(native_manifest,stage/'original_native_input_manifest.json')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage);raise
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['native_manifest','forcing_directory','new_output_directory']:parser.add_argument(key,type=Path)
    for key in ['producer_catalog','original_export_manifest','station_metadata']:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    m=prepare(args.native_manifest,args.forcing_directory,args.new_output_directory,producer_catalog=args.producer_catalog,original_export_manifest=args.original_export_manifest,station_metadata=args.station_metadata)
    print(f"Prepared native v2: {m['row_count']} intervals; {len(m['daily_pet'])} reference ET days; no fitting.")
