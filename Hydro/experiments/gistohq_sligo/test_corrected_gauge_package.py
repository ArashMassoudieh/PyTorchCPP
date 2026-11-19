import csv
import json
from pathlib import Path
import pytest
from shapely.geometry import box,mapping
from shapely.ops import transform
from pyproj import Transformer
from audit_hydrofusion_producer_package import sha,EPOCH
from test_hydrofusion_prepared_package import fabricated_package
from prepare_hydrofusion_causal_package import prepare as prepare_native
from prepare_corrected_gauge_package import prepare as join

def prepare(native,weather,output):
    return join(native,weather,output,producer_catalog=weather/'station_catalog.json',original_export_manifest=weather/'station_source_manifest.json',station_metadata=weather/'station.rdb')


def inputs(tmp_path):
    source,policy=fabricated_package(tmp_path)
    native=tmp_path/'native';prepare_native(source,policy,native)
    weather=tmp_path/'weather';weather.mkdir()
    basin=box(-77.01,38.99,-77.,39.)
    area=transform(Transformer.from_crs(4326,5070,always_xy=True).transform,basin).area
    (weather/'basin.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{},'geometry':mapping(basin)}]}))
    with (weather/'rainfall.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['interval_start_epoch_seconds','rain_mm','valid','spatial_coverage'])
        for h in range(300):w.writerow([EPOCH+h*3600,h*.01,1,1.])
    with (weather/'reference_et.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['day_start_epoch_seconds','reference_et_mm_day','valid','spatial_coverage'])
        for h in range(7,300-24+1,24):w.writerow([EPOCH+h*3600,2.4,1,1.])
    for name in ['spatial_weights.json','site_spec.yaml','request.json','raw_sources.json']:(weather/name).write_text('{}')
    manifest={'schema':'GaugeBasinForcing-v1','et_day_start_hour_utc':7,
              'et_semantics':'ASCE short-grass reference evapotranspiration',
              'start_utc':'2015-01-01T00:00:00+00:00','end_exclusive_utc':'2015-01-13T12:00:00+00:00',
              'catchment_area_m2':area,'catchment_area_source':'fabricated geometry only',
              'files':{p.name:sha(p.read_bytes()) for p in weather.iterdir()}}
    import yaml,shutil
    (weather/'site_spec.yaml').write_text(yaml.safe_dump({'geometry':{'outlet':{'longitude':-77.005,'latitude':38.995}},'catchment':{'area_m2':area},'sources':{'discharge':{'station_id':'12345678'}}}))
    manifest['files']['site_spec.yaml']=sha((weather/'site_spec.yaml').read_bytes())
    shutil.copy2(source/'manifest.json',weather/'station_source_manifest.json')
    (weather/'station_catalog.json').write_text(json.dumps({'assets':[
        {'asset_id':'fake','provider':'usgs','parent_asset_ids':['parent']},
        {'asset_id':'parent','provider':'usgs','processing_status':'native','station_id':'12345678','request_parameters':{'sites':'12345678'}}]}))
    (weather/'station.rdb').write_text('site_no\tdrain_area_va\tdec_lat_va\tdec_long_va\n15s\t10n\t10n\t10n\n12345678\t'+str(area/1609.344**2)+'\t38.995\t-77.005\n')
    (weather/'manifest.json').write_text(json.dumps(manifest))
    return native/'native_input_manifest.json',weather


def rehash(weather,name):
    p=weather/'manifest.json';m=json.loads(p.read_text());m['files'][name]=sha((weather/name).read_bytes());p.write_text(json.dumps(m))


def test_join_preserves_q_and_old_metadata_and_versions_et(tmp_path):
    native,weather=inputs(tmp_path);before=native.read_bytes()
    result=prepare(native,weather,tmp_path/'result')
    assert native.read_bytes()==before
    assert result['schema']=='HydroFusionPreparedNative-v2'
    assert result['daily_pet_utc_start_hour']==7
    assert result['daily_pet'][0]['reference_et_mm_day']==2.4
    assert result['daily_pet'][0]['day_start_epoch_seconds']==EPOCH+7*3600
    with (native.parent/'intervals.csv').open() as a,(tmp_path/'result/intervals.csv').open() as b:
        for old,new in zip(csv.DictReader(a),csv.DictReader(b)):
            for key in ['epoch_seconds','q_m3_s','t_c','discharge_valid','native_coverage']:assert old[key]==new[key]
    with pytest.raises(FileExistsError):prepare(native,weather,tmp_path/'result')


@pytest.mark.parametrize('bug',['hash','day_offset','nan_rain','partial_valid','sealed_native','unsafe_path'])
def test_corrupt_inputs_never_publish_package(tmp_path,bug):
    native,weather=inputs(tmp_path)
    if bug=='hash':(weather/'rainfall.csv').write_text('changed')
    if bug in {'nan_rain','partial_valid'}:
        p=weather/'rainfall.csv';s=p.read_text()
        s=s.replace(',0.0,1,1.0',',nan,1,1.0',1) if bug=='nan_rain' else s.replace(',0.0,1,1.0',',0.0,1,0.5',1)
        p.write_text(s);rehash(weather,'rainfall.csv')
    if bug=='day_offset':
        p=weather/'reference_et.csv';p.write_text(p.read_text().replace(str(EPOCH+7*3600),str(EPOCH),1));rehash(weather,p.name)
    if bug=='sealed_native':
        m=json.loads(native.read_text());m['end_interval_x_exclusive']=77017;native.write_text(json.dumps(m));(native.parent/'intervals.csv').unlink()
    if bug=='unsafe_path':
        p=weather/'manifest.json';m=json.loads(p.read_text());m['files']['../outside']='a'*64;p.write_text(json.dumps(m))
    with pytest.raises(ValueError):prepare(native,weather,tmp_path/'result')
    assert not (tmp_path/'result').exists()


@pytest.mark.parametrize('bug',['station_swap','wrong_parent','export_swap','missing_evidence'])
def test_target_station_binding_is_required(tmp_path,bug):
    native,weather=inputs(tmp_path)
    if bug=='station_swap':
        import yaml
        p=weather/'site_spec.yaml';s=yaml.safe_load(p.read_text());s['sources']['discharge']['station_id']='87654321';p.write_text(yaml.safe_dump(s));rehash(weather,p.name)
    elif bug=='wrong_parent':
        p=weather/'station_catalog.json';d=json.loads(p.read_text());d['assets'][0]['parent_asset_ids']=['absent'];p.write_text(json.dumps(d))
    elif bug=='export_swap':
        p=weather/'station_source_manifest.json';p.write_text('{}')
    else:
        from prepare_corrected_gauge_package import prepare as direct
        with pytest.raises(ValueError,match='station-provenance'):direct(native,weather,tmp_path/'result')
        return
    with pytest.raises((ValueError,KeyError)):prepare(native,weather,tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_projected_coordinates_mislabeled_as_geojson_fail(tmp_path):
    native,weather=inputs(tmp_path)
    p=weather/'basin.geojson'
    p.write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{},'geometry':mapping(box(320000,4300000,321000,4301000))}]}))
    rehash(weather,p.name)
    with pytest.raises(ValueError,match='geometry disagree'):prepare(native,weather,tmp_path/'result')
    assert not (tmp_path/'result').exists()
