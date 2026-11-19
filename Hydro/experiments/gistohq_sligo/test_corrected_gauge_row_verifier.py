import csv,json,hashlib
import pytest
from prepare_hydrofusion_causal_package import CSV_HEADER
from verify_corrected_gauge_rows import verify,EPOCH


def fixture(tmp_path):
    path=tmp_path/'intervals.csv'
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(CSV_HEADER)
        for h in range(2):w.writerow([EPOCH+h*3600,h+1,0,5,50,1,.1,0,1,1,1,0,1,'fake'])
    m={'schema':'HydroFusionPreparedNative-v2','first_interval_x':0,'end_interval_x_exclusive':2,'row_count':2,
       'interval_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'daily_pet':[],'catchment_area_m2':3600000,
       'hourly_forcing_delay_hours':0,'daily_pet_delay_hours':0,'daily_pet_utc_start_hour':7}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(m))
    decoded=tmp_path/'decoded.csv';decoded.write_text('x,q,p,t,rh,wind,solar,pet,forcing_valid,discharge_valid\n1,1,0,5,50,1,.1,0,0,1\n2,2,0,5,50,1,.1,0,0,1\n')
    return manifest,decoded


def test_valid_mapping_and_no_overwrite(tmp_path):
    m,d=fixture(tmp_path);out=tmp_path/'receipt.json'
    assert verify(m,d,out)['rows_checked']==2
    with pytest.raises(FileExistsError):verify(m,d,out)


@pytest.mark.parametrize('bad',['nan','inf','99'])
def test_nonfinite_comparison_cannot_pass_as_zero_error(tmp_path,bad):
    m,d=fixture(tmp_path);d.write_text(d.read_text().replace('1,1,0,5',f'1,{bad},0,5'))
    with pytest.raises(ValueError):verify(m,d,tmp_path/'receipt.json')
    assert not (tmp_path/'receipt.json').exists()
