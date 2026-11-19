import copy
import pytest
from audit_hydrofusion_forcing_semantics import assess


def fixture():
    common = dict(provider='nasa-power', processing_status='native', longitude=-77., latitude=39., spatial_support='provider_point')
    return {'assets': [dict(common, asset_id='et', product='pet-et', variables=['EVPTRNS'],
                           native_units={'EVPTRNS': 'MJ/m^2/day'}, temporal_resolution='daily'),
                       dict(common, asset_id='met', product='historical-meteorology', temporal_resolution='hourly',
                            variables=['PRECTOTCORR','T2M','RH2M','WS2M','ALLSKY_SFC_SW_DWN'],
                            native_units={'PRECTOTCORR':'mm/hour','T2M':'C','RH2M':'%','WS2M':'m/s','ALLSKY_SFC_SW_DWN':'MJ/hr'})]}


def test_energy_conversion_does_not_certify_pet_or_real_time():
    data = fixture()
    original = copy.deepcopy(data)
    report = assess(data)
    assert data == original
    assert not report['research_ready'] and not report['training_authorized']
    assert 'potential/reference ET not established' in report['assets'][0]['scientific_interpretation']
    assert report['latency_review']['published_typical_solar_days'] == [5,7]


@pytest.mark.parametrize('mutation', ['daily_rain', 'duplicate', 'missing_et', 'nan_point', 'mm_et', 'basin_label', 'hourly_et'])
def test_unsupported_metadata_fails_closed(mutation):
    data = fixture()
    if mutation == 'daily_rain': data['assets'][1]['native_units']['PRECTOTCORR'] = 'mm/day'
    if mutation == 'duplicate': data['assets'][1]['asset_id'] = 'et'
    if mutation == 'missing_et': data['assets'].pop(0)
    if mutation == 'nan_point': data['assets'][0]['latitude'] = float('nan')
    if mutation == 'mm_et': data['assets'][0]['native_units']['EVPTRNS'] = 'mm/day'
    if mutation == 'hourly_et': data['assets'][0]['temporal_resolution'] = 'hourly'
    if mutation == 'basin_label': data['assets'][1]['spatial_support'] = 'basin_average'
    with pytest.raises(ValueError): assess(data)
