#!/usr/bin/env python3
"""Review catalog metadata only; never open source observations or rewrite inputs.

This is a scientific preflight report, not a loader gate or training consent.
A correct energy conversion does not turn model ET into potential/reference ET.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

SOURCES = {
    'power_parameter_definition': 'https://gis.earthdata.nasa.gov/portal/rest/services/POWER/POWER_MONTHLY_METEOROLOGY_LST/ImageServer/multiDimensionalInfo',
    'power_hourly_units': 'https://power.larc.nasa.gov/docs/services/api/temporal/hourly/',
    'power_latency_and_spatial_support': 'https://power.larc.nasa.gov/docs/faqs/data/',
    'energy_to_water': 'https://www.fao.org/4/X0490E/x0490e04.htm',
}


def assess(catalog):
    assets = [a for a in catalog['assets'] if a.get('provider') == 'nasa-power'
              and a.get('processing_status') == 'native']
    identities = [a.get('asset_id') for a in assets]
    if not all(isinstance(i, str) and i for i in identities) or len(set(identities)) != len(identities):
        raise ValueError('Unique nonempty native asset identities required')
    products = {a.get('product') for a in assets}
    if products != {'pet-et', 'historical-meteorology'} or len(assets) != 2:
        raise ValueError('Exactly one native meteorology and one native ET asset required')
    evidence = []
    blockers = ['Provider grid-cell identity and upstream-basin representativeness unresolved',
                'Operational delivery times have not been established for archived inputs']
    for asset in assets:
        for key, limit in [('longitude', 180), ('latitude', 90)]:
            value = asset.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
                raise ValueError('Invalid provider request coordinates')
        units = asset.get('native_units', {})
        if asset['product'] == 'pet-et':
            if asset.get('temporal_resolution') != 'daily':
                raise ValueError('Daily ET energy contract required')
            if asset.get('variables') != ['EVPTRNS'] or units.get('EVPTRNS') not in {'MJ/m^2/day', 'MJ/m2/day'}:
                raise ValueError('This review supports only the documented EVPTRNS energy contract')
            semantics = 'provider_evapotranspiration_energy_flux; potential/reference ET not established'
            blockers.append('EVPTRNS energy-to-depth conversion is not proof of potential/reference ET semantics')
        else:
            required = {'PRECTOTCORR', 'T2M', 'RH2M', 'WS2M', 'ALLSKY_SFC_SW_DWN'}
            if set(asset.get('variables', [])) != required or set(units) != required:
                raise ValueError('Complete native meteorology variable/unit metadata required')
            if asset.get('temporal_resolution') != 'hourly' or units['PRECTOTCORR'] != 'mm/hour':
                raise ValueError('Hourly precipitation-rate contract required')
            semantics = 'provider meteorology; not an established catchment-average forcing'
        evidence.append({k: asset.get(k) for k in ['asset_id', 'product', 'native_units',
                         'temporal_resolution', 'spatial_support', 'longitude', 'latitude']})
        evidence[-1]['scientific_interpretation'] = semantics
        if asset.get('spatial_support') != 'provider_point':
            raise ValueError('Unexpected support type; requires a separate spatial review')
    return {'schema': 'HydroFusionForcingSemanticsReview-v1', 'research_ready': False,
            'training_authorized': False, 'reads_observations': False, 'assets': evidence,
            'et_conversion': {'formula': 'energy_MJ_m2_day / 2.45 / 24',
                              'output_unit': 'mm/hour', 'latent_heat_MJ_kg': 2.45,
                              'meaning': 'Equivalent ET water depth, not a PET estimation method'},
            'latency_review': {'published_typical_meteorology_days': [2, 3],
                               'published_typical_solar_days': [5, 7],
                               'limitation': 'Current documentation is not a historical per-record delivery log; do not infer exact availability'},
            'blockers': blockers, 'sources': SOURCES,
            'limitations': 'Metadata-only review. Does not prove forcing invalidity, cell equivalence, basin averages, or model skill. Existing archive recipes remain unchanged.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('new_output_directory', type=Path)
    args = parser.parse_args()
    if args.new_output_directory.exists():
        raise FileExistsError(args.new_output_directory)
    raw = args.catalog.read_bytes()
    report = assess(json.loads(raw))
    report['catalog_sha256'] = hashlib.sha256(raw).hexdigest()
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.new_output_directory.mkdir(parents=True, exist_ok=False)
    (args.new_output_directory / 'forcing_semantics_review.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Metadata review saved; new research inputs remain blocked.')


if __name__ == '__main__':
    main()
