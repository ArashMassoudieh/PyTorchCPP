#!/usr/bin/env python3
"""Presealed GIStoOHQ 1.2 evidence audit; never a training authorization.

Only run on an independently prepared presealed export. Reads no observation
asset until manifest bounds pass. Qualifier acceptance is an explicit policy,
not a statement about rating-curve uncertainty or real-time availability.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

EPOCH = 1420070400
SEAL = EPOCH + 77016 * 3600
HEADER = ['timestamp_utc', 'variable', 'value', 'native_unit', 'provider_qualifiers']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(data):
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def number(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not numeric')
    n = float(value)
    if not math.isfinite(n):
        raise ValueError('nonfinite number')
    return n


def timestamp(text):
    # Full UTC seconds only. Dates in manifest handled separately, inclusively.
    dt = datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    if dt.strftime('%Y-%m-%dT%H:%M:%SZ') != text:
        raise ValueError('noncanonical UTC timestamp')
    return int(dt.timestamp())


def boundary(text, end=False):
    if len(text) == 10:
        return timestamp(text + 'T00:00:00Z') + (86400 if end else 0)
    return timestamp(text) + (3600 if end else 0)


def audit(root, policy):
    root = Path(root).resolve(strict=True)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = read_json(manifest_bytes)
    if sha(manifest_bytes) != policy['manifest_sha256']:
        raise ValueError('manifest hash mismatch')
    if (manifest['schema_name'], manifest['schema_version'], manifest['profile']) != (
            'HydroPINNExport', '1.2', 'water-balance-v1'):
        raise ValueError('unsupported export schema/profile')
    start = boundary(manifest['study_start'])
    end = boundary(manifest['study_end'], True)
    # Crucially before ANY observation asset is opened (even for hashing).
    if not EPOCH <= start < end <= SEAL or start % 3600 or end % 3600:
        raise ValueError('export must be wholly presealed and hourly bounded')
    area = number(manifest['catchment_area_m2'])
    if area <= 0 or not manifest['catchment_area_source']:
        raise ValueError('missing positive area/source')
    if manifest['source_package_id'] != policy['source_package_id']:
        raise ValueError('source identity mismatch')
    if not isinstance(policy['qualifier_policy_evidence'], str) or not policy['qualifier_policy_evidence'].strip():
        raise ValueError('explicit qualifier-policy evidence required')
    allowed = policy['accepted_discharge_qualifier_sets']
    if not isinstance(allowed, list) or not allowed or any(not isinstance(v, str) for v in allowed) or len(set(allowed)) != len(allowed):
        raise ValueError('explicit unique qualifier sets required')
    coverage_min = number(policy['minimum_native_hour_coverage'])
    if not 0 < coverage_min <= 1:
        raise ValueError('invalid native coverage threshold')
    assets = manifest['assets']
    if not assets or not isinstance(assets, list):
        raise ValueError('no assets')
    identities, paths, all_rows, asset_hashes = set(), set(), {}, {}
    for asset in assets:
        identity, relative = asset['asset_id'], asset['path']
        path = PurePosixPath(relative)
        if (not isinstance(identity, str) or not identity or identity in identities or
                path.is_absolute() or '..' in path.parts or path.parts[:1] != ('observations',) or
                str(path) != relative or relative in paths):
            raise ValueError('duplicate/unsafe asset identity or path')
        identities.add(identity)
        paths.add(relative)
        target = root / relative
        # No symlinks anywhere, including within root.
        if any((root.joinpath(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)):
            raise ValueError('symlink asset forbidden')
        data = target.read_bytes()
        digest = sha(data)
        if digest != asset['sha256']:
            raise ValueError('asset hash mismatch')
        asset_hashes[identity] = digest
        reader = csv.reader(io.StringIO(data.decode('utf-8'), newline=''), strict=True)
        if next(reader, None) != HEADER:
            raise ValueError('unexpected producer CSV header')
        for row in reader:
            if len(row) != 5:
                raise ValueError('producer CSV requires five fields')
            t = timestamp(row[0])
            if not start <= t < end:
                raise ValueError('asset timestamp outside presealed manifest')
            key = row[1], t
            if key in all_rows:
                raise ValueError('duplicate variable/timestamp across assets')
            # Numeric conversion postponed until all row timestamps pass.
            all_rows[key] = (row, identity)
    if set(policy['discharge_sample_support_seconds']) - identities:
        raise ValueError('policy references unknown asset')
    discharge, inventory, qualifier_counts = [], {}, {}
    for (variable, t), (row, identity) in sorted(all_rows.items()):
        inventory.setdefault(variable, {'rows': 0, 'missing': 0, 'units': set()})
        entry = inventory[variable]
        entry['rows'] += 1
        entry['units'].add(row[3])
        value = None if row[2] == '' else number(row[2])
        entry['missing'] += value is None
        if variable != '00060':
            continue
        if row[3] not in ('ft3/s', 'ft³/s', 'cfs', 'm3/s'):
            raise ValueError('unknown discharge unit')
        duration = policy['discharge_sample_support_seconds'].get(identity)
        if type(duration) is not int or not 0 < duration <= 3600:
            raise ValueError('explicit discharge sample support required')
        qualifier_counts[row[4]] = qualifier_counts.get(row[4], 0) + 1
        if value is not None and value < 0:
            raise ValueError('negative discharge')
        accepted = value is not None and row[4] in allowed
        discharge.append((t, duration, value, row[3], accepted))
    if not discharge:
        raise ValueError('no discharge records')
    hours = {}
    for i, (t, duration, value, unit, accepted) in enumerate(discharge):
        # Every record, including rejected/missing records, terminates prior support.
        stop = min(t + duration, end, discharge[i + 1][0] if i + 1 < len(discharge) else end)
        if not accepted:
            continue
        q = value if unit == 'm3/s' else value * 0.028316846592
        if not math.isfinite(q):
            raise ValueError('unit conversion overflow')
        while t < stop:
            hour = t // 3600 * 3600
            seconds = min(stop, hour + 3600) - t
            volume, covered = hours.get(hour, (0., 0))
            volume += q * seconds
            if not math.isfinite(volume):
                raise ValueError('volume overflow')
            hours[hour] = volume, covered + seconds
            t += seconds
    support = []
    for t in range(start, end, 3600):
        volume, seconds = hours.get(t, (0., 0))
        coverage = seconds / 3600
        valid = coverage >= coverage_min
        q = volume / seconds if valid else None
        depth = q * 3600000 / area if valid else None
        if depth is not None and not math.isfinite(depth):
            raise ValueError('area conversion overflow')
        support.append({'interval_start_x': (t - EPOCH) // 3600,
                        'interval_end_x_exclusive': (t - EPOCH) // 3600 + 1,
                        'native_coverage': coverage,
                        'eligible_discharge': valid, 'q_m3_s': q, 'q_mm_h': depth})
    for entry in inventory.values():
        entry['units'] = sorted(entry['units'])
    return {'schema': 'HydroFusionProducerAudit-v1', 'manifest_sha256': sha(manifest_bytes),
            'policy_sha256': sha(json.dumps(policy, sort_keys=True, allow_nan=False).encode()),
            'asset_sha256': asset_hashes, 'catchment_area_m2': area,
            'source_package_id': manifest['source_package_id'], 'variable_inventory': inventory,
            'source_qc_status': manifest.get('source_package_qc_status'),
            'source_failed_qc_rule_ids': manifest.get('source_failed_qc_rule_ids'),
            'qualifier_counts': qualifier_counts, 'native_discharge_hours': support,
            'research_ready': False, 'training_authorized': False,
            'unresolved': ['hourly interval-end availability versus forecast-origin labeling',
                           'forcing units/cadence/availability and daily PET causality',
                           'variables.json consistency and source QC/policy acceptance',
                           'independent qualifier-policy and sample-support acceptance',
                           'matched all-method support and real-data loader integration',
                           'explicit research budget approval']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('policy', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.package, read_json(args.policy.read_bytes()))
    # Never overwrite an existing scientific record.
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
