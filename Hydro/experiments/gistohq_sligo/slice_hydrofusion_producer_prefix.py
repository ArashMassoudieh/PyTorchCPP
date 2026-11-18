#!/usr/bin/env python3
"""Create a presealed slice without reading target values at/after the cutoff.

Requires GIStoOHQ's canonical timestamp-first, sorted native CSV contract. Does
not verify the parent asset's whole-file hash (doing so would read reserved
values). Records that limitation explicitly, and hashes the exact copied prefix.
"""
import argparse
import csv
import io
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from audit_hydrofusion_producer_package import HEADER, EPOCH, SEAL, read_json, sha, timestamp

HEADER_BYTES = (','.join(HEADER) + '\n').encode()


def copy_prefix(source, destination, end):
    if not EPOCH < end <= SEAL or end % 3600:
        raise ValueError('invalid preseal exclusive cutoff')
    previous = None
    count = 0
    with source.open('rb', buffering=0) as stream, destination.open('xb') as output:
        header = stream.readline()
        if header != HEADER_BYTES:
            raise ValueError('canonical producer header required')
        output.write(header)
        while True:
            # EXACT 20-byte canonical timestamp plus comma. No buffered read or
            # readline until this timestamp has passed the seal check.
            token = stream.read(21)
            if not token:
                break
            if len(token) != 21 or token[-1:] != b',':
                raise ValueError('canonical timestamp-first producer rows required')
            t = timestamp(token[:20].decode('ascii'))
            if previous is not None and t < previous:
                raise ValueError('producer rows must be sorted by UTC timestamp')
            previous = t
            if t >= end:
                # Do not read the variable/value/qualifiers of this record.
                break
            if t < EPOCH:
                raise ValueError('source must start at/after development epoch')
            rest = stream.readline()
            if not rest.endswith(b'\n'):
                raise ValueError('unterminated producer record')
            # Producer fields may be quoted, but embedded record newlines are
            # deliberately unsupported. Full audit validates values after slicing.
            row = next(csv.reader(io.StringIO((token + rest).decode('utf-8')), strict=True))
            if len(row) != 5:
                raise ValueError('five-field producer record required')
            output.write(token + rest)
            count += 1
    return count


def slice_package(source, destination, end=SEAL):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    raw = (source / 'manifest.json').read_bytes()
    manifest = read_json(raw)
    if (manifest['schema_name'], manifest['schema_version'], manifest['profile']) != ('HydroPINNExport', '1.2', 'water-balance-v1'):
        raise ValueError('unsupported producer schema')
    if timestamp(manifest['study_start']) >= end:
        raise ValueError('no development overlap')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.hydrofusion-preseal-', dir=destination.parent))
    try:
        (stage / 'observations').mkdir()
        identities, paths, provenance = set(), set(), []
        for entry in manifest['assets']:
            relative = entry['path']
            path = PurePosixPath(relative)
            if (path.is_absolute() or '..' in path.parts or path.parts[:1] != ('observations',) or str(path) != relative
                    or relative in paths or entry['asset_id'] in identities):
                raise ValueError('unsafe/duplicate producer asset')
            paths.add(relative)
            identities.add(entry['asset_id'])
            if any(source.joinpath(*path.parts[:i]).is_symlink() for i in range(1, len(path.parts) + 1)):
                raise ValueError('symlink asset forbidden')
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            count = copy_prefix(source / relative, target, end)
            prefix_hash = sha(target.read_bytes())
            provenance.append({'asset_id': entry['asset_id'], 'rows_copied': count,
                               'copied_prefix_sha256': prefix_hash,
                               'parent_declared_sha256': entry['sha256'],
                               'parent_whole_file_hash_verified': False})
            entry['sha256'] = prefix_hash
        from datetime import datetime, timezone
        manifest['study_end'] = datetime.fromtimestamp(end - 3600, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # Metadata only; contains variable names/units, never target observations.
        (stage / 'variables.json').write_bytes((source / 'variables.json').read_bytes())
        (stage / 'preseal_slice_provenance.json').write_text(json.dumps({
            'schema': 'HydroFusionPresealPrefix-v1', 'parent_manifest_sha256': sha(raw),
            'end_epoch_seconds_exclusive': end, 'assets': provenance,
            'limitation': 'Whole parent assets not hashed to avoid reserved observations; prefix integrity is independently pinned.',
            'training_authorized': False}, indent=2) + '\n')
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--end-exclusive', default=SEAL, type=int)
    args = parser.parse_args()
    slice_package(args.source, args.destination, args.end_exclusive)
