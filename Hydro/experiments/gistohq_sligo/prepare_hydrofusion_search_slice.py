#!/usr/bin/env python3
"""Create a NEW train/validation-only native package; never convert later Q tokens."""
import argparse
import hashlib
import json
from pathlib import Path

EPOCH = 1420070400
SEAL = 77016

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(source, output, validation_end_x):
    source, output = Path(source), Path(output)
    if isinstance(validation_end_x, bool) or not isinstance(validation_end_x, int) or not 1 <= validation_end_x < SEAL:
        raise ValueError('Presealed integer validation cutoff required')
    manifest = json.loads(source.read_text())
    if manifest['interval_csv'] != 'intervals.csv':
        raise ValueError('Local intervals.csv required')
    csv = source.parent / 'intervals.csv'
    if sha(csv) != manifest['interval_csv_sha256']:
        raise ValueError('Source CSV hash mismatch')
    if manifest['end_interval_x_exclusive'] > SEAL:
        raise ValueError('Reserved-period source forbidden')
    output.mkdir(parents=True, exist_ok=False)
    count = 0
    previous = None
    with csv.open() as inp, (output/'intervals.csv').open('x') as out:
        out.write(next(inp))
        for line in inp:
            # Only identify the timestamp. Do not parse any later observation.
            token = line.partition(',')[0]
            if not token.isdecimal():
                raise ValueError('Malformed timestamp')
            epoch = int(token)
            x, rem = divmod(epoch-EPOCH, 3600)
            if rem or x < 0 or x >= SEAL or previous is not None and x != previous+1:
                raise ValueError('Nonhourly/unordered/sealed source identity')
            previous = x
            # Native interval x is adapted to discharge identity x+1.
            if x+1 <= validation_end_x:
                out.write(line)
                count += 1
    if not count or previous is None or validation_end_x > previous+1:
        raise ValueError('Requested validation end not covered')
    manifest.update(row_count=count, end_interval_x_exclusive=validation_end_x,
                    interval_csv_sha256=sha(output/'intervals.csv'),
                    execution_scope='train_validation_only',
                    parent_manifest_sha256=sha(source), parent_csv_sha256=sha(csv),
                    target_cutoff_x=validation_end_x,
                    slicer_sha256=sha(__file__))
    manifest['daily_pet'] = [row for row in manifest['daily_pet']
                             if row['day_start_epoch_seconds'] < EPOCH+validation_end_x*3600]
    path = output/'native_input_manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source_manifest'); p.add_argument('new_output'); p.add_argument('validation_end_x', type=int)
    a = p.parse_args()
    print(prepare(a.source_manifest, a.new_output, a.validation_end_x))
