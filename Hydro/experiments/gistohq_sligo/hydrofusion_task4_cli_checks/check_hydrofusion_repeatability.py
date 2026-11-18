#!/usr/bin/env python3
"""Three tiny synthetic fits verify seed handling and validation selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('binary')
    p.add_argument('output')
    args = p.parse_args()
    binary, output = Path(args.binary).resolve(strict=True), Path(args.output).absolute()
    output.mkdir(exist_ok=False, parents=True)
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    template = json.loads((here/'config.json').read_text())
    assert template['use_hydro_package'] is False and template['use_csv_data'] is False
    record = {'binary_sha256':sha(binary), 'checks':{}}
    predictions = []
    # The CLI resolves repository context from the batch path. Keep only the
    # temporary input copies under the repo; retain exact copies with evidence.
    with tempfile.TemporaryDirectory(prefix='.scratch_hydrofusion_repeatability_',dir=repo) as temp:
        temp = Path(temp)
        for name, seed in [('same_a',42), ('same_b',42), ('different',123)]:
            config = dict(template, experiment_id=name, random_seed=seed)
            config_path = temp/(name+'.json')
            config_path.write_text(json.dumps(config,indent=2)+'\n')
            batch = temp/(name+'.batch')
            batch.write_text('ffn_hydrofusion '+config_path.name+'\n')
            (output/config_path.name).write_bytes(config_path.read_bytes())
            (output/batch.name).write_bytes(batch.read_bytes())
            with (output/(name+'.log')).open('w') as log:
                subprocess.run([str(binary),str(batch),str(output/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
            bundle = output/name/name
            hist = list(csv.DictReader((bundle/'training_history.csv').open()))
            if len(hist) != config['epochs']:
                raise AssertionError('Missing training/validation history.')
            minimum = min(hist,key=lambda r:float(r['validation_loss']))
            chosen = [r for r in hist if r['selected_checkpoint']=='1']
            if len(chosen)!=1 or chosen[0]['epoch'] != minimum['epoch']:
                raise AssertionError('Saved checkpoint is not validation-selected.')
            predicted = bundle/'predictions.csv'
            predictions.append(predicted.read_bytes())
            record['checks'][name]={'seed':seed,'predictions_sha256':sha(predicted),
                                   'selected_epoch':int(chosen[0]['epoch']),
                                   'total_epochs':len(hist),'config_sha256':sha(config_path)}
    if predictions[0] != predictions[1]:
        raise AssertionError('Same configured seed did not reproduce predictions exactly.')
    if predictions[0] == predictions[2]:
        raise AssertionError('Configured seed had no effect on this nondegenerate fixture.')
    record['same_seed_exact'] = True
    record['different_seed_changes_predictions'] = True
    (output/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PASS: same-seed exact reproduction, distinct-seed response, validation-selected checkpoints')


if __name__ == '__main__':
    main()
