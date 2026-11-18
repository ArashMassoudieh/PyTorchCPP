#!/usr/bin/env python3
"""Freeze fold-specific regional-pretraining and fine-tuning configs.

Every donor-basin window ends before the calendar year containing the Sligo
test event. This removes the shared-storm-time explanation left open by the
original 2015-2022 regional pilot while preserving its basin lists,
architecture, seeds and local fold definitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FOLDS = {
    "A": {"regional_end": "2020-01-01T00:00:00Z", "test_end_x": 50251},
    "B": {"regional_end": "2021-01-01T00:00:00Z", "test_end_x": 58784},
    "C": {"regional_end": "2022-01-01T00:00:00Z", "test_end_x": 66910},
}
SEEDS = (42, 123, 27182)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regional-template", type=Path, required=True)
    parser.add_argument("--local-template-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    regional_dir = args.output / "regional"
    finetune_dir = args.output / "finetune"
    regional_dir.mkdir(exist_ok=True)
    finetune_dir.mkdir(exist_ok=True)

    regional_template = json.loads(args.regional_template.read_text())
    jobs = []
    for fold, fold_spec in FOLDS.items():
        local_template_path = args.local_template_dir / f"control_fold{fold}.json"
        local_template = json.loads(local_template_path.read_text())
        if int(local_template["test_end_x"]) != fold_spec["test_end_x"]:
            raise RuntimeError(f"Fold {fold} local template has a different frozen test_end_x")
        for seed in SEEDS:
            pretrain_id = f"calendar_safe_regional_fold{fold}_s{seed}"
            pretrain = dict(regional_template)
            pretrain.update({
                "experiment_id": pretrain_id,
                "random_seed": seed,
                "regional_window_start": "2015-01-01T00:00:00Z",
                "regional_window_end": fold_spec["regional_end"],
            })
            pretrain_path = (regional_dir / f"{pretrain_id}.json").resolve()
            pretrain_batch = (regional_dir / f"{pretrain_id}.batch").resolve()
            write_json(pretrain_path, pretrain)
            pretrain_batch.write_text(f"lstm_regional_pretrain {pretrain_path}\n")
            checkpoint = (args.run_root / "regional" / pretrain_id /
                          pretrain_id / "models" / "lstm_regional_pretrain.pt").resolve()

            finetune_id = f"calendar_safe_finetune_fold{fold}_s{seed}"
            finetune = dict(local_template)
            finetune.update({
                "experiment_id": finetune_id,
                "random_seed": seed,
                "lstm_warm_start_checkpoint_path": str(checkpoint),
            })
            finetune_path = (finetune_dir / f"{finetune_id}.json").resolve()
            finetune_batch = (finetune_dir / f"{finetune_id}.batch").resolve()
            write_json(finetune_path, finetune)
            finetune_batch.write_text(f"lstm {finetune_path}\n")
            jobs.append({
                "fold": fold,
                "seed": seed,
                "regional_end_exclusive": fold_spec["regional_end"],
                "test_end_x": fold_spec["test_end_x"],
                "pretrain_id": pretrain_id,
                "pretrain_config": str(pretrain_path),
                "pretrain_config_sha256": digest(pretrain_path),
                "pretrain_batch": str(pretrain_batch),
                "checkpoint": str(checkpoint),
                "finetune_id": finetune_id,
                "finetune_config": str(finetune_path),
                "finetune_config_sha256": digest(finetune_path),
                "finetune_batch": str(finetune_batch),
            })

    manifest = {
        "purpose": "calendar-safe regional pretraining replication",
        "interpretation": (
            "Donor observations end before the Sligo test-event calendar year; "
            "the reserved 2023-10-15+ target period remains sealed."
        ),
        "regional_template": str(args.regional_template.resolve()),
        "regional_template_sha256": digest(args.regional_template),
        "local_template_dir": str(args.local_template_dir.resolve()),
        "seeds": list(SEEDS),
        "folds": FOLDS,
        "jobs": jobs,
    }
    write_json(args.output / "run_manifest.json", manifest)
    print(f"Wrote {len(jobs)} paired calendar-safe jobs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
