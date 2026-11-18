#!/usr/bin/env python3
"""Run frozen regional-pretrain/fine-tune pairs with verified resume."""
import argparse
import concurrent.futures
import csv
import json
import math
from pathlib import Path
import subprocess
import time

from run_isolated_batch import sha, validate_generated_wrapper_build


RESERVED_START_X = 77016


def verified_file(root, identity, artifact, artifact_key):
    marker = root / "VERIFIED.json"
    if not marker.exists():
        return False
    old = json.loads(marker.read_text())
    if not all(old.get(key) == value for key, value in identity.items()):
        raise ValueError(f"Existing verification has different provenance: {root}")
    if not artifact.exists() or sha(artifact) != old.get(artifact_key):
        raise ValueError(f"Verified artifact is missing or changed: {artifact}")
    return True


def run_pair(job, binary, output_root):
    pair_start = time.time()
    pretrain_root = output_root / "regional" / job["pretrain_id"]
    finetune_root = output_root / "finetune" / job["finetune_id"]
    pretrain_root.mkdir(parents=True, exist_ok=True)
    finetune_root.mkdir(parents=True, exist_ok=True)
    binary_hash = sha(binary)

    pretrain_identity = {
        "config_sha256": sha(job["pretrain_config"]),
        "binary_sha256": binary_hash,
    }
    if pretrain_identity["config_sha256"] != job["pretrain_config_sha256"]:
        raise ValueError(f"Frozen pretrain config changed: {job['pretrain_id']}")
    checkpoint = Path(job["checkpoint"])
    pretrain_verified = verified_file(
        pretrain_root, pretrain_identity, checkpoint, "checkpoint_sha256"
    )
    if not pretrain_verified:
        nested = pretrain_root / job["pretrain_id"]
        if nested.exists():
            raise ValueError(f"Unverified pretrain output needs inspection: {nested}")
        start = time.time()
        with (pretrain_root / "run.log").open("w") as log:
            subprocess.run(
                [str(binary), job["pretrain_batch"], str(pretrain_root)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = nested / "regional_pretrain_summary.csv"
        if not checkpoint.exists() or not summary.exists():
            raise ValueError(f"Regional outputs incomplete: {job['pretrain_id']}")
        config = json.loads(Path(job["pretrain_config"]).read_text())
        with summary.open() as handle:
            epochs = list(csv.DictReader(handle))
        if len(epochs) != int(config["epochs"]):
            raise ValueError(
                f"Regional epoch count mismatch for {job['pretrain_id']}: {len(epochs)}"
            )
        pretrain_result = {
            "id": job["pretrain_id"],
            "status": "complete",
            "seconds": time.time() - start,
            "epochs": len(epochs),
            "checkpoint_sha256": sha(checkpoint),
            **pretrain_identity,
        }
        (pretrain_root / "VERIFIED.json").write_text(
            json.dumps(pretrain_result, indent=2) + "\n"
        )

    finetune_identity = {
        "config_sha256": sha(job["finetune_config"]),
        "binary_sha256": binary_hash,
        "checkpoint_sha256": sha(checkpoint),
    }
    if finetune_identity["config_sha256"] != job["finetune_config_sha256"]:
        raise ValueError(f"Frozen fine-tune config changed: {job['finetune_id']}")
    nested = finetune_root / job["finetune_id"]
    prediction = nested / "predictions.csv"
    if verified_file(finetune_root, finetune_identity, prediction, "predictions_sha256"):
        return {
            "id": job["finetune_id"],
            "status": "verified_resume",
            "seconds": time.time() - pair_start,
        }
    recovered = nested.exists()
    if recovered and (not prediction.exists() or "[batch] complete: jobs=1 failures=0" not in (finetune_root / "run.log").read_text()):
        raise ValueError(f"Unverified partial fine-tune output needs inspection: {nested}")

    start = time.time()
    if not recovered:
      with (finetune_root / "run.log").open("w") as log:
        subprocess.run(
            [str(binary), job["finetune_batch"], str(finetune_root)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not prediction.exists():
        raise ValueError(f"Fine-tune predictions missing: {job['finetune_id']}")
    config = json.loads(Path(job["finetune_config"]).read_text())
    bounds = {key: int(config.get(key, -1)) for key in ("train_end_x", "validation_end_x", "test_end_x")}
    with prediction.open() as handle:
        total = sum(1 for _ in csv.DictReader(handle))
    train_count = int(total * min(0.9, max(0.1, float(config["train_fraction"]))))
    validation_end_count = train_count + max(1, int(total * min(0.4, max(0.01, float(config["validation_fraction"])))))
    if bounds["test_end_x"] != int(job["test_end_x"]):
        raise ValueError(f"Frozen test bound changed: {job['finetune_id']}")
    counts = {split: 0 for split in ("train", "validation", "test", "beyond_test_end")}
    previous = -math.inf
    with prediction.open() as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            x = float(row["x"])
            if x <= previous:
                raise ValueError(f"Duplicate or unordered time: {job['finetune_id']}")
            previous = x
            if x >= RESERVED_START_X:
                raise ValueError(f"Reserved-period row exported: {job['finetune_id']}")
            if not all(math.isfinite(float(row[key])) for key in ("x", "observed", "predicted")):
                raise ValueError(f"Nonfinite prediction: {job['finetune_id']}")
            if bounds["train_end_x"] < 0:
                expected = ("train" if position < train_count else
                            "validation" if position < validation_end_count else
                            "test" if x <= bounds["test_end_x"] else "beyond_test_end")
            else:
              expected = (
                "train" if x <= bounds["train_end_x"] else
                "validation" if x <= bounds["validation_end_x"] else
                "test" if x <= bounds["test_end_x"] else
                "beyond_test_end"
            )
            if row["split"] != expected:
                raise ValueError(f"Split mismatch at {x}: {job['finetune_id']}")
            counts[expected] += 1
    if not all(counts[split] > 0 for split in ("train", "validation", "test")):
        raise ValueError(f"Empty required split: {job['finetune_id']}")
    result = {
        "id": job["finetune_id"],
        "status": "complete",
        "seconds": time.time() - start,
        "pair_seconds": time.time() - pair_start,
        "fold": job["fold"],
        "seed": job["seed"],
        "counts": counts,
        "predictions_sha256": sha(prediction),
        **finetune_identity,
    }
    (finetune_root / "VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    validate_generated_wrapper_build(args.binary)
    manifest = json.loads(args.manifest.read_text())
    jobs = manifest["jobs"]
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}
        iterator = iter(jobs)
        completed = 0
        failed = False

        def fill():
            while len(active) < args.workers:
                job = next(iterator, None)
                if job is None:
                    return
                future = pool.submit(
                    run_pair, job, args.binary.resolve(), args.output.resolve()
                )
                active[future] = job["finetune_id"]

        fill()
        while active:
            done, _ = concurrent.futures.wait(
                active, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                name = active.pop(future)
                try:
                    result = future.result()
                    completed += 1
                except Exception as error:
                    failed = True
                    result = {"id": name, "status": "failed", "error": str(error)}
                with (args.output / "progress.jsonl").open("a") as handle:
                    handle.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
            print(
                json.dumps(
                    {
                        "completed_pairs": completed,
                        "total_pairs": len(jobs),
                        "active": list(active.values()),
                        "failed": failed,
                    }
                ),
                flush=True,
            )
            if not failed:
                fill()
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
