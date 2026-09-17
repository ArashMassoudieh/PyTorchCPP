#!/usr/bin/env python3
"""Paper tuning refinement with bias-aware validation selection for real catchments.

This is a thin override around run_adaptive_full_pipeline. Synthetic verification
is unchanged. For real rainfall-runoff data it:
  * keeps test data fully held out;
  * ranks non-degenerate candidates by validation KGE, NSE, |PBIAS|, then RMSE;
  * refines the process-aware LSTM+PINN routing grid around the boundary solution
    found by the previous paper run, without increasing the sweep size drastically;
  * shards independent HydroBatch jobs across multiple processes so the paper
    pipeline can use the host CPU instead of running every experiment serially.

Parallelism is process-level on purpose. HydroBatch currently pins LibTorch to one
intra-op/inter-op thread, so independent processes are the safest way to use a
multi-core workstation without introducing shared LibTorch/model state. Set
HYDROPINN_BATCH_JOBS to override the automatic worker count; set it to 1 for the
legacy serial execution path.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_adaptive_full_pipeline as base


def winner(a, rows, mode):
    candidates = [r for r in rows if r.get("mode") == mode]
    if not candidates:
        raise RuntimeError(f"No candidates for mode={mode}")
    if a.data_source == "synthetic":
        return min(candidates, key=lambda r: (base.finite_float(r.get("validation_mse")), r.get("experiment_id", "")))

    def key(r):
        degenerate = r.get("validation_near_constant", "false") == "true"
        kge = base.finite_float(r.get("validation_kge"), -math.inf)
        nse = base.finite_float(r.get("validation_nse"), -math.inf)
        pbias = abs(base.finite_float(r.get("validation_pbias"), math.inf))
        rmse = base.finite_float(r.get("validation_rmse"), math.inf)
        return (1 if degenerate else 0, -kge, -nse, pbias, rmse, r.get("experiment_id", ""))

    return min(candidates, key=key)


def _batch_workers(job_count: int) -> int:
    override = os.environ.get("HYDROPINN_BATCH_JOBS", "").strip()
    if override:
        try:
            requested = int(override)
        except ValueError as exc:
            raise RuntimeError("HYDROPINN_BATCH_JOBS must be an integer >= 1") from exc
        if requested < 1:
            raise RuntimeError("HYDROPINN_BATCH_JOBS must be >= 1")
    else:
        # HydroBatch is currently one LibTorch thread per process. Use at most
        # half the logical CPUs by default to leave memory/IO headroom on long
        # LSTM/PINN jobs. On the 24-core Hydro workstation this selects 12.
        requested = max(1, (os.cpu_count() or 1) // 2)
    return max(1, min(requested, job_count))


def _active_batch_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    if not lines:
        raise RuntimeError(f"Generated HydroBatch file contains no jobs: {path}")
    return lines


def _contiguous_chunks(lines: list[str], count: int) -> list[list[str]]:
    count = max(1, min(count, len(lines)))
    q, r = divmod(len(lines), count)
    chunks: list[list[str]] = []
    start = 0
    for i in range(count):
        size = q + (1 if i < r else 0)
        chunks.append(lines[start:start + size])
        start += size
    return chunks


def _merge_shard_outputs(shard_roots: list[Path], out: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for shard_root in shard_roots:
        rows.extend(base.load_summary(shard_root / "batch_summary.csv"))
        for child in shard_root.iterdir():
            if child.name == "batch_summary.csv":
                continue
            destination = out / child.name
            if destination.exists():
                raise RuntimeError(
                    f"Parallel HydroBatch produced duplicate output path: {destination}"
                )
            if child.is_dir():
                shutil.move(str(child), str(destination))
            else:
                shutil.move(str(child), str(destination))
    return rows


def parallel_run_generated(a, generator_args, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base.run([sys.executable, str(base.GENERATOR), *generator_args, *base.source_args(a)])

    lines = _active_batch_lines(base.BATCH_FILE)
    workers = _batch_workers(len(lines))
    if workers == 1:
        print("[adaptive] HydroBatch execution: serial (HYDROPINN_BATCH_JOBS=1)", flush=True)
        base.run(
            [str(a.hydrobatch.resolve()), str(base.BATCH_FILE.resolve()), str(out.resolve())],
            cwd=base.HERE.parent.parent.parent,
        )
        rows = base.load_summary(out / "batch_summary.csv")
        base.annotate_validation(rows, out)
        base.write_rows(out / "batch_summary.csv", rows)
        return rows

    print(
        f"[adaptive] HydroBatch execution: {len(lines)} independent jobs across {workers} processes "
        f"(logical_cpus={os.cpu_count() or 1}; override with HYDROPINN_BATCH_JOBS)",
        flush=True,
    )

    shard_parent = out / ".parallel_shards"
    if shard_parent.exists():
        shutil.rmtree(shard_parent)
    shard_parent.mkdir(parents=True)

    chunks = _contiguous_chunks(lines, workers)
    batch_files: list[Path] = []
    shard_roots: list[Path] = []
    # Keep shard batch files beside unified_sweep.batch so its relative config
    # paths retain exactly the same interpretation as the serial runner.
    for i, chunk in enumerate(chunks):
        batch_file = base.BATCH_FILE.parent / f".unified_sweep.parallel_{os.getpid()}_{i:02d}.batch"
        batch_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        batch_files.append(batch_file)
        shard_root = shard_parent / f"shard_{i:02d}"
        shard_root.mkdir(parents=True)
        shard_roots.append(shard_root)

    def run_shard(index: int) -> None:
        cmd = [
            str(a.hydrobatch.resolve()),
            str(batch_files[index].resolve()),
            str(shard_roots[index].resolve()),
        ]
        print(f"[adaptive] shard {index + 1}/{workers}: {' '.join(cmd)}", flush=True)
        subprocess.run(
            cmd,
            cwd=base.HERE.parent.parent.parent,
            check=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_shard, i): i for i in range(workers)}
            for future in as_completed(futures):
                i = futures[future]
                future.result()
                print(f"[adaptive] shard {i + 1}/{workers} complete", flush=True)

        rows = _merge_shard_outputs(shard_roots, out)
        base.annotate_validation(rows, out)
        base.write_rows(out / "batch_summary.csv", rows)
        return rows
    finally:
        for batch_file in batch_files:
            try:
                batch_file.unlink()
            except FileNotFoundError:
                pass
        if shard_parent.exists():
            shutil.rmtree(shard_parent)


def stage2(a, root, s1):
    if a.data_source == "synthetic":
        return ORIGINAL_STAGE2(a, root, s1)

    print("\n[adaptive] STAGE 2: bias-aware process/physics tuning", flush=True)
    ffn, lstm = s1["ffn"], s1["lstm"]
    stage_root = root / "02_stage2_physics"

    legacy_rows = base.run_generated(a, [
        "--methods", "ffn_pinn,pinn",
        "--ffn-architectures", base.q(ffn, "hidden_layers", "16,16"),
        "--ffn-activations", base.q(ffn, "activation", "relu"),
        "--pinn-architectures", "16,16;24,24;32,32",
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0.000001,0.00001,0.0001,0.0005,0.001,0.005,0.01,0.025",
        "--recession-k", "0.0025,0.005,0.01,0.02,0.04,0.08,0.16",
    ], stage_root / "legacy")

    hybrid_architectures = base.unique_semicolon([
        base.q(lstm, "hidden_layers", "48"),
        "32,16", "48,24", "48,32,16", "64,32,16"
    ])
    arch_rows = base.run_generated(a, [
        "--methods", "lstm_pinn",
        "--lstm-architectures", hybrid_architectures,
        "--lstm-sequences", "6,12,24,48",
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0.0001",
        "--lstm-pinn-profile", base.PROCESS_PROFILE,
        "--fast-k", "0.10", "--slow-k", "0.04", "--routing-alpha", "0.85",
    ], stage_root / "lstm_pinn_architecture")
    arch_winner = winner(a, arch_rows, "lstm_pinn")

    # Previous validation-selected solution landed on fast_k=0.10, slow_k=0.04,
    # alpha=0.85 and w=1e-4, several of them at/near grid boundaries. Refine that
    # neighborhood while keeping the number of expensive jobs controlled.
    routing_rows = base.run_generated(a, [
        "--methods", "lstm_pinn",
        "--lstm-architectures", base.q(arch_winner, "hidden_layers", "64,32,16"),
        "--lstm-sequences", base.q(arch_winner, "lstm_sequence_length", "48"),
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0,0.00003,0.0001,0.0003",
        "--lstm-pinn-profile", base.PROCESS_PROFILE,
        "--fast-k", "0.05,0.10,0.20",
        "--slow-k", "0.02,0.04,0.06",
        "--routing-alpha", "0.75,0.85,0.95",
    ], stage_root / "lstm_pinn_routing")

    rows = legacy_rows + arch_rows + routing_rows
    base.write_rows(stage_root / "batch_summary.csv", rows)
    winners = {
        "ffn_pinn": winner(a, legacy_rows, "ffn_pinn"),
        "pinn": winner(a, legacy_rows, "pinn"),
        "lstm_pinn": winner(a, routing_rows, "lstm_pinn"),
    }
    return rows, winners


ORIGINAL_STAGE2 = base.stage2
base.run_generated = parallel_run_generated
base.winner = winner
base.stage2 = stage2
base.selection_label = lambda a: (
    "validation_mse" if a.data_source == "synthetic"
    else "validation_KGE_then_NSE_then_absPBIAS_then_RMSE_non_degenerate"
)

if __name__ == "__main__":
    raise SystemExit(base.main())
