"""Raspberry Pi 4 deployment benchmark for SRAS selectors.

This script profiles SRAS selector latency and throughput under RPi-realistic
constraints:
  - CPU-only inference (no CUDA)
  - Single-threaded execution (RPi 4 Cortex-A72, 4 cores but we pin to 1)
  - Conservative pool sizes (10, 20, 30): practical for edge RAG
  - 64-bit ARM float32 precision (no bfloat16)

Usage (on RPi 4 or any CPU host for dry-run):
    python benchmark_rpi.py --config configs/base.yaml
    python benchmark_rpi.py --config configs/base.yaml --dry-run   # no real HW needed

On actual Raspberry Pi 4:
    python benchmark_rpi.py --rpi --threads 1    # single-core (stress test)
    python benchmark_rpi.py --rpi --threads 4    # all cores

Results are saved to results/deployment/rpi4_benchmark.json and compared
against the server-CPU results already in results/deployment/deployment_profile.json.

NOTE: Run this on the actual RPi 4 hardware to get meaningful latency numbers.
      Numbers from x86 CPU are included as "estimated" only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List

import torch

from sras.config.loader import load_config
from sras.models.selector import load_selector
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger
from sras.utils.reproducibility import seed_everything

logger = get_logger("benchmark_rpi")

# RPi 4 target pool sizes, smaller than server benchmark (which goes to 100)
RPI_POOL_SIZES = [10, 20, 30, 50]
RPI_ITERATIONS = 200          # enough for stable p50/p95
RPI_WARMUP     = 20           # warmup iterations before timing
RPI_EMB_DIM    = 384          # sentence-transformer embedding size
RPI_HIDDEN_DIM = 256          # SRAS hidden dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RPi 4 deployment benchmark for SRAS")
    parser.add_argument("--config",     type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/sras_selector_sras_ppo_journal.pt",
                        help="Model checkpoint to benchmark (default: journal model)")
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=RPI_POOL_SIZES)
    parser.add_argument("--iterations", type=int, default=RPI_ITERATIONS)
    parser.add_argument("--threads",    type=int, default=None,
                        help="Number of CPU threads (default: all available). Use 1 for RPi stress test.")
    parser.add_argument("--rpi",        action="store_true",
                        help="Flag: running on actual Raspberry Pi 4 hardware")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Skip actual inference, generate placeholder results for testing")
    return parser.parse_args()


def detect_hardware() -> Dict[str, Any]:
    """Collect hardware info for the results file."""
    info: Dict[str, Any] = {
        "platform":    platform.platform(),
        "processor":   platform.processor(),
        "python":      platform.python_version(),
        "torch":       torch.__version__,
        "cpu_count":   os.cpu_count(),
        "is_rpi":      False,
    }
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "Raspberry Pi" in cpuinfo or "BCM2711" in cpuinfo:
            info["is_rpi"] = True
            info["rpi_model"] = "Raspberry Pi 4" if "BCM2711" in cpuinfo else "unknown RPi"
    except FileNotFoundError:
        pass
    return info


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * p / 100) - 1)
    return sorted_v[idx]


def benchmark_selector(
    model: torch.nn.Module,
    pool_size: int,
    emb_dim: int,
    n_iter: int,
    warmup: int,
    device: torch.device,
) -> Dict[str, float]:
    """Time n_iter forward passes with random embeddings."""
    model.eval()
    latencies: List[float] = []

    with torch.no_grad():
        for i in range(warmup + n_iter):
            q_emb   = torch.randn(1, emb_dim, device=device)
            doc_embs = torch.randn(pool_size, emb_dim, device=device)

            t0 = time.perf_counter()
            _ = model(q_emb, doc_embs)
            t1 = time.perf_counter()

            if i >= warmup:
                latencies.append((t1 - t0) * 1000.0)  # ms

    qps = 1000.0 / (_percentile(latencies, 50) + 1e-9)
    return {
        "p50_ms":  round(_percentile(latencies, 50),  4),
        "p95_ms":  round(_percentile(latencies, 95),  4),
        "p99_ms":  round(_percentile(latencies, 99),  4),
        "mean_ms": round(sum(latencies) / len(latencies), 4),
        "qps_p50": round(qps, 1),
        "n_iter":  n_iter,
    }


def main() -> None:
    args = parse_args()
    seed_everything(42)

    # ── Thread pinning ──────────────────────────────────────────────────────
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        logger.info("Pinned PyTorch to %d thread(s)", args.threads)
    n_threads = torch.get_num_threads()

    # ── Hardware detection ──────────────────────────────────────────────────
    hw = detect_hardware()
    if args.rpi and not hw["is_rpi"]:
        logger.warning("--rpi flag set but /proc/cpuinfo does not identify this as a Raspberry Pi")
    logger.info("Hardware: %s | threads=%d | is_rpi=%s", hw["platform"], n_threads, hw["is_rpi"])

    # ── Config ──────────────────────────────────────────────────────────────
    try:
        config = load_config(args.config, {"evaluation": {"device": "cpu"}})
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    device = torch.device("cpu")

    # ── Model ───────────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        # Try model dir prefix
        ckpt_path = os.path.join("models", os.path.basename(args.checkpoint))
    if not os.path.exists(ckpt_path):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    try:
        model = load_selector(
            ckpt_path, device,
            model_kwargs={
                "doc_emb_dim": config.model.doc_emb_dim,
                "hidden_dim":  config.model.hidden_dim,
                "dropout":     0.0,
            },
        )
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model loaded: %d parameters | checkpoint: %s", n_params, ckpt_path)
    except Exception as e:
        logger.error("Could not load model: %s", e)
        sys.exit(1)

    # ── Benchmark ───────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    emb_dim = config.model.doc_emb_dim

    for pool_size in args.pool_sizes:
        logger.info("Benchmarking pool_size=%d …", pool_size)

        if args.dry_run:
            # Placeholder results: useful for CI and testing without hardware
            result = {
                "p50_ms": round(0.05 * pool_size / 10, 4),
                "p95_ms": round(0.12 * pool_size / 10, 4),
                "p99_ms": round(0.20 * pool_size / 10, 4),
                "mean_ms": round(0.06 * pool_size / 10, 4),
                "qps_p50": round(1000 / (0.05 * pool_size / 10 + 1e-9), 1),
                "n_iter": 0,
                "dry_run": True,
            }
        else:
            result = benchmark_selector(
                model, pool_size, emb_dim,
                args.iterations, RPI_WARMUP, device,
            )

        entry = {
            "pool_size":       pool_size,
            "n_threads":       n_threads,
            "is_rpi_hardware": hw["is_rpi"] or args.rpi,
            **result,
        }
        all_results.append(entry)
        logger.info(
            "pool=%d | p50=%.3fms | p95=%.3fms | QPS=%.0f",
            pool_size, result["p50_ms"], result["p95_ms"], result["qps_p50"],
        )

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "hardware":    hw,
        "n_threads":   n_threads,
        "checkpoint":  ckpt_path,
        "n_params":    n_params,
        "results":     all_results,
        "note": (
            "Run on actual Raspberry Pi 4 hardware for paper-quality numbers. "
            "Use: python benchmark_rpi.py --rpi --threads 1  (single-core stress test)"
            if not (hw["is_rpi"] or args.rpi)
            else "Measured on Raspberry Pi 4 hardware."
        ),
    }

    ensure_dir("results/deployment")
    out_path = "results/deployment/rpi4_benchmark.json"
    save_json(output, out_path)
    logger.info("Results saved to %s", out_path)

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n── RPi 4 Benchmark Summary ─────────────────────────────────────")
    print(f"{'Pool':>6}  {'p50 (ms)':>10}  {'p95 (ms)':>10}  {'QPS':>8}  {'HW':>5}")
    print("-" * 55)
    for r in all_results:
        hw_tag = "RPi4" if r["is_rpi_hardware"] else "x86"
        print(f"{r['pool_size']:>6}  {r['p50_ms']:>10.3f}  {r['p95_ms']:>10.3f}  {r['qps_p50']:>8.0f}  {hw_tag:>5}")
    print(f"\nModel params: {n_params:,}  |  Threads: {n_threads}")
    print(f"Results: {out_path}\n")


if __name__ == "__main__":
    main()
