"""
run_all.py: SRAS end-to-end pipeline runner (journal edition)
==============================================================
Runs every stage in order:

  data          setup_data.py
  contrastive   run_contrastive.py     ← NEW (journal): CSP pre-training
  train         train.py --mode full
  evaluate      evaluate.py            (first pass: ppo_base + supervised + baselines)
  ablations     run_ablations.py --journal
  journal-eval  evaluate.py            (second pass: picks up journal variant checkpoints)
  compress      run_compression.py
  robustness    run_robustness.py
  failure       run_failure_analysis.py
  benchmark     benchmark.py
  deploy        run_deployment.py

Usage:
    python run_all.py                          # full pipeline
    python run_all.py --skip-data              # skip setup_data (already done)
    python run_all.py --skip-train             # skip contrastive + supervised + PPO
    python run_all.py --from evaluate          # start from a specific stage
    python run_all.py --only evaluate          # run a single stage
    python run_all.py --skip-contrastive       # skip CSP (use existing ckpt or random init)
    python run_all.py --no-log                 # disable terminal tee-logging

All terminal output (stdout + stderr from every subprocess) is also written
to a timestamped log file under logs/runs/<timestamp>_run_all.log
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

# ── Tee-logging setup (must happen before any print) ────────────────────────
# Import lazily so the module still works if sras package is not installed yet
try:
    from sras.utils.tee_logger import start_run_log, finish_run_log, tee_subprocess
    _TEE_AVAILABLE = True
except ImportError:
    _TEE_AVAILABLE = False

    # fallback stubs
    def start_run_log(name: str = "sras") -> str:  # type: ignore[misc]
        return ""

    def finish_run_log(path: str, name: str = "sras") -> None:  # type: ignore[misc]
        pass

    def tee_subprocess(cmd, stage, log_file=None, **kw):  # type: ignore[misc]
        import subprocess
        result = subprocess.run(cmd)
        return result.returncode == 0, result.returncode


# ── Stage definitions ────────────────────────────────────────────────────────

STAGES = [
    ("data",         ["python", "setup_data.py",          "--config", "configs/base.yaml"]),
    ("contrastive",  ["python", "run_contrastive.py",     "--config", "configs/base.yaml"]),
    ("train",        ["python", "train.py",               "--mode", "full", "--config", "configs/base.yaml"]),
    ("evaluate",     ["python", "evaluate.py",            "--config", "configs/base.yaml"]),
    ("ablations",    ["python", "run_ablations.py",       "--config", "configs/base.yaml", "--journal"]),
    ("journal-eval", ["python", "evaluate.py",            "--config", "configs/base.yaml"]),
    ("compress",     ["python", "run_compression.py",     "--config", "configs/base.yaml"]),
    ("robustness",   ["python", "run_robustness.py",      "--config", "configs/base.yaml"]),
    ("failure",      ["python", "run_failure_analysis.py","--config", "configs/base.yaml"]),
    ("benchmark",    ["python", "benchmark.py",           "--config", "configs/base.yaml"]),
    ("deploy",       ["python", "run_deployment.py",      "--config", "configs/base.yaml"]),
]

STAGE_NAMES = [s[0] for s in STAGES]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _run(cmd: List[str], stage: str, use_tee: bool = True) -> bool:
    """Run a subprocess, stream output (+ tee to log), return True on success."""
    _banner(f"STAGE: {stage.upper()}  ->  {' '.join(cmd)}")
    start = time.time()

    if use_tee and _TEE_AVAILABLE:
        ok, returncode = tee_subprocess(cmd, stage)
    else:
        import subprocess
        result = subprocess.run(cmd)
        ok = result.returncode == 0
        returncode = result.returncode

    elapsed = time.time() - start
    if not ok:
        print(
            f"\n[run_all] FAILED  Stage '{stage}' FAILED (exit {returncode}) "
            f"after {elapsed:.1f}s",
            file=sys.stderr,
        )
        return False
    print(f"\n[run_all] OK  Stage '{stage}' completed in {elapsed:.1f}s")
    return True


def _stage_index(name: str) -> int:
    try:
        return STAGE_NAMES.index(name)
    except ValueError:
        print(f"[run_all] Unknown stage '{name}'. Valid stages: {STAGE_NAMES}",
              file=sys.stderr)
        sys.exit(1)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRAS end-to-end pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip the data-setup stage (data already prepared)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip contrastive pre-training, supervised + PPO training (checkpoints already exist)",
    )
    parser.add_argument(
        "--skip-contrastive", action="store_true",
        help="Skip only the contrastive pre-training stage (run_contrastive.py)",
    )
    parser.add_argument(
        "--from", dest="from_stage", metavar="STAGE", default=None,
        help="Start from this stage (inclusive). Earlier stages are skipped.",
    )
    parser.add_argument(
        "--only", metavar="STAGE", default=None,
        help="Run only this one stage.",
    )
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Abort the pipeline if any stage exits with a non-zero code.",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable tee-logging (do not write a run log file).",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Start tee-logging ───────────────────────────────────────────────────
    use_tee = (not args.no_log) and _TEE_AVAILABLE
    log_path: str = ""
    if use_tee:
        log_path = start_run_log("run_all")
        print(f"[run_all] Logging all output to: {log_path}")

    try:
        _main_inner(args, use_tee)
    finally:
        if use_tee and log_path:
            finish_run_log(log_path, "run_all")


def _main_inner(args: argparse.Namespace, use_tee: bool) -> None:
    # Determine which stages to run
    if args.only:
        idx = _stage_index(args.only)
        to_run = [STAGES[idx]]
    else:
        start_idx = _stage_index(args.from_stage) if args.from_stage else 0
        to_run = STAGES[start_idx:]

    # Apply skip flags
    skip_names: set = set()
    if args.skip_data:
        skip_names.add("data")
    if args.skip_train:
        skip_names.update({"train", "contrastive"})
    if args.skip_contrastive:
        skip_names.add("contrastive")

    total_start = time.time()
    failures: List[str] = []

    print(f"\n[run_all] Pipeline: {' -> '.join(s[0] for s in to_run)}")
    if skip_names:
        print(f"[run_all] Skipping:  {', '.join(skip_names)}")

    for stage_name, cmd in to_run:
        if stage_name in skip_names:
            _banner(f"STAGE: {stage_name.upper()}  ->  SKIPPED")
            continue

        ok = _run(cmd, stage_name, use_tee=use_tee)
        if not ok:
            failures.append(stage_name)
            if args.stop_on_failure:
                print(f"\n[run_all] Stopping pipeline due to failure in '{stage_name}'.",
                      file=sys.stderr)
                break

    total_elapsed = time.time() - total_start
    _banner(f"PIPELINE COMPLETE -- {total_elapsed/60:.1f} min total")

    if failures:
        print(f"[run_all] WARNING  Stages with errors: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    else:
        print("[run_all] SUCCESS  All stages completed successfully.")


if __name__ == "__main__":
    main()
