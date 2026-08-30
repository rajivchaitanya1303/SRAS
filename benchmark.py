from __future__ import annotations

import argparse
import sys

from sras.config.loader import load_config
from sras.evaluation.benchmarking import EdgeBenchmarker
from sras.analysis.visualization import PlotGenerator
from sras.utils.io import load_json
from sras.utils.logging_utils import get_logger

logger = get_logger("benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge benchmarking for SRAS selectors")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--iterations", type=int, default=None, help="Override n_iterations")
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=None, help="Override pool sizes")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.device:
        overrides.setdefault("benchmark", {})["device"] = args.device
    if args.iterations is not None:
        overrides.setdefault("benchmark", {})["n_iterations"] = args.iterations
    if args.pool_sizes:
        overrides.setdefault("benchmark", {})["candidate_pool_sizes"] = args.pool_sizes
    return overrides


def main() -> None:
    args = parse_args()
    overrides = build_overrides(args)

    try:
        config = load_config(args.config, overrides)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    logger.info("Starting edge benchmark | config=%s", args.config)

    benchmarker = EdgeBenchmarker(config)
    results = benchmarker.run()

    if not args.no_plots and results:
        plotter = PlotGenerator(output_dir=config.benchmark.results_dir)
        summary = [
            {
                "name": r["variant"],
                "avg_latency_ms": r.get("avg_latency_ms", 0.0),
                "num_params": r.get("num_params", 0),
            }
            for r in results
            if r.get("pool_size") == config.benchmark.candidate_pool_sizes[-1]
        ]

        eval_summary_path = f"{config.evaluation.figures_dir}/summary_eval.json"
        try:
            eval_summary = load_json(eval_summary_path)
            for entry in summary:
                variant = entry["name"]
                if variant in eval_summary:
                    ds_results = eval_summary[variant]
                    if "internal" in ds_results:
                        m = ds_results["internal"].get("metrics", {})
                    elif ds_results:
                        m = next(iter(ds_results.values())).get("metrics", {})
                    else:
                        m = {}
                    entry["relaxed_f1"] = m.get("relaxed_f1", 0.0)
                    entry["bertscore_f1"] = m.get("bertscore_f1", 0.0)
        except (FileNotFoundError, KeyError):
            pass

        if summary:
            plotter.plot_eval_vs_latency(summary)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("benchmark"):
            main()
    except ImportError:
        main()
