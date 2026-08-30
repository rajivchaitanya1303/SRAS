from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

from sras.config.loader import load_config
from sras.evaluation.evaluator import SelectorEvaluator
from sras.evaluation.failure_analysis import FailureAnalyzer
from sras.analysis.visualization import PlotGenerator
from sras.utils.io import save_json
from sras.utils.logging_utils import get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SRAS selectors")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=None, help="Override evaluation datasets")
    parser.add_argument("--pool-size", type=int, default=None, help="Override candidate pool size")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k")
    parser.add_argument("--noise", type=float, default=None, help="Noise distractor rate [0,1]")
    parser.add_argument("--redundant", type=float, default=None, help="Redundant distractor rate [0,1]")
    parser.add_argument("--adversarial", type=float, default=None, help="Adversarial distractor rate [0,1]")
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-failure-analysis", action="store_true", help="Skip failure analysis plots")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.device:
        overrides.setdefault("evaluation", {})["device"] = args.device
    if args.datasets:
        overrides.setdefault("evaluation", {})["datasets"] = args.datasets
    if args.pool_size is not None:
        overrides.setdefault("evaluation", {})["candidate_pool_size"] = args.pool_size
    if args.top_k is not None:
        overrides.setdefault("evaluation", {})["top_k"] = args.top_k
    if args.noise is not None:
        overrides.setdefault("evaluation", {})["noise_distractor_rate"] = args.noise
    if args.redundant is not None:
        overrides.setdefault("evaluation", {})["redundant_distractor_rate"] = args.redundant
    if args.adversarial is not None:
        overrides.setdefault("evaluation", {})["adversarial_distractor_rate"] = args.adversarial
    if args.no_baselines:
        overrides.setdefault("evaluation", {})["run_baselines"] = False
    return overrides


def main() -> None:
    args = parse_args()
    overrides = build_overrides(args)

    try:
        config = load_config(args.config, overrides)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    logger.info("Starting evaluation | config=%s | datasets=%s", args.config, config.evaluation.datasets)

    evaluator = SelectorEvaluator(config)

    selector_registry: Dict[str, str] = {}
    for variant_name, ckpt_path in config.benchmark.model_registry.items():
        selector_registry[variant_name] = ckpt_path

    all_results = evaluator.run_full_evaluation(selector_registry)

    summary_path = os.path.join(config.evaluation.figures_dir, "summary_eval.json")
    save_json(all_results, summary_path)
    logger.info("Summary saved to %s", summary_path)

    if not args.no_plots and all_results:
        plotter = PlotGenerator(output_dir=config.evaluation.figures_dir)

        flat_metrics: Dict[str, Dict] = {}
        for variant, ds_results in all_results.items():
            if "internal" in ds_results:
                flat_metrics[variant] = ds_results["internal"].get("metrics", {})
            elif ds_results:
                first_ds = next(iter(ds_results.values()))
                flat_metrics[variant] = first_ds.get("metrics", {})

        if flat_metrics:
            plotter.plot_comparison_bar(flat_metrics)

        log_paths: Dict[str, str] = {}
        for variant in selector_registry:
            lp = os.path.join(config.training.log_dir, f"ppo_training_log_{variant}.json")
            if os.path.exists(lp):
                log_paths[variant] = lp
        if log_paths:
            plotter.plot_reward_curves(log_paths)

        if not args.no_failure_analysis:
            for variant, ds_results in all_results.items():
                for ds_name, ds_result in ds_results.items():
                    if "per_question_type" in ds_result:
                        try:
                            plotter.plot_failure_breakdown(
                                ds_result,
                                output_filename=f"failure_breakdown_{variant}_{ds_name}.pdf",
                            )
                        except Exception as e:
                            logger.warning("Failure breakdown plot failed: %s", e)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("evaluate"):
            main()
    except ImportError:
        main()
