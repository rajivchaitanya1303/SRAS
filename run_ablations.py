from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

from sras.config.loader import load_config, save_config
from sras.config.schema import AblationConfig
from sras.evaluation.evaluator import SelectorEvaluator
from sras.training.ppo_trainer import PPOTrainer
from sras.analysis.visualization import PlotGenerator
from sras.utils.io import save_json
from sras.utils.logging_utils import get_logger

logger = get_logger("run_ablations")

_CORE_ABLATIONS: Dict[str, Dict] = {
    "sras_ppo_base": {
        "use_supervised_warmup": True,
        "use_reward_shaping": True,
        "use_curriculum_learning": True,
    },
    "sras_ppo_nosw": {
        "use_supervised_warmup": False,
        "use_reward_shaping": True,
        "use_curriculum_learning": True,
    },
    "sras_ppo_nors": {
        "use_supervised_warmup": True,
        "use_reward_shaping": False,
        "use_curriculum_learning": True,
    },
    "sras_ppo_nocl": {
        "use_supervised_warmup": True,
        "use_reward_shaping": True,
        "use_curriculum_learning": False,
    },
}

_EXPANDED_ABLATIONS: Dict[str, Dict] = {
    "ablation_topk_1": {"top_k": 1},
    "ablation_topk_5": {"top_k": 5},
    "ablation_pool_10": {"candidate_pool_size": 10},
    "ablation_pool_50": {"candidate_pool_size": 50},
    "ablation_reward_0507": {"reward_f1_weight": 0.5, "reward_bertscore_weight": 0.7},
    "ablation_reward_0802": {"reward_f1_weight": 0.8, "reward_bertscore_weight": 0.2},
    "ablation_warmup_50": {"supervised_epochs": 50},
    "ablation_warmup_100": {"supervised_epochs": 100},
    "ablation_curriculum_linear": {"curriculum_schedule": "linear"},
    "ablation_curriculum_fixed": {"curriculum_schedule": "fixed"},
}

# ── Journal-extension ablations (novel contributions) ─────────────────────────
# Each variant isolates one novel contribution against the strong ppo_base baseline.
_JOURNAL_ABLATIONS: Dict[str, Dict] = {
    # Novel contribution #1: Contrastive Selector Pre-training (CSP)
    "sras_ppo_csp": {
        "use_supervised_warmup":           True,
        "use_reward_shaping":              True,
        "use_curriculum_learning":         True,
        "use_contrastive_warmup":          True,   # ← CSP ON
        "use_diversity_reward":            False,
        "use_adaptive_budget":             False,
        "use_query_complexity_curriculum": False,
    },
    # Novel contribution #3: Diversity-Aware Reward (DAR)
    "sras_ppo_dar": {
        "use_supervised_warmup":           True,
        "use_reward_shaping":              True,
        "use_curriculum_learning":         True,
        "use_contrastive_warmup":          False,
        "use_diversity_reward":            True,   # ← DAR ON
        "use_adaptive_budget":             False,
        "use_query_complexity_curriculum": False,
    },
    # Novel contribution #2: Adaptive Document Budget (ADB)
    "sras_ppo_adb": {
        "use_supervised_warmup":           True,
        "use_reward_shaping":              True,
        "use_curriculum_learning":         True,
        "use_contrastive_warmup":          False,
        "use_diversity_reward":            False,
        "use_adaptive_budget":             True,   # ← ADB ON
        "use_query_complexity_curriculum": False,
    },
    # Novel contribution #4: Query-Complexity Curriculum (QCC)
    "sras_ppo_qcc": {
        "use_supervised_warmup":           True,
        "use_reward_shaping":              True,
        "use_curriculum_learning":         True,
        "use_contrastive_warmup":          False,
        "use_diversity_reward":            False,
        "use_adaptive_budget":             False,
        "use_query_complexity_curriculum": True,   # ← QCC ON
    },
    # Full journal model: all novel contributions combined
    "sras_ppo_journal": {
        "use_supervised_warmup":           True,
        "use_reward_shaping":              True,
        "use_curriculum_learning":         True,
        "use_contrastive_warmup":          True,   # ← ALL ON
        "use_diversity_reward":            True,
        "use_adaptive_budget":             True,
        "use_query_complexity_curriculum": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRAS ablation variants")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Specific variants to run (default: all core ablations)")
    parser.add_argument("--expanded", action="store_true",
                        help="Include expanded ablations (top-k, pool, reward, warmup, curriculum)")
    parser.add_argument("--journal", action="store_true",
                        help="Include journal-extension ablations (CSP, DAR, ADB, QCC, full journal model)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only run evaluation")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only run training")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed for training and evaluation. "
                             "Appends _seedN to the variant name and checkpoint/result "
                             "filenames so multiple seed runs do not overwrite each other "
                             "(used to measure cross-seed variance).")
    return parser.parse_args()


def run_core_ablation(
    variant_name: str,
    ablation_flags: Dict,
    config_path: str,
    device: str,
    skip_training: bool,
    skip_eval: bool,
    seed: int = None,
) -> Dict:
    overrides: dict = {"name": variant_name}
    if device:
        overrides.setdefault("training", {})["device"] = device
        overrides.setdefault("evaluation", {})["device"] = device
    if seed is not None:
        overrides.setdefault("training", {})["seed"] = seed
        overrides.setdefault("evaluation", {})["seed"] = seed
    overrides.setdefault("training", {})["ablations"] = ablation_flags

    config = load_config(config_path, overrides)

    os.makedirs("logs", exist_ok=True)
    config_out = os.path.join("logs", f"config_{variant_name}.yaml")
    save_config(config, config_out)

    if not skip_training:
        trainer = PPOTrainer(config, ablations=AblationConfig(**ablation_flags), variant_name=variant_name)
        trainer.train()
        logger.info("Training complete for %s", variant_name)

    if skip_eval:
        return {}

    evaluator = SelectorEvaluator(config)
    ckpt_path = os.path.join(config.training.checkpoint_dir, f"sras_selector_{variant_name}.pt")
    if not os.path.exists(ckpt_path):
        logger.warning("Checkpoint not found for %s: %s -- skipping eval", variant_name, ckpt_path)
        return {}

    results = evaluator.run_full_evaluation({variant_name: ckpt_path})
    return results.get(variant_name, {})


def run_expanded_ablation(
    variant_name: str,
    training_overrides: Dict,
    config_path: str,
    device: str,
    skip_training: bool,
    skip_eval: bool,
) -> Dict:
    overrides: dict = {"name": variant_name}
    if device:
        overrides.setdefault("training", {})["device"] = device
        overrides.setdefault("evaluation", {})["device"] = device

    for key in ["top_k", "supervised_epochs", "curriculum_schedule", "curriculum_fixed_docs"]:
        if key in training_overrides:
            overrides.setdefault("training", {})[key] = training_overrides[key]

    for key in ["reward_f1_weight", "reward_bertscore_weight"]:
        if key in training_overrides:
            overrides.setdefault("training", {})[key] = training_overrides[key]

    for key in ["candidate_pool_size"]:
        if key in training_overrides:
            overrides.setdefault("evaluation", {})[key] = training_overrides[key]

    config = load_config(config_path, overrides)

    os.makedirs("logs", exist_ok=True)
    config_out = os.path.join("logs", f"config_{variant_name}.yaml")
    save_config(config, config_out)

    if not skip_training:
        trainer = PPOTrainer(config, variant_name=variant_name)
        trainer.train()
        logger.info("Training complete for %s", variant_name)

    if skip_eval:
        return {}

    evaluator = SelectorEvaluator(config)
    ckpt_path = os.path.join(config.training.checkpoint_dir, f"sras_selector_{variant_name}.pt")
    if not os.path.exists(ckpt_path):
        logger.warning("Checkpoint not found for %s: %s -- skipping eval", variant_name, ckpt_path)
        return {}

    results = evaluator.run_full_evaluation({variant_name: ckpt_path})
    return results.get(variant_name, {})


def main() -> None:
    args = parse_args()

    try:
        base_config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    all_variants = dict(_CORE_ABLATIONS)
    if args.expanded:
        all_variants.update(_EXPANDED_ABLATIONS)
    if args.journal:
        all_variants.update(_JOURNAL_ABLATIONS)

    variants_to_run = args.variants or list(
        list(_CORE_ABLATIONS.keys()) +
        (list(_JOURNAL_ABLATIONS.keys()) if args.journal else [])
    )
    unknown = set(variants_to_run) - set(all_variants.keys())
    if unknown:
        logger.error("Unknown variants: %s. Available: %s", unknown, list(all_variants.keys()))
        sys.exit(1)

    all_eval_results: Dict[str, Dict] = {}

    for base_variant_name in variants_to_run:
        variant_name = (
            f"{base_variant_name}_seed{args.seed}" if args.seed is not None else base_variant_name
        )
        logger.info("=== Variant: %s ===", variant_name)
        flags = all_variants[base_variant_name]

        if base_variant_name in _CORE_ABLATIONS or base_variant_name in _JOURNAL_ABLATIONS:
            # Core ablations AND journal-extension ablations both use run_core_ablation()
            # so that AblationConfig flags (use_contrastive_warmup, use_diversity_reward,
            # use_adaptive_budget, use_query_complexity_curriculum) are correctly applied.
            result = run_core_ablation(
                variant_name, flags, args.config,
                args.device, args.skip_training, args.skip_eval,
                seed=args.seed,
            )
        else:
            result = run_expanded_ablation(
                variant_name, flags, args.config,
                args.device, args.skip_training, args.skip_eval,
            )

        if result:
            all_eval_results[variant_name] = result

    if all_eval_results and not args.skip_eval:
        summary_path = os.path.join(base_config.evaluation.figures_dir, "ablation_eval_summary.json")
        save_json(all_eval_results, summary_path)
        logger.info("Ablation eval summary saved to %s", summary_path)

        plotter = PlotGenerator(output_dir=base_config.evaluation.figures_dir)

        flat_metrics: Dict[str, Dict] = {}
        for variant, ds_results in all_eval_results.items():
            if "internal" in ds_results:
                flat_metrics[variant] = ds_results["internal"].get("metrics", {})

        if flat_metrics:
            ablation_order = [v for v in variants_to_run if v in flat_metrics]
            summary_list = [{"name": v, **flat_metrics[v]} for v in ablation_order]
            core_order = [v for v in ablation_order if v in _CORE_ABLATIONS]
            if core_order:
                plotter.plot_ablation_bar(summary_list, core_order)
            journal_order = [v for v in ablation_order if v in _JOURNAL_ABLATIONS]
            if journal_order:
                plotter.plot_ablation_bar(
                    summary_list, journal_order,
                    output_filename="journal_ablation_bar_plot.pdf",
                )
            plotter.plot_comparison_bar(flat_metrics)

    logger.info("Ablation run complete.")


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("run_ablations"):
            main()
    except ImportError:
        main()
