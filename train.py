from __future__ import annotations

import argparse
import sys

from sras.config.loader import load_config
from sras.training.supervised import SupervisedTrainer
from sras.training.ppo_trainer import PPOTrainer
from sras.utils.logging_utils import get_logger

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRAS selector")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["supervised", "ppo", "full"], default="full",
                        help="Training mode: supervised, ppo, or full (supervised then ppo)")
    parser.add_argument("--name", type=str, default=None, help="Override config name")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--ppo-epochs", type=int, default=None, help="Override ppo_epochs")
    parser.add_argument("--supervised-epochs", type=int, default=None, help="Override supervised_epochs")
    parser.add_argument("--no-supervised-warmup", action="store_true", help="Disable supervised warmup (ablation)")
    parser.add_argument("--no-reward-shaping", action="store_true", help="Disable reward shaping (ablation)")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning (ablation)")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.name:
        overrides["name"] = args.name
    if args.device:
        overrides.setdefault("training", {})["device"] = args.device
    if args.seed is not None:
        overrides.setdefault("training", {})["seed"] = args.seed
    if args.ppo_epochs is not None:
        overrides.setdefault("training", {})["ppo_epochs"] = args.ppo_epochs
    if args.supervised_epochs is not None:
        overrides.setdefault("training", {})["supervised_epochs"] = args.supervised_epochs

    ablation_overrides: dict = {}
    if args.no_supervised_warmup:
        ablation_overrides["use_supervised_warmup"] = False
    if args.no_reward_shaping:
        ablation_overrides["use_reward_shaping"] = False
    if args.no_curriculum:
        ablation_overrides["use_curriculum_learning"] = False
    if ablation_overrides:
        overrides.setdefault("training", {}).setdefault("ablations", {}).update(ablation_overrides)

    return overrides


def main() -> None:
    args = parse_args()
    overrides = build_overrides(args)

    try:
        config = load_config(args.config, overrides)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    logger.info("Starting training | config=%s | mode=%s | name=%s", args.config, args.mode, config.name)

    if args.mode in ("supervised", "full"):
        logger.info("=== Supervised pretraining ===")
        sup_trainer = SupervisedTrainer(config)
        sup_trainer.train()

    if args.mode in ("ppo", "full"):
        logger.info("=== PPO training ===")
        ppo_trainer = PPOTrainer(config, variant_name=config.name)
        ppo_trainer.train()

    logger.info("Training complete.")


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("train"):
            main()
    except ImportError:
        main()
