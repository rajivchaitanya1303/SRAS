"""
run_contrastive.py
==================
Runs Contrastive Selector Pre-training (CSP): journal contribution #1.

Usage
-----
    python run_contrastive.py                           # default config
    python run_contrastive.py --config configs/base.yaml
    python run_contrastive.py --epochs 30 --lr 5e-5 --temperature 0.05

The trained checkpoint is saved to config.training.contrastive_checkpoint
(default: models/sras_selector_contrastive.pt) and is automatically picked up
by run_train.py (supervised) and run_ppo.py (PPO variants that use CSP).
"""
from __future__ import annotations

import argparse
import sys

from sras.config.loader import load_config
from sras.training.contrastive_trainer import ContrastiveTrainer
from sras.utils.logging_utils import get_logger

logger = get_logger("run_contrastive")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive Selector Pre-training (CSP)")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--epochs",      type=int,   default=None, help="Override contrastive_epochs")
    p.add_argument("--lr",          type=float, default=None, help="Override contrastive_lr")
    p.add_argument("--temperature", type=float, default=None, help="Override contrastive_temperature")
    p.add_argument("--n-negatives", type=int,   default=None, help="Override contrastive_n_negatives")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides: dict = {}
    training_ovr: dict = {}
    if args.epochs      is not None: training_ovr["contrastive_epochs"]      = args.epochs
    if args.lr          is not None: training_ovr["contrastive_lr"]          = args.lr
    if args.temperature is not None: training_ovr["contrastive_temperature"] = args.temperature
    if args.n_negatives is not None: training_ovr["contrastive_n_negatives"] = args.n_negatives
    if training_ovr:
        overrides["training"] = training_ovr

    try:
        config = load_config(args.config, overrides)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    logger.info(
        "Starting CSP | epochs=%d | lr=%.2e | tau=%.3f | n_neg=%d",
        config.training.contrastive_epochs,
        config.training.contrastive_lr,
        config.training.contrastive_temperature,
        config.training.contrastive_n_negatives,
    )

    trainer = ContrastiveTrainer(config)
    model = trainer.train()

    logger.info(
        "CSP complete. Checkpoint saved to: %s",
        config.training.contrastive_checkpoint,
    )
    logger.info("Parameters: %d", model.count_parameters())


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("run_contrastive"):
            main()
    except ImportError:
        main()
