from __future__ import annotations

import argparse
import sys

import torch

from sras.config.loader import load_config
from sras.compression.evaluator import CompressionEvaluator
from sras.models.selector import load_selector
from sras.analysis.visualization import PlotGenerator
from sras.utils.logging_utils import get_logger

logger = get_logger("run_compression")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRAS compression experiments")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint to compress (default: from config)")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    parser.add_argument("--prune", action="store_true", help="Enable pruning")
    parser.add_argument("--distill", action="store_true", help="Enable distillation")
    parser.add_argument("--prune-amount", type=float, default=None,
                        help="Override pruning sparsity amount [0,1]")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    compression: dict = {}
    if args.quantize:
        compression["use_quantization"] = True
    if args.prune:
        compression["use_pruning"] = True
    if args.distill:
        compression["use_distillation"] = True
    if args.prune_amount is not None:
        compression["pruning_amount"] = args.prune_amount
    if compression:
        overrides["benchmark"] = {"compression": compression}
    return overrides


def main() -> None:
    args = parse_args()
    overrides = build_overrides(args)

    try:
        config = load_config(args.config, overrides)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    checkpoint_path = args.checkpoint or list(config.benchmark.model_registry.values())[0]
    logger.info("Compression study | checkpoint=%s", checkpoint_path)

    # Load the base model from checkpoint
    device = torch.device("cpu")
    try:
        base_model = load_selector(checkpoint_path, device=device)
        base_model.eval()
    except Exception as e:
        logger.error("Failed to load model from %s: %s", checkpoint_path, e)
        sys.exit(1)

    # Infer embedding dimension from the model's query projection weight
    try:
        embed_dim = base_model.query_proj.in_features
    except AttributeError:
        embed_dim = 384  # default: all-MiniLM-L6-v2

    # Create synthetic embeddings for profiling (shape: [50 docs, embed_dim])
    n_docs = 50
    torch.manual_seed(0)
    q_emb = torch.randn(embed_dim)
    doc_embs = torch.randn(n_docs, embed_dim)

    evaluator = CompressionEvaluator(config)
    results = evaluator.run(
        base_model=base_model,
        q_emb=q_emb,
        doc_embs=doc_embs,
        base_checkpoint_path=checkpoint_path,
    )

    if not args.no_plots and results:
        plotter = PlotGenerator(output_dir=config.benchmark.results_dir)
        plotter.plot_compression_comparison(results)

    logger.info("Compression study complete.")


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("run_compression"):
            main()
    except ImportError:
        main()
