from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import torch

from sras.config.loader import load_config
from sras.data.embeddings import EmbeddingStore, encode_query
from sras.data.corpus import CorpusStore
from sras.deployment.edge_runner import EdgeRunner, get_hardware_info
from sras.deployment.profiler import DeploymentProfiler
from sras.models.selector import load_selector
from sras.utils.io import save_json
from sras.utils.logging_utils import get_logger

logger = get_logger("run_deployment")

_DUMMY_QUESTION = "What is the primary mechanism of action of transformer-based language models?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SRAS deployment profiling")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path")
    parser.add_argument("--label", type=str, default=None,
                        help="Label to record results under when --checkpoint is set "
                             "(default: the first key in benchmark.model_registry, which "
                             "is misleading when profiling a different checkpoint).")
    parser.add_argument("--force-cpu", action="store_true", default=True,
                        help="Force CPU inference (default: True for edge)")
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--hardware-info", action="store_true",
                        help="Print hardware info and exit")
    parser.add_argument("--rpi", action="store_true",
                        help="Set benchmark.deployment.is_raspberry_pi=true (enables vcgencmd "
                             "thermal reading) without needing to edit the YAML config on-device.")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    dcfg: dict = {}
    if args.n_iterations is not None:
        dcfg["n_iterations"] = args.n_iterations
    if args.pool_sizes:
        dcfg["candidate_pool_sizes"] = args.pool_sizes
    if args.rpi:
        dcfg["is_raspberry_pi"] = True
    if dcfg:
        overrides["benchmark"] = {"deployment": dcfg}
    return overrides


def main() -> None:
    args = parse_args()

    if args.hardware_info:
        info = get_hardware_info()
        import json
        print(json.dumps(info, indent=2))
        return

    overrides = build_overrides(args)

    try:
        config = load_config(args.config, overrides)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    logger.info("Deployment profiling | config=%s", args.config)

    hw_info = get_hardware_info()
    logger.info(
        "Hardware | Pi=%s | RAM=%.1f GB | CPUs=%s",
        hw_info.get("is_raspberry_pi", False),
        hw_info.get("ram_total_gb", 0.0),
        hw_info.get("cpu_count", "?"),
    )

    corpus = CorpusStore(config.benchmark.corpus_metadata_path)
    embed_store = EmbeddingStore(
        config.data.doc_embeddings_path, corpus.doc_ids, torch.device("cpu")
    )

    q_emb = encode_query(_DUMMY_QUESTION, config.data.embedding_model, torch.device("cpu"))

    model_registry: Dict[str, any] = {}
    for variant_name, checkpoint_path in config.benchmark.model_registry.items():
        if args.checkpoint:
            checkpoint_path = args.checkpoint
            if args.label:
                variant_name = args.label
        try:
            runner = EdgeRunner(checkpoint_path, force_cpu=args.force_cpu)
            model_registry[variant_name] = runner.model
            logger.info("Loaded %s: params=%d", variant_name, runner.model.count_parameters())
        except FileNotFoundError:
            logger.warning("Checkpoint not found for %s: %s -- skipping", variant_name, checkpoint_path)
        except Exception as e:
            logger.error("Failed to load %s: %s", variant_name, e)

        if args.checkpoint:
            break

    if not model_registry:
        logger.error("No models loaded. Exiting.")
        sys.exit(1)

    max_pool = max(config.benchmark.deployment.candidate_pool_sizes)
    n_docs = min(max_pool, embed_store.size())
    doc_embs = embed_store.get_batch_by_indices(list(range(n_docs))).cpu()

    profiler = DeploymentProfiler(config)
    results = profiler.run_full(model_registry, [q_emb], doc_embs)

    hw_path = f"{config.benchmark.deployment.results_dir}/hardware_info.json"
    save_json(hw_info, hw_path)
    logger.info("Hardware info saved to %s", hw_path)
    logger.info("Deployment profiling complete. %d result entries.", len(results))


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("run_deployment"):
            main()
    except ImportError:
        main()
