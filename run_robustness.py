from __future__ import annotations

import argparse
import sys
from typing import List

import torch

from sras.config.loader import load_config
from sras.data.corpus import CorpusStore
from sras.data.datasets import QADataset
from sras.data.embeddings import EmbeddingStore, encode_queries_batch
from sras.evaluation.robustness import RobustnessEvaluator
from sras.models.selector import load_selector
from sras.analysis.visualization import PlotGenerator
from sras.utils.reproducibility import get_device, seed_everything
from sras.utils.logging_utils import get_logger

logger = get_logger("run_robustness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRAS robustness evaluation")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint to evaluate (default: first in registry)")
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    seed_everything(config.evaluation.seed)
    device = get_device(config.evaluation.device)

    checkpoint_path = args.checkpoint or list(config.benchmark.model_registry.values())[0]
    logger.info("Robustness evaluation | checkpoint=%s", checkpoint_path)

    corpus = CorpusStore(config.data.corpus_metadata_path, max_docs=config.data.max_corpus_docs)
    embed_store = EmbeddingStore(config.data.doc_embeddings_path, corpus.doc_ids, device)

    model = load_selector(
        checkpoint_path, device,
        model_kwargs={
            "doc_emb_dim": config.model.doc_emb_dim,
            "hidden_dim": config.model.hidden_dim,
            "dropout": 0.0,
            "use_layer_norm": config.model.use_layer_norm,
            "use_residual": config.model.use_residual,
        },
    )
    model.eval()

    qa_dataset = QADataset(config.data.qa_pairs_path)
    qa_pairs = [{"question": item["question"], "answer": item["answer"]} for item in qa_dataset]
    questions = [qa["question"] for qa in qa_pairs]

    q_embs = encode_queries_batch(
        questions, config.data.embedding_model, device, batch_size=config.evaluation.batch_size
    )

    corpus_docs = [
        {"doc_id": doc_id, "content": corpus.get_text(doc_id)}
        for doc_id in corpus.doc_ids
    ]
    doc_embs = embed_store.get_batch_by_indices(list(range(len(corpus.doc_ids))))

    evaluator = RobustnessEvaluator(config.evaluation.robustness)
    results = evaluator.run_full(
        model, qa_pairs, corpus_docs, doc_embs, q_embs,
        pool_size=args.pool_size, top_k=args.top_k,
    )

    if not args.no_plots:
        plotter = PlotGenerator(output_dir=config.evaluation.figures_dir)
        plotter.plot_robustness_sweep(results)
        if "domain_shift" in results:
            plotter.plot_domain_shift(results["domain_shift"])

    logger.info("Robustness evaluation complete.")


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("run_robustness"):
            main()
    except ImportError:
        main()
