"""
setup_squad_eval.py
-------------------
One-time script that builds a proper SQuAD evaluation corpus for SRAS.

What it does
~~~~~~~~~~~~
1. Downloads the SQuAD v1.1 validation split from HuggingFace (cached in
   data/hf_cache so re-runs are free).
2. Samples `--n_questions` QA triples that each have a non-empty context
   paragraph and a short answer (≤ 10 tokens).
3. Deduplicates context passages and saves them as corpus documents to
   data/squad_contexts.json  (format matches flat_corpus.json).
4. Saves QA pairs with their context doc_id to
   data/squad_eval_pairs.json.
5. Pre-computes sentence-transformer embeddings for the SQuAD context
   documents and saves them to data/squad_doc_embeddings.pt.

After running this script, evaluate.py will automatically use it when
'squad_eval' is listed in evaluation.datasets in the config.

Usage
~~~~~
    python setup_squad_eval.py                        # defaults
    python setup_squad_eval.py --n_questions 500      # larger sample
    python setup_squad_eval.py --n_distractors 29     # pool = 1 correct + 29 random
    python setup_squad_eval.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from sras.utils.logging_utils import get_logger

try:
    from sras.utils.tee_logger import RunLogger
    _HAS_TEE = True
except ImportError:
    _HAS_TEE = False

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DEFAULT_CACHE_DIR    = "data/hf_cache"
DEFAULT_CONTEXTS_OUT = "data/squad_contexts.json"
DEFAULT_PAIRS_OUT    = "data/squad_eval_pairs.json"
DEFAULT_EMBEDS_OUT   = "data/squad_doc_embeddings.pt"
DEFAULT_N_QUESTIONS  = 300
DEFAULT_MAX_ANSWER_TOKENS = 10
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SQuAD evaluation corpus")
    p.add_argument("--config",          default="configs/base.yaml")
    p.add_argument("--n_questions",     type=int, default=DEFAULT_N_QUESTIONS)
    p.add_argument("--max_answer_tokens", type=int, default=DEFAULT_MAX_ANSWER_TOKENS,
                   help="Discard QA pairs whose answer is longer than this many tokens")
    p.add_argument("--cache_dir",       default=DEFAULT_CACHE_DIR)
    p.add_argument("--contexts_out",    default=DEFAULT_CONTEXTS_OUT)
    p.add_argument("--pairs_out",       default=DEFAULT_PAIRS_OUT)
    p.add_argument("--embeds_out",      default=DEFAULT_EMBEDS_OUT)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    p.add_argument("--device",          default="cpu")
    return p.parse_args()


# ------------------------------------------------------------------
def load_squad_validation(cache_dir: str) -> list:
    """Load SQuAD v1.1 validation split. Returns flat list of
    {question, answer, context} dicts."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' library is not installed. "
            "Run: pip install datasets"
        )

    logger.info("Loading SQuAD v1.1 validation split (cached at %s)...", cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset("squad", split="validation", cache_dir=cache_dir)

    flat = []
    for item in ds:
        context  = item["context"].strip()
        question = item["question"].strip()
        answers  = item["answers"]["text"]
        if not answers:
            continue
        answer = answers[0].strip()
        flat.append({"question": question, "answer": answer, "context": context})

    logger.info("SQuAD validation loaded: %d QA triples", len(flat))
    return flat


# ------------------------------------------------------------------
def filter_and_sample(
    triples: list,
    n: int,
    max_answer_tokens: int,
    seed: int,
) -> list:
    """Keep triples with short answers; sample n."""
    filtered = [
        t for t in triples
        if t["answer"] and len(t["answer"].split()) <= max_answer_tokens
    ]
    logger.info(
        "%d / %d triples have answer <= %d tokens",
        len(filtered), len(triples), max_answer_tokens,
    )
    rng = random.Random(seed)
    rng.shuffle(filtered)
    selected = filtered[:n]
    logger.info("Sampled %d triples", len(selected))
    return selected


# ------------------------------------------------------------------
def build_corpus_and_pairs(triples: list):
    """Deduplicate contexts, assign doc IDs, return (corpus_docs, qa_pairs)."""
    text_to_id: dict = {}
    corpus_docs = []

    qa_pairs = []
    for i, t in enumerate(triples):
        ctx = t["context"]
        if ctx not in text_to_id:
            doc_id = "squad_ctx_%04d" % len(corpus_docs)
            text_to_id[ctx] = doc_id
            corpus_docs.append({
                "id":       doc_id,
                "text":     ctx,
                "category": "squad_wikipedia",
            })
        qa_pairs.append({
            "question":   t["question"],
            "answer":     t["answer"],
            "context_doc_id": text_to_id[ctx],
        })

    logger.info(
        "Built %d corpus docs from %d QA pairs (%d duplicates merged)",
        len(corpus_docs), len(triples), len(triples) - len(corpus_docs),
    )
    return corpus_docs, qa_pairs


# ------------------------------------------------------------------
def compute_embeddings(
    corpus_docs: list,
    embedding_model: str,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute sentence-transformer embeddings for corpus docs."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", embedding_model)
    model = SentenceTransformer(embedding_model, device=device)

    texts = [doc["text"] for doc in corpus_docs]
    logger.info("Encoding %d documents...", len(texts))

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding SQuAD contexts"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embs.append(embs.cpu())

    embeddings = torch.cat(all_embs, dim=0)
    logger.info("Embeddings shape: %s", list(embeddings.shape))
    return embeddings


# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # 1. Load SQuAD
    triples = load_squad_validation(args.cache_dir)

    # 2. Filter and sample
    triples = filter_and_sample(triples, args.n_questions, args.max_answer_tokens, args.seed)

    # 3. Build corpus + QA pairs
    corpus_docs, qa_pairs = build_corpus_and_pairs(triples)

    # 4. Save corpus
    os.makedirs(os.path.dirname(args.contexts_out) or ".", exist_ok=True)
    with open(args.contexts_out, "w", encoding="utf-8") as f:
        json.dump(corpus_docs, f, indent=2, ensure_ascii=False)
    logger.info("Saved SQuAD corpus to %s (%d docs)", args.contexts_out, len(corpus_docs))

    # 5. Save QA pairs
    with open(args.pairs_out, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    logger.info("Saved SQuAD QA pairs to %s (%d pairs)", args.pairs_out, len(qa_pairs))

    # 6. Compute and save embeddings
    embeddings = compute_embeddings(corpus_docs, args.embedding_model, args.device)

    # Save as a plain tensor: EmbeddingStore expects torch.Tensor
    torch.save(embeddings, args.embeds_out)
    logger.info("Saved SQuAD embeddings to %s", args.embeds_out)

    print()
    print("=" * 60)
    print("SQuAD evaluation corpus ready.")
    print("  Corpus docs :  %d  ->  %s" % (len(corpus_docs), args.contexts_out))
    print("  QA pairs    :  %d  ->  %s" % (len(qa_pairs),    args.pairs_out))
    print("  Embeddings  :  %s  ->  %s" % (list(embeddings.shape), args.embeds_out))
    print()
    print("Next step: add 'squad_eval' to evaluation.datasets in configs/base.yaml")
    print("and run:  python run_all.py --skip-contrastive --only evaluate")
    print("=" * 60)


if __name__ == "__main__":
    if _HAS_TEE:
        with RunLogger("setup_squad_eval"):
            main()
    else:
        main()
