from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import List

from sras.config.loader import load_config
from sras.utils.io import ensure_dir, load_json, save_json
from sras.utils.logging_utils import get_logger
from sras.utils.reproducibility import get_device

logger = get_logger("setup_data")

_SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up SRAS data pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--steps", nargs="+",
                        choices=["flatten", "embed", "qa", "rewards", "squad"],
                        default=["flatten", "embed", "qa", "rewards", "squad"],
                        help="Which setup steps to run")
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--squad-subset-n", type=int, default=250)
    return parser.parse_args()


def step_flatten(config, force: bool) -> None:
    out_path = config.data.corpus_path
    if not force and os.path.exists(out_path):
        logger.info("Corpus already exists at %s, skipping flatten", out_path)
        return
    try:
        from corpus_data import CORPUS_DATA
    except ImportError:
        logger.error("corpus_data.py not found. Cannot flatten corpus.")
        sys.exit(1)

    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    import uuid
    flat: List[dict] = []
    for category, docs in CORPUS_DATA.items():
        for text in docs:
            if not text or not text.strip():
                continue
            flat.append({
                "id": str(uuid.uuid4()),
                "text": text.strip(),
                "category": category,
            })

    save_json(flat, out_path)
    logger.info("Flattened corpus: %d documents → %s", len(flat), out_path)


def step_embed(config, device, force: bool) -> None:
    meta_path = config.data.corpus_metadata_path
    emb_path = config.data.doc_embeddings_path

    if not force and os.path.exists(meta_path) and os.path.exists(emb_path):
        logger.info("Embeddings already exist, skipping embed step")
        return

    corpus = load_json(config.data.corpus_path)
    doc_ids = [d["id"] for d in corpus]
    texts = [d["text"] for d in corpus]

    from sras.data.embeddings import build_embeddings
    ensure_dir(os.path.dirname(os.path.abspath(emb_path)))
    build_embeddings(texts, doc_ids, config.data.embedding_model, device, emb_path)

    save_json(corpus, meta_path)
    logger.info("Corpus metadata saved to %s", meta_path)


def step_qa(config, device, force: bool) -> None:
    out_path = config.data.qa_pairs_path
    if not force and os.path.exists(out_path):
        logger.info("QA pairs already exist at %s, skipping qa step", out_path)
        return

    logger.info("Generating QA pairs from corpus using T5 QG...")
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        logger.error("transformers not installed.")
        sys.exit(1)

    corpus = load_json(config.data.corpus_metadata_path)

    import torch
    qa_model_name = "valhalla/t5-base-qg-hl"
    tokenizer = T5Tokenizer.from_pretrained(qa_model_name)
    model = T5ForConditionalGeneration.from_pretrained(qa_model_name).to(device)
    model.eval()

    qa_pairs = []
    max_per_category = 50
    from collections import defaultdict
    category_counts: dict = defaultdict(int)

    for doc in corpus:
        cat = doc.get("category", "unknown")
        if category_counts[cat] >= max_per_category:
            continue
        text = doc["text"][:512]
        prompt = f"generate question: {text}"
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(inp.input_ids, max_length=64)
        question = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if question:
            qa_pairs.append({
                "question": question,
                "answer": text[:200],
                "doc_id": doc["id"],
                "topic": cat,
            })
            category_counts[cat] += 1

    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    save_json(qa_pairs, out_path)
    logger.info("Generated %d QA pairs → %s", len(qa_pairs), out_path)


def step_rewards(config, device, force: bool) -> None:
    out_path = config.data.reward_matrix_path
    if not force and os.path.exists(out_path):
        logger.info("Reward matrix already exists at %s, skipping", out_path)
        return

    logger.info("Computing reward matrix...")
    import torch
    from sentence_transformers import SentenceTransformer, util
    from bert_score import BERTScorer
    from sras.rl.rewards import relaxed_f1

    qa_pairs = load_json(config.data.qa_pairs_path)
    corpus = load_json(config.data.corpus_path)
    doc_id_to_text = {d["id"]: d["text"] for d in corpus}
    doc_ids = [d["id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]

    embed_model = SentenceTransformer(config.data.embedding_model, device=str(device))
    scorer = BERTScorer(lang="en", device=device)

    doc_embs = embed_model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)

    TOP_K = 8
    ALPHA = 0.6
    reward_matrix: dict = {}

    from tqdm import tqdm
    for pair in tqdm(qa_pairs, desc="Computing rewards"):
        question = pair["question"]
        gold = pair["answer"]
        q_emb = embed_model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, doc_embs, top_k=TOP_K)[0]

        entries = []
        for hit in hits:
            doc_id = doc_ids[hit["corpus_id"]]
            text = doc_id_to_text[doc_id]
            f1 = relaxed_f1(text, gold)
            try:
                _, _, F1 = scorer.score([text], [gold])
                bs = float(F1[0].item())
            except Exception:
                bs = f1
            reward = ALPHA * f1 + (1 - ALPHA) * bs
            entries.append({"candidate_doc_id": doc_id, "reward": reward})

        reward_matrix[question] = entries

    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    save_json(reward_matrix, out_path)
    logger.info("Reward matrix saved to %s", out_path)


def step_squad(config, force: bool, subset_n: int = 250) -> None:
    full_path = config.data.squad_full_path
    subset_path = config.data.squad_subset_path

    if not force and os.path.exists(subset_path):
        logger.info("SQuAD subset already exists at %s, skipping", subset_path)
        return

    if not os.path.exists(full_path):
        logger.info("Downloading SQuAD dev set...")
        ensure_dir(os.path.dirname(os.path.abspath(full_path)))
        urllib.request.urlretrieve(_SQUAD_URL, full_path.replace("_pairs", ""))
        logger.info("Downloaded SQuAD dev set")

    raw_path = full_path.replace("_pairs", "")
    if not os.path.exists(raw_path):
        logger.error("SQuAD raw file not found: %s", raw_path)
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        squad_raw = json.load(f)

    pairs = []
    for article in squad_raw.get("data", []):
        for para in article.get("paragraphs", []):
            for qa in para.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                pairs.append({
                    "question": qa["question"].strip(),
                    "answer": answers[0]["text"].strip(),
                    "id": qa.get("id", ""),
                })

    ensure_dir(os.path.dirname(os.path.abspath(full_path)))
    save_json(pairs, full_path)
    logger.info("Parsed %d SQuAD QA pairs → %s", len(pairs), full_path)

    import random
    rng = random.Random(42)
    subset = rng.sample(pairs, min(subset_n, len(pairs)))
    ensure_dir(os.path.dirname(os.path.abspath(subset_path)))
    save_json(subset, subset_path)
    logger.info("SQuAD subset (%d samples) → %s", len(subset), subset_path)


def main() -> None:
    args = parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    device = get_device(args.device or "auto")

    if "flatten" in args.steps:
        step_flatten(config, args.force)

    if "embed" in args.steps:
        step_embed(config, device, args.force)

    if "qa" in args.steps:
        step_qa(config, device, args.force)

    if "rewards" in args.steps:
        step_rewards(config, device, args.force)

    if "squad" in args.steps:
        step_squad(config, args.force, args.squad_subset_n)

    logger.info("Data setup complete.")


if __name__ == "__main__":
    try:
        from sras.utils.tee_logger import RunLogger
        with RunLogger("setup_data"):
            main()
    except ImportError:
        main()
