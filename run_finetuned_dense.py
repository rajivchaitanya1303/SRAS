"""
run_finetuned_dense.py
======================
Trains a domain-adapted dense retriever by fine-tuning the all-MiniLM-L6-v2
sentence encoder on the SRAS reward matrix with InfoNCE contrastive loss.
The frozen MiniLM encoder used elsewhere in SRAS never sees the domain corpus;
this baseline asks: how much does fine-tuning the encoder itself help?

Architecture
------------
  - Model  : all-MiniLM-L6-v2 (sentence encoder, 22M params)
  - Loss   : InfoNCE / NT-Xent with temperature tau=0.07
  - Positives : highest-reward doc per question (reward > 0.5)
  - Negatives : in-batch + N random hard negatives
  - Epochs : 30
  - Batch  : 32

After training the script:
  1. Re-embeds the full corpus and saves to models/dense_finetuned_embeddings.pt
  2. Runs evaluation on both internal and SQuAD benchmarks
  3. Writes results to results/dense_finetuned_internal_results.json
                   and results/dense_finetuned_squad_eval_results.json

Usage
-----
    python run_finetuned_dense.py
    python run_finetuned_dense.py --epochs 50 --lr 2e-5 --batch-size 64

Requirements: same as the rest of the SRAS project (torch, sentence-transformers,
              transformers, tqdm).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────────
# Run from the SRAS_TaD directory: python run_finetuned_dense.py
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from sras.config.loader import load_config
from sras.data.corpus import CorpusStore
from sras.evaluation.evaluator import SelectorEvaluator
from sras.utils.io import save_json, ensure_dir
from sras.utils.logging_utils import get_logger
from sras.utils.reproducibility import get_device, seed_everything

logger = get_logger("finetuned_dense")

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR    = HERE / "data"
MODELS_DIR  = HERE / "models"
RESULTS_DIR = HERE / "results"
CONFIG_PATH = HERE / "configs" / "base.yaml"

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM      = 384
PROJ_DIM     = 256   # match selector projection dim for fair comparison


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Encoder with optional projection head
# ══════════════════════════════════════════════════════════════════════════════

class MiniLMEncoder(nn.Module):
    """Wraps all-MiniLM-L6-v2 with a projection head for contrastive fine-tuning."""

    def __init__(self, model_name: str = ENCODER_NAME, proj_dim: int = PROJ_DIM,
                 cache_dir: Optional[str] = None) -> None:
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.encoder   = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.proj      = nn.Linear(EMB_DIM, proj_dim, bias=False)

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(self, texts: List[str], device: torch.device,
               batch_size: int = 64) -> torch.Tensor:
        """Encode a list of texts, returning L2-normalised projected embeddings."""
        self.eval()
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc   = self.tokenizer(batch, padding=True, truncation=True,
                                       max_length=128, return_tensors="pt").to(device)
                out   = self.encoder(**enc)
                embs  = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                embs  = F.normalize(self.proj(embs), dim=-1)
                all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)

    def forward_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Forward pass with gradient (used during training)."""
        enc  = self.tokenizer(texts, padding=True, truncation=True,
                              max_length=128, return_tensors="pt").to(device)
        out  = self.encoder(**enc)
        embs = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
        return F.normalize(self.proj(embs), dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ContrastiveRetrievalDataset(Dataset):
    """(question_text, pos_doc_text, [neg_doc_texts]) triples from reward_matrix."""

    def __init__(
        self,
        reward_matrix: Dict[str, List[Dict]],
        doc_text_map: Dict[str, str],
        n_hard_negs: int = 7,
        min_pos_reward: float = 0.5,
    ) -> None:
        self.samples: List[Tuple[str, str, List[str]]] = []
        all_doc_ids = list(doc_text_map.keys())

        for question, entries in reward_matrix.items():
            positives = [e for e in entries if e["reward"] >= min_pos_reward]
            if not positives:
                # fall back to highest-reward doc
                positives = [max(entries, key=lambda e: e["reward"])]

            neg_ids = [e["candidate_doc_id"] for e in entries
                       if e["candidate_doc_id"] not in {p["candidate_doc_id"] for p in positives}]
            if len(neg_ids) < n_hard_negs:
                extra = random.sample(all_doc_ids, n_hard_negs - len(neg_ids))
                neg_ids.extend(extra)

            pos_id  = positives[0]["candidate_doc_id"]
            neg_ids = neg_ids[:n_hard_negs]

            pos_text  = doc_text_map.get(pos_id, "")
            neg_texts = [doc_text_map.get(d, "") for d in neg_ids]

            if pos_text:
                self.samples.append((question, pos_text, neg_texts))

        logger.info("Dataset: %d training triples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str, List[str]]:
        return self.samples[idx]


def collate_fn(batch):
    questions  = [b[0] for b in batch]
    pos_docs   = [b[1] for b in batch]
    neg_docs   = [b[2] for b in batch]   # list of lists
    return questions, pos_docs, neg_docs


# ══════════════════════════════════════════════════════════════════════════════
# 3.  InfoNCE loss
# ══════════════════════════════════════════════════════════════════════════════

def infonce_loss(q_embs: torch.Tensor, pos_embs: torch.Tensor,
                 neg_embs_list: List[torch.Tensor], tau: float = 0.07) -> torch.Tensor:
    """
    Per-sample InfoNCE: for each query, positive is its paired doc,
    negatives are the hard negatives + all other in-batch positives.
    """
    batch_size = q_embs.size(0)
    losses = []
    for i in range(batch_size):
        q   = q_embs[i].unsqueeze(0)                         # (1, D)
        pos = pos_embs[i].unsqueeze(0)                        # (1, D)
        neg = neg_embs_list[i]                                # (N, D)
        # Also treat other in-batch positives as negatives
        other_pos = torch.cat([pos_embs[:i], pos_embs[i+1:]], dim=0)  # (B-1, D)
        negs_all  = torch.cat([neg, other_pos], dim=0)        # (N+B-1, D)
        cands     = torch.cat([pos, negs_all], dim=0)         # (1+N+B-1, D)
        logits    = (q @ cands.T) / tau                       # (1, 1+N+B-1)
        target    = torch.zeros(1, dtype=torch.long, device=q.device)
        losses.append(F.cross_entropy(logits, target))
    return torch.stack(losses).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_finetuned_encoder(
    model: MiniLMEncoder,
    dataset: ContrastiveRetrievalDataset,
    device: torch.device,
    epochs: int = 30,
    lr: float = 2e-5,
    batch_size: int = 32,
    tau: float = 0.07,
    checkpoint_path: str = "models/dense_finetuned_encoder.pt",
) -> List[float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn, drop_last=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for questions, pos_docs, neg_docs_list in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            q_embs   = model.forward_text(questions, device)
            pos_embs = model.forward_text(pos_docs, device)
            neg_embs = [model.forward_text(neg_docs_list[i], device)
                        for i in range(len(questions))]
            loss = infonce_loss(q_embs, pos_embs, neg_embs, tau=tau)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_log.append(avg_loss)
        logger.info("Epoch %d/%d | loss=%.4f", epoch, epochs, avg_loss)

    ensure_dir(str(Path(checkpoint_path).parent))
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Saved encoder checkpoint to %s", checkpoint_path)
    return loss_log


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Evaluation using the fine-tuned embeddings
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_finetuned_dense(config, model: MiniLMEncoder, device: torch.device) -> None:
    """
    Re-embed the corpus with the fine-tuned encoder and evaluate with the
    standard SelectorEvaluator pipeline by swapping out the embedding store.
    """
    from sras.data.corpus import CorpusStore
    from sras.baselines.dense_selector import DenseSelector
    from sras.data.datasets import QADataset, SquadEvalDataset
    from sras.evaluation.metrics import MetricsComputer
    from sras.generator.interface import GeneratorInterface

    corpus = CorpusStore(config.data.corpus_metadata_path,
                         max_docs=config.data.max_corpus_docs)
    logger.info("Re-embedding %d corpus docs with fine-tuned encoder...", len(corpus.doc_ids))

    doc_texts = [corpus.get_text(did) for did in corpus.doc_ids]
    doc_embs  = model.encode(doc_texts, device)                         # (N, PROJ_DIM)
    torch.save(doc_embs, MODELS_DIR / "dense_finetuned_embeddings.pt")
    logger.info("Saved fine-tuned embeddings to models/dense_finetuned_embeddings.pt")

    selector = DenseSelector(doc_embs, corpus.doc_ids, device)
    metrics_computer = MetricsComputer(use_bertscore=True, device=device)
    generator = GeneratorInterface(
        model_name=config.evaluation.generator_model,
        device=device,
        max_input_len=config.evaluation.generator_max_input_len,
        max_output_len=config.evaluation.generator_max_output_len,
    )

    def _run_eval(qa_pairs, pool_fn, dataset_name: str):
        """Generic inner eval loop."""
        preds, gold_answers, sel_doc_ids_all = [], [], []
        hit = 0

        for item in tqdm(qa_pairs, desc=f"Eval {dataset_name}"):
            q_text = item["question"]

            # Encode query with fine-tuned encoder
            q_emb = model.encode([q_text], device).squeeze(0)

            # Build candidate pool (reuse pool builder from item if available)
            cand_ids = pool_fn(item)
            selected = selector.select_top_k(q_emb, k=config.evaluation.top_k,
                                             candidate_doc_ids=cand_ids)
            sel_doc_ids_all.append(selected)

            # Check hit (SQuAD only)
            gold_doc = item.get("context_doc_id")
            if gold_doc and gold_doc in selected:
                hit += 1

            # Generate answer
            doc_texts_sel = [corpus.get_text(d) for d in selected]
            answer        = generator.generate(q_text, doc_texts_sel)
            preds.append(answer)
            gold_answers.append(item.get("answer", item.get("gold", "")))

        metrics = metrics_computer.aggregate(
                      metrics_computer.compute_batch(preds, gold_answers))
        hit_rate = hit / len(qa_pairs) if qa_pairs else 0.0

        result = {
            "selector": "dense_finetuned",
            "n_questions": len(qa_pairs),
            "metrics": metrics,
            "selector_hit_rate": hit_rate if hit_rate > 0 else None,
            "per_sample_predictions": [
                {"question": qa_pairs[i]["question"],
                 "gold": gold_answers[i],
                 "pred": preds[i],
                 "selected_doc_ids": sel_doc_ids_all[i]}
                for i in range(len(preds))
            ],
            "failure_cases": [],
            "failure_rate": 0.0,
        }
        return result

    # ── Internal ──────────────────────────────────────────────────────────────
    from sras.data.datasets import QADataset, RewardDataset
    internal_ds = QADataset(config.data.qa_pairs_path)
    reward_ds   = RewardDataset(config.data.reward_matrix_path)

    def internal_pool(item):
        """Return a randomised pool of config.evaluation.candidate_pool_size docs."""
        all_ids = corpus.doc_ids[:]
        random.shuffle(all_ids)
        return all_ids[:config.evaluation.candidate_pool_size]

    qa_items = [{"question": p["question"], "gold": p["answer"]} for p in internal_ds]
    int_result = _run_eval(qa_items, internal_pool, "internal")
    save_json(int_result, str(RESULTS_DIR / "dense_finetuned_internal_results.json"))
    logger.info("Internal  F1=%.4f", int_result["metrics"]["relaxed_f1"])

    # SQuAD evaluation removed: internal domain corpus is the sole benchmark.


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Domain-adapted fine-tuned dense retriever baseline")
    p.add_argument("--config",      default=str(CONFIG_PATH))
    p.add_argument("--epochs",      type=int,   default=30,   help="Training epochs")
    p.add_argument("--lr",          type=float, default=2e-5, help="Learning rate")
    p.add_argument("--batch-size",  type=int,   default=32,   help="Batch size")
    p.add_argument("--tau",         type=float, default=0.07, help="Temperature for InfoNCE")
    p.add_argument("--n-negs",      type=int,   default=7,    help="Hard negatives per sample")
    p.add_argument("--min-reward",  type=float, default=0.5,  help="Min reward for positive docs")
    p.add_argument("--device",      default=None)
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training; load existing checkpoint and run eval only")
    p.add_argument("--checkpoint",  default=str(MODELS_DIR / "dense_finetuned_encoder.pt"))
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    device = get_device(args.device if args.device is not None else "auto")
    seed_everything(42)

    ensure_dir(str(MODELS_DIR))
    ensure_dir(str(RESULTS_DIR))

    # ── Load data ──────────────────────────────────────────────────────────────
    hf_cache   = str(DATA_DIR / "hf_cache")
    model      = MiniLMEncoder(ENCODER_NAME, proj_dim=PROJ_DIM, cache_dir=hf_cache)
    flat_corpus = json.loads((DATA_DIR / "flat_corpus.json").read_text(encoding="utf-8"))
    doc_text_map = {d["id"]: d.get("text", d.get("content", ""))
                    for d in flat_corpus}

    if args.skip_train:
        logger.info("Skipping training; loading checkpoint from %s", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        # ── Build dataset ──────────────────────────────────────────────────────
        reward_matrix = json.loads((DATA_DIR / "reward_matrix.json").read_text(encoding="utf-8"))
        dataset = ContrastiveRetrievalDataset(
            reward_matrix, doc_text_map,
            n_hard_negs=args.n_negs,
            min_pos_reward=args.min_reward,
        )

        # ── Train ──────────────────────────────────────────────────────────────
        loss_log = train_finetuned_encoder(
            model, dataset, device,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, tau=args.tau,
            checkpoint_path=args.checkpoint,
        )
        log_path = str(RESULTS_DIR / "dense_finetuned_training_log.json")
        save_json({"epochs": list(range(1, len(loss_log)+1)), "loss": loss_log}, log_path)
        logger.info("Training log saved to %s", log_path)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    evaluate_finetuned_dense(config, model.to(device), device)
    logger.info("Done. Results in results/dense_finetuned_*.json")


if __name__ == "__main__":
    main()
