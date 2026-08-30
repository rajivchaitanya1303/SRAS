"""
Contrastive Selector Pre-training (CSP)
========================================
Novel journal contribution #1.

Motivation
----------
The standard SRAS pipeline initialises the CrossAttentionSelector with random
weights, then runs supervised warm-up.  This means the query/document projection
matrices learn purely from cross-entropy ranking signals, which is slow to
converge and sensitive to the limited positive-label supervision available in the
reward matrix.

CSP adds a *contrastive pre-training* stage **before** supervised warm-up.
Using an InfoNCE objective, it directly optimises the projection matrices to
embed queries close to their positive documents and far from negative ones.  The
learnt metric space gives the subsequent supervised and PPO stages a far better
initialisation, analogous to how SimCLR / MoCo pre-training benefits downstream
fine-tuning in computer vision.

Architecture
------------
Only the shared projections (q_proj / d_proj / layer-norm heads) of the existing
CrossAttentionSelector are trained; the attn scoring head is kept random.  This
ensures CSP is *drop-in* compatible: the same checkpoint can be loaded directly
by the existing supervised trainer and PPO trainer.

Loss
----
For each query we sample:
  * 1 positive document  (highest reward in reward matrix for that query)
  * N negative documents (mixture of: random corpus docs, and hard-negatives,
    docs that dense search ranks highly but that are not the true positive)

Loss = InfoNCE with temperature τ:

    L = -log [ exp(s(q,d+)/τ) / ( exp(s(q,d+)/τ) + Σᵢ exp(s(q,dᵢ⁻)/τ) ) ]

where s(q, d) = cosine_similarity(q_proj(q_emb), d_proj(d_emb)).
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sras.config.schema import SRASConfig
from sras.data.corpus import CorpusStore
from sras.data.datasets import RewardDataset
from sras.data.embeddings import EmbeddingStore, encode_query
from sras.models.selector import CrossAttentionSelector, save_selector
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger
from sras.utils.reproducibility import get_device, seed_everything

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# InfoNCE loss helper
# ─────────────────────────────────────────────────────────────────────────────

def _infonce_loss(
    q_proj: nn.Module,
    d_proj: nn.Module,
    q_norm: Optional[nn.Module],
    d_norm: Optional[nn.Module],
    q_emb: torch.Tensor,         # [emb_dim]
    pos_emb: torch.Tensor,       # [emb_dim]
    neg_embs: torch.Tensor,      # [N, emb_dim]
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute InfoNCE loss for one (query, positive, negatives) triplet.

    Returns a scalar loss tensor with gradients.
    """
    # ── Project and L2-normalise ────────────────────────────────────────────
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)        # [1, emb_dim]

    q_h = q_proj(q_emb)                   # [1, hidden]
    if q_norm is not None:
        q_h = q_norm(q_h)
    q_h = F.normalize(q_h, dim=-1)        # [1, hidden]

    pos_h = d_proj(pos_emb.unsqueeze(0))  # [1, hidden]
    if d_norm is not None:
        pos_h = d_norm(pos_h)
    pos_h = F.normalize(pos_h, dim=-1)    # [1, hidden]

    neg_h = d_proj(neg_embs)              # [N, hidden]
    if d_norm is not None:
        neg_h = d_norm(neg_h)
    neg_h = F.normalize(neg_h, dim=-1)    # [N, hidden]

    # ── Similarity scores ──────────────────────────────────────────────────
    pos_sim = (q_h * pos_h).sum(dim=-1) / temperature   # [1]
    neg_sims = (q_h * neg_h).sum(dim=-1) / temperature  # [N]

    # ── InfoNCE via cross-entropy with label 0 ─────────────────────────────
    # logits = [pos_sim, neg_sim_0, ..., neg_sim_N-1]
    logits = torch.cat([pos_sim, neg_sims], dim=0).unsqueeze(0)  # [1, N+1]
    target = torch.zeros(1, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


# ─────────────────────────────────────────────────────────────────────────────
# Hard-negative mining helper
# ─────────────────────────────────────────────────────────────────────────────

def _mine_hard_negatives(
    q_emb: torch.Tensor,       # [emb_dim]
    all_embs: torch.Tensor,    # [D, emb_dim]
    pos_idx: int,
    n: int,
    top_hard: int = 30,
) -> List[int]:
    """
    Return indices of hard negatives: docs that are most similar to the query
    but are NOT the positive document.  Combined with random negatives for
    diversity.
    """
    with torch.no_grad():
        q_n = F.normalize(q_emb.float().unsqueeze(0), dim=-1)  # [1, D]
        a_n = F.normalize(all_embs.float(), dim=-1)             # [D, emb]
        sims = (q_n @ a_n.T).squeeze(0)                        # [D]
        sims[pos_idx] = -1e9                                    # mask positive
        top_k = min(top_hard, sims.shape[0] - 1)
        hard_idxs = torch.topk(sims, top_k).indices.tolist()

    n_hard = min(n // 2, len(hard_idxs))
    n_rand = n - n_hard
    selected = random.sample(hard_idxs, n_hard)

    all_idxs = list(range(sims.shape[0]))
    remaining = [i for i in all_idxs if i != pos_idx and i not in set(hard_idxs)]
    if remaining:
        selected += random.sample(remaining, min(n_rand, len(remaining)))
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer class
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveTrainer:
    """
    Trains the CrossAttentionSelector projections via InfoNCE before supervised
    warm-up.  The trained checkpoint is saved to
    ``config.training.contrastive_checkpoint`` and is automatically picked up
    by :class:`~sras.training.supervised.SupervisedTrainer` (when
    ``use_contrastive_warmup=True``) to initialise supervised training.
    """

    def __init__(self, config: SRASConfig) -> None:
        self.config = config
        self.tcfg   = config.training
        self.dcfg   = config.data
        self.mcfg   = config.model

        self.device = get_device(self.tcfg.device)
        seed_everything(self.tcfg.seed)

        ensure_dir(self.tcfg.checkpoint_dir)
        ensure_dir(self.tcfg.log_dir)

    # ── Build a fresh selector ────────────────────────────────────────────────
    def _build_model(self) -> CrossAttentionSelector:
        return CrossAttentionSelector(
            doc_emb_dim  = self.mcfg.doc_emb_dim,
            hidden_dim   = self.mcfg.hidden_dim,
            dropout      = self.mcfg.dropout,
            use_layer_norm = self.mcfg.use_layer_norm,
            use_residual   = self.mcfg.use_residual,
        ).to(self.device)

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self) -> CrossAttentionSelector:
        logger.info("=== Contrastive Selector Pre-training (CSP) ===")

        reward_dataset = RewardDataset(self.dcfg.reward_matrix_path)
        corpus = CorpusStore(
            self.dcfg.corpus_metadata_path,
            max_docs=self.dcfg.max_corpus_docs,
        )
        embed_store = EmbeddingStore(
            self.dcfg.doc_embeddings_path,
            corpus.doc_ids,
            self.device,
        )

        model     = self._build_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.tcfg.contrastive_lr,
            weight_decay=1e-4,
        )

        # Pre-fetch all document embeddings for hard-negative mining
        all_doc_embs = embed_store.get_batch_by_indices(
            list(range(embed_store.size()))
        )  # [D, emb_dim]

        q_norm = model.q_norm if model.use_layer_norm else None
        d_norm = model.d_norm if model.use_layer_norm else None

        log_history: List[Dict] = []
        best_loss   = float("inf")

        for epoch in range(1, self.tcfg.contrastive_epochs + 1):
            model.train()
            total_loss   = 0.0
            n_samples    = 0
            items        = list(reward_dataset.items())
            random.shuffle(items)

            for question, candidates in tqdm(
                items,
                desc=f"CSP epoch {epoch}/{self.tcfg.contrastive_epochs}",
                leave=False,
            ):
                if len(candidates) < 2:
                    continue

                # ── Identify positive doc (highest reward) ────────────────────
                best = max(candidates, key=lambda c: c["reward"])
                if best["reward"] <= 0.0:
                    continue

                pos_doc_id = best["candidate_doc_id"]
                pos_idx    = corpus.doc_ids.index(pos_doc_id) \
                    if pos_doc_id in corpus.doc_ids else -1
                if pos_idx < 0:
                    continue

                try:
                    q_emb   = encode_query(
                        question, self.dcfg.embedding_model, self.device
                    )
                    pos_emb = embed_store.get_batch_by_indices([pos_idx])
                    if pos_emb.dim() == 2:
                        pos_emb = pos_emb.squeeze(0)

                    # ── Sample negatives (hard + random) ────────────────────
                    neg_idxs = _mine_hard_negatives(
                        q_emb.detach(),
                        all_doc_embs,
                        pos_idx,
                        n=self.tcfg.contrastive_n_negatives,
                    )
                    if not neg_idxs:
                        continue

                    neg_embs = embed_store.get_batch_by_indices(neg_idxs)

                    # ── InfoNCE loss ─────────────────────────────────────────
                    loss = _infonce_loss(
                        q_proj      = model.q_proj,
                        d_proj      = model.d_proj,
                        q_norm      = q_norm,
                        d_norm      = d_norm,
                        q_emb       = q_emb,
                        pos_emb     = pos_emb,
                        neg_embs    = neg_embs,
                        temperature = self.tcfg.contrastive_temperature,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    n_samples  += 1

                except Exception as e:
                    logger.debug("CSP: skipping sample (%s)", e)
                    continue

            if n_samples == 0:
                logger.warning("CSP epoch %d: no valid samples", epoch)
                continue

            avg_loss = total_loss / n_samples
            log_history.append({
                "epoch": epoch,
                "avg_loss": avg_loss,
                "n_samples": n_samples,
            })
            logger.info(
                "CSP epoch %d/%d | loss=%.4f | n=%d",
                epoch, self.tcfg.contrastive_epochs, avg_loss, n_samples,
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_selector(
                    model,
                    self.tcfg.contrastive_checkpoint,
                    {"epoch": epoch, "avg_contrastive_loss": avg_loss},
                )
                logger.info("CSP best model saved (loss=%.4f)", best_loss)

        log_path = os.path.join(
            self.tcfg.log_dir, "contrastive_training_log.json"
        )
        save_json(log_history, log_path)
        logger.info("CSP complete. Log saved to %s", log_path)
        return model
