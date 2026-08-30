from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DenseSelector:
    def __init__(
        self,
        doc_embeddings: torch.Tensor,
        doc_ids: List[str],
        device: torch.device,
    ) -> None:
        if doc_embeddings.shape[0] != len(doc_ids):
            raise ValueError("doc_embeddings rows must match len(doc_ids)")
        self._embeddings = doc_embeddings.float().to(device)
        self._doc_ids = doc_ids
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        self.device = device

    def score_candidates(
        self,
        q_emb: torch.Tensor,
        candidate_doc_ids: List[str],
    ) -> List[float]:
        indices = [self._id_to_idx[d] for d in candidate_doc_ids if d in self._id_to_idx]
        if not indices:
            return [0.0] * len(candidate_doc_ids)
        cand_embs = self._embeddings[indices]
        q = q_emb.float().to(self.device)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        scores = F.cosine_similarity(q.expand(cand_embs.shape[0], -1), cand_embs, dim=-1)
        return [float(s.item()) for s in scores]

    def select_top_k(
        self,
        q_emb: torch.Tensor,
        k: int,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if candidate_doc_ids is None:
            candidate_doc_ids = self._doc_ids
        scores = self.score_candidates(q_emb, candidate_doc_ids)
        ranked = sorted(zip(scores, candidate_doc_ids), key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in ranked[:k]]
