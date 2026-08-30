from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from sras.baselines.bm25_selector import BM25Selector
from sras.baselines.dense_selector import DenseSelector
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _min_max_normalize(scores: List[float]) -> List[float]:
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-12:
        return [0.0] * len(scores)
    return [(s - lo) / span for s in scores]


class HybridSelector:
    def __init__(
        self,
        bm25: BM25Selector,
        dense: DenseSelector,
        bm25_weight: float = 0.5,
    ) -> None:
        if not (0.0 <= bm25_weight <= 1.0):
            raise ValueError("bm25_weight must be in [0, 1]")
        self._bm25 = bm25
        self._dense = dense
        self._bm25_weight = bm25_weight
        self._dense_weight = 1.0 - bm25_weight

    def score_candidates(
        self,
        query: str,
        q_emb: torch.Tensor,
        candidate_doc_ids: List[str],
    ) -> List[float]:
        if not candidate_doc_ids:
            return []
        bm25_raw = self._bm25.score_candidates(query, candidate_doc_ids)
        dense_raw = self._dense.score_candidates(q_emb, candidate_doc_ids)
        bm25_norm = _min_max_normalize(bm25_raw)
        dense_norm = _min_max_normalize(dense_raw)
        return [
            self._bm25_weight * b + self._dense_weight * d
            for b, d in zip(bm25_norm, dense_norm)
        ]

    def select_top_k(
        self,
        query: str,
        q_emb: torch.Tensor,
        k: int,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if candidate_doc_ids is None:
            raise ValueError("candidate_doc_ids must be provided for HybridSelector")
        scores = self.score_candidates(query, q_emb, candidate_doc_ids)
        ranked = sorted(zip(scores, candidate_doc_ids), key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in ranked[:k]]
