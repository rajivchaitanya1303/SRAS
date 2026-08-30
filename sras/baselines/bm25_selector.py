from __future__ import annotations

import re
import string
from typing import List, Optional

from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    return [t for t in text.split() if t]


class BM25Selector:
    def __init__(self, doc_texts: List[str], doc_ids: List[str]) -> None:
        if len(doc_texts) != len(doc_ids):
            raise ValueError("doc_texts and doc_ids must have equal length")
        if not doc_texts:
            raise ValueError("doc_texts must not be empty")

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank-bm25 is required: pip install rank-bm25") from e

        self._doc_ids = doc_ids
        self._tokenized = [_tokenize(t) for t in doc_texts]
        self._index = BM25Okapi(self._tokenized)
        logger.info("BM25Selector indexed %d documents", len(doc_texts))

    def score(self, query: str, doc_indices: Optional[List[int]] = None) -> List[float]:
        tokens = _tokenize(query)
        if not tokens:
            n = len(doc_indices) if doc_indices is not None else len(self._doc_ids)
            return [0.0] * n
        all_scores = self._index.get_scores(tokens)
        if doc_indices is not None:
            return [float(all_scores[i]) for i in doc_indices]
        return [float(s) for s in all_scores]

    def select_top_k(self, query: str, k: int, candidate_doc_ids: Optional[List[str]] = None) -> List[str]:
        if candidate_doc_ids is not None:
            id_to_global = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}
            indices = [id_to_global[d] for d in candidate_doc_ids if d in id_to_global]
            scores = self.score(query, indices)
            ranked = sorted(zip(scores, candidate_doc_ids), key=lambda x: x[0], reverse=True)
            return [doc_id for _, doc_id in ranked[:k]]
        tokens = _tokenize(query)
        if not tokens:
            return self._doc_ids[:k]
        top_n = self._index.get_top_n(tokens, self._doc_ids, n=k)
        return list(top_n)

    def score_candidates(self, query: str, candidate_doc_ids: List[str]) -> List[float]:
        id_to_global = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}
        indices = [id_to_global.get(d, 0) for d in candidate_doc_ids]
        return self.score(query, indices)
