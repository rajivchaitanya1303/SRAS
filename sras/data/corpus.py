from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from sras.utils.io import load_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CorpusStore:
    def __init__(self, corpus_path: str, max_docs: int = 0) -> None:
        raw = load_json(corpus_path)
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"Corpus at {corpus_path} must be a non-empty list of documents.")

        required_fields = {"id", "text"}
        for i, doc in enumerate(raw[:5]):
            missing = required_fields - set(doc.keys())
            if missing:
                raise ValueError(f"Corpus document {i} missing fields: {missing}")

        self._docs: List[Dict] = raw if max_docs == 0 else raw[:max_docs]
        self._id_to_doc: Dict[str, Dict] = {doc["id"]: doc for doc in self._docs}
        self._ids: List[str] = [doc["id"] for doc in self._docs]

        logger.info("Corpus loaded: %d documents", len(self._docs))

    @property
    def doc_ids(self) -> List[str]:
        return self._ids

    @property
    def size(self) -> int:
        return len(self._docs)

    def get_text(self, doc_id: str) -> str:
        if doc_id not in self._id_to_doc:
            raise KeyError(f"Document ID not found: {doc_id}")
        return self._id_to_doc[doc_id]["text"]

    def get_doc(self, doc_id: str) -> Dict:
        if doc_id not in self._id_to_doc:
            raise KeyError(f"Document ID not found: {doc_id}")
        return self._id_to_doc[doc_id]

    def get_texts_by_ids(self, doc_ids: List[str]) -> List[str]:
        return [self.get_text(d) for d in doc_ids]

    def sample_doc_ids(self, n: int, exclude: Optional[List[str]] = None, rng: Optional[random.Random] = None) -> List[str]:
        pool = [d for d in self._ids if exclude is None or d not in exclude]
        if n > len(pool):
            n = len(pool)
        if rng is not None:
            return rng.sample(pool, n)
        return random.sample(pool, n)

    def get_category_map(self) -> Dict[str, List[str]]:
        cat_map: Dict[str, List[str]] = {}
        for doc in self._docs:
            cat = doc.get("category", "unknown")
            cat_map.setdefault(cat, []).append(doc["id"])
        return cat_map

    def get_all_texts(self) -> List[str]:
        return [doc["text"] for doc in self._docs]

    def get_doc_index(self, doc_id: str) -> int:
        return self._ids.index(doc_id)
