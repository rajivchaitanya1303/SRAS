from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingStore:
    def __init__(
        self,
        embeddings_path: str,
        doc_ids: List[str],
        device: torch.device,
    ) -> None:
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

        raw = torch.load(embeddings_path, map_location=device, weights_only=False)
        # Support both plain tensors and dict format {"doc_ids": [...], "embeddings": tensor}
        if isinstance(raw, dict) and "embeddings" in raw:
            raw = raw["embeddings"]
        if isinstance(raw, np.ndarray):
            raw = torch.from_numpy(raw)
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor embeddings, got {type(raw)}")

        raw = raw.float()

        if raw.shape[0] != len(doc_ids):
            raise ValueError(
                f"Embeddings count {raw.shape[0]} does not match doc_ids count {len(doc_ids)}"
            )

        self._embeddings: torch.Tensor = raw.to(device).clone()
        self._doc_ids: List[str] = doc_ids
        self._id_to_idx: dict = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        self.device = device
        self.embedding_dim: int = raw.shape[1]

        logger.info(
            "EmbeddingStore loaded: %d vectors of dim %d on %s",
            raw.shape[0],
            raw.shape[1],
            device,
        )

    @property
    def all_embeddings(self) -> torch.Tensor:
        return self._embeddings

    def get_by_idx(self, idx: int) -> torch.Tensor:
        return self._embeddings[idx]

    def get_by_id(self, doc_id: str) -> torch.Tensor:
        if doc_id not in self._id_to_idx:
            raise KeyError(f"doc_id not found: {doc_id}")
        return self._embeddings[self._id_to_idx[doc_id]]

    def get_batch_by_ids(self, doc_ids: List[str]) -> torch.Tensor:
        indices = []
        for doc_id in doc_ids:
            if doc_id not in self._id_to_idx:
                raise KeyError(f"doc_id not found: {doc_id}")
            indices.append(self._id_to_idx[doc_id])
        return self._embeddings[indices]

    def get_batch_by_indices(self, indices: List[int]) -> torch.Tensor:
        return self._embeddings[indices]

    def size(self) -> int:
        return self._embeddings.shape[0]


def build_embeddings(
    texts: List[str],
    doc_ids: List[str],
    model_name: str,
    device: torch.device,
    output_path: str,
    batch_size: int = 64,
) -> EmbeddingStore:
    from sentence_transformers import SentenceTransformer
    import os

    logger.info("Building embeddings with %s for %d documents...", model_name, len(texts))
    model = SentenceTransformer(model_name, device=str(device))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    embeddings = embeddings.float().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(embeddings, output_path)
    logger.info("Saved embeddings to %s", output_path)
    return EmbeddingStore(output_path, doc_ids, device)


def encode_query(
    question: str,
    model_name: str,
    device: torch.device,
    _model_cache: dict = {},
) -> torch.Tensor:
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name, device=str(device))
    model = _model_cache[model_name]
    emb = model.encode(question, convert_to_tensor=True, normalize_embeddings=False)
    return emb.float().to(device).clone()


def encode_queries_batch(
    questions: List[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 64,
    _model_cache: dict = {},
) -> torch.Tensor:
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name, device=str(device))
    model = _model_cache[model_name]
    embs = model.encode(
        questions,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return embs.float().to(device)
