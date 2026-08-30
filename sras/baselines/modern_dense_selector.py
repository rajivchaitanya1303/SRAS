"""Modern compact dense-retriever baselines: BGE-small-en and E5-small.

The original submission excluded ColBERT/SPLADE as edge-infeasible (gigabyte-scale
indices) but did not compare against modern *compact* sentence-embedding retrievers
that are edge-feasible in the same way MiniLM and DPR are.  BGE-small-en-v1.5 and
E5-small-v2 are both ~130M-parameter, ~130 MB bi-encoders specifically trained for
retrieval (contrastive pre-training + supervised fine-tuning on MS MARCO-style
data), making them a fair "modern retriever" comparison point at a similar
deployment budget to the existing MiniLM/DPR baselines.

Both models require specific query/passage prefixing to reproduce their reported
retrieval quality (unprefixed encoding under-performs their published numbers):
  - BGE  (BAAI/bge-small-en-v1.5): query gets an instruction prefix, passages
    get no prefix.  https://huggingface.co/BAAI/bge-small-en-v1.5
  - E5   (intfloat/e5-small-v2): both query and passage get a role prefix
    ("query: " / "passage: ").  https://huggingface.co/intfloat/e5-small-v2

This mirrors the caching pattern already used by DPRSelector (sras/baselines/
dpr_selector.py): documents are embedded once and cached to disk; queries are
embedded on the fly with the model's own encoder (not the shared MiniLM
embeddings used elsewhere in the pipeline), because these models define their
own embedding space and prefixing convention.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Known prefixing conventions for the two models this module supports out of
# the box.  A caller can also pass an unrecognised model name; in that case no
# prefixes are applied and a warning is logged.
_PREFIX_PRESETS: Dict[str, Dict[str, str]] = {
    "BAAI/bge-small-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ",
        "passage": "",
    },
    "BAAI/bge-base-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ",
        "passage": "",
    },
    "intfloat/e5-small-v2": {
        "query": "query: ",
        "passage": "passage: ",
    },
    "intfloat/e5-base-v2": {
        "query": "query: ",
        "passage": "passage: ",
    },
}


class ModernDenseSelector:
    """Generic frozen sentence-embedding retriever with model-specific prefixing.

    Parameters
    ----------
    model_name : str
        HuggingFace / sentence-transformers model id, e.g.
        "BAAI/bge-small-en-v1.5" or "intfloat/e5-small-v2".
    doc_ids, doc_texts : parallel lists describing the corpus.
    device : torch.device
    cache_path : str
        Where to cache the precomputed passage embeddings (analogous to
        DPRSelector's cache_path). Recomputed if the corpus size on disk
        does not match len(doc_ids).
    label : str
        Human-readable name used only in log messages.
    """

    def __init__(
        self,
        model_name: str,
        doc_ids: List[str],
        doc_texts: List[str],
        device: torch.device,
        cache_path: str,
        batch_size: int = 64,
        label: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.label = label or model_name
        self._doc_ids = doc_ids
        self._id_to_idx: Dict[str, int] = {d: i for i, d in enumerate(doc_ids)}
        self.device = device
        self._ready = False
        self._model = None

        presets = _PREFIX_PRESETS.get(model_name)
        if presets is None:
            logger.warning(
                "No known prefix preset for %s; encoding without query/passage "
                "prefixes. Retrieval quality may be understated relative to the "
                "model's published numbers.",
                model_name,
            )
        self._query_prefix = presets["query"] if presets else ""
        self._passage_prefix = presets["passage"] if presets else ""

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading modern dense baseline: %s (%s)", self.label, model_name)
            self._model = SentenceTransformer(model_name, device=str(device))

            cached = None
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, map_location=device, weights_only=False)
                if isinstance(cached, dict):
                    cached = cached.get("embeddings", list(cached.values())[0])
                if cached.shape[0] != len(doc_ids):
                    logger.info(
                        "%s cache size (%d) != corpus size (%d); recomputing.",
                        self.label, cached.shape[0], len(doc_ids),
                    )
                    cached = None

            if cached is None:
                logger.info(
                    "Encoding %d documents with %s (this may take a few minutes)...",
                    len(doc_texts), self.label,
                )
                prefixed_texts = [self._passage_prefix + t for t in doc_texts]
                all_embs: List[torch.Tensor] = []
                for start in range(0, len(prefixed_texts), batch_size):
                    batch = prefixed_texts[start: start + batch_size]
                    embs = self._model.encode(
                        batch, convert_to_tensor=True, show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    all_embs.append(embs.cpu())
                cached = torch.cat(all_embs, dim=0)
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                torch.save(cached, cache_path)
                logger.info("%s document embeddings cached to %s", self.label, cache_path)

            self._doc_embs = cached.float().to(device)
            self._ready = True
            logger.info(
                "%s ready, doc embeddings shape: %s", self.label, tuple(self._doc_embs.shape)
            )

        except Exception as e:
            logger.warning(
                "%s initialisation failed (%s). This baseline will not be available.",
                self.label, e,
            )

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _encode_query(self, question: str) -> torch.Tensor:
        text = self._query_prefix + question
        emb = self._model.encode(
            text, convert_to_tensor=True, normalize_embeddings=True,
        )
        return emb.float().to(self.device)

    def score_candidates(self, question: str, candidate_doc_ids: List[str]) -> List[float]:
        if not self._ready:
            return [0.0] * len(candidate_doc_ids)
        indices = [self._id_to_idx[d] for d in candidate_doc_ids if d in self._id_to_idx]
        if not indices:
            return [0.0] * len(candidate_doc_ids)
        cand_embs = self._doc_embs[indices]
        q_emb = self._encode_query(question)
        scores = F.cosine_similarity(
            q_emb.unsqueeze(0).expand(cand_embs.shape[0], -1), cand_embs, dim=-1
        )
        return [float(s.item()) for s in scores]

    def select_top_k(
        self,
        question: str,
        k: int,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not self._ready:
            return (candidate_doc_ids or self._doc_ids)[:k]
        if candidate_doc_ids is None:
            candidate_doc_ids = self._doc_ids
        scores = self.score_candidates(question, candidate_doc_ids)
        ranked = sorted(zip(scores, candidate_doc_ids), key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in ranked[:k]]
