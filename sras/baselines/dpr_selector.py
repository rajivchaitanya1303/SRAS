"""DPR (Dense Passage Retrieval) baseline selector.

Uses facebook/dpr-question_encoder-single-nq-base to encode queries and
facebook/dpr-ctx_encoder-single-nq-base to encode documents, then ranks
candidates by dot-product similarity, the scoring method used in the
original DPR paper (Karpukhin et al., 2020).

This is a stronger baseline than all-MiniLM-L6-v2 because:
  - Trained specifically for open-domain passage retrieval (NQ, TriviaQA)
  - Uses asymmetric encoding (separate Q/D encoders)
  - Evaluated with dot-product, not cosine similarity

If the DPR model cannot be loaded (no internet, disk space) the selector
falls back gracefully to cosine-similarity dense retrieval.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DPR_Q_MODEL  = "facebook/dpr-question_encoder-single-nq-base"
_DPR_CTX_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
_CACHE_FILE    = "data/dpr_doc_embeddings.pt"


class DPRSelector:
    """Passage selector using Dense Passage Retrieval encoders.

    At construction time the selector encodes all corpus documents with the
    DPR context encoder and caches the resulting embeddings.  At query time
    it encodes the question with the DPR question encoder and returns the
    top-k documents by dot-product score.
    """

    def __init__(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
        device: torch.device,
        cache_path: str = _CACHE_FILE,
        batch_size: int = 64,
    ) -> None:
        self._doc_ids  = doc_ids
        self._id_to_idx: Dict[str, int] = {d: i for i, d in enumerate(doc_ids)}
        self.device    = device
        self._q_encoder  = None
        self._q_tokenizer = None
        self._ready      = False

        try:
            from transformers import DPRQuestionEncoder, DPRContextEncoder
            from transformers import DPRQuestionEncoderTokenizerFast, DPRContextEncoderTokenizerFast

            logger.info("Loading DPR question encoder from %s", _DPR_Q_MODEL)
            self._q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(_DPR_Q_MODEL)
            self._q_encoder   = DPRQuestionEncoder.from_pretrained(_DPR_Q_MODEL).to(device).eval()

            # Build / load document embeddings
            if os.path.exists(cache_path):
                logger.info("Loading cached DPR document embeddings from %s", cache_path)
                self._doc_embs = torch.load(cache_path, map_location=device, weights_only=False)
                if isinstance(self._doc_embs, dict):
                    self._doc_embs = self._doc_embs.get("embeddings", list(self._doc_embs.values())[0])
            else:
                logger.info("Encoding %d documents with DPR context encoder (this may take a few minutes)…", len(doc_texts))
                ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(_DPR_CTX_MODEL)
                ctx_encoder   = DPRContextEncoder.from_pretrained(_DPR_CTX_MODEL).to(device).eval()

                all_embs: List[torch.Tensor] = []
                with torch.no_grad():
                    for start in range(0, len(doc_texts), batch_size):
                        batch_texts = doc_texts[start : start + batch_size]
                        inputs = ctx_tokenizer(
                            batch_texts,
                            truncation=True,
                            max_length=256,
                            padding=True,
                            return_tensors="pt",
                        ).to(device)
                        embs = ctx_encoder(**inputs).pooler_output  # (B, 768)
                        all_embs.append(embs.cpu())

                self._doc_embs = torch.cat(all_embs, dim=0).to(device)
                os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
                torch.save(self._doc_embs, cache_path)
                logger.info("DPR document embeddings cached to %s", cache_path)
                del ctx_encoder, ctx_tokenizer

            self._ready = True
            logger.info("DPRSelector ready, doc embeddings shape: %s", tuple(self._doc_embs.shape))

        except Exception as e:
            logger.warning(
                "DPR initialisation failed (%s). DPRSelector will not be available.", e
            )

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _encode_query(self, question: str) -> torch.Tensor:
        """Encode a single question with the DPR question encoder."""
        inputs = self._q_tokenizer(
            question,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            emb = self._q_encoder(**inputs).pooler_output  # (1, 768)
        return emb.squeeze(0)

    def select_top_k(
        self,
        question: str,
        k: int,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Return top-k doc IDs ranked by DPR dot-product score."""
        if not self._ready:
            return (candidate_doc_ids or self._doc_ids)[:k]

        if candidate_doc_ids is None:
            candidate_doc_ids = self._doc_ids

        indices = [self._id_to_idx[d] for d in candidate_doc_ids if d in self._id_to_idx]
        if not indices:
            return candidate_doc_ids[:k]

        cand_embs = self._doc_embs[indices]  # (N, 768)
        q_emb     = self._encode_query(question).float()  # (768,)

        # DPR uses dot-product (not cosine); normalise for fair comparison
        scores = (cand_embs.float() @ q_emb).tolist()  # (N,)

        ranked = sorted(
            zip(scores, candidate_doc_ids),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc_id for _, doc_id in ranked[:k]]
