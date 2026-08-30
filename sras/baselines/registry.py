from __future__ import annotations

from typing import Dict, List, Optional

import torch

from sras.baselines.bm25_selector import BM25Selector
from sras.baselines.dense_selector import DenseSelector
from sras.baselines.dpr_selector import DPRSelector
from sras.baselines.hybrid_selector import HybridSelector
from sras.baselines.learned_ranker import LearnedRanker
from sras.baselines.modern_dense_selector import ModernDenseSelector
from sras.config.schema import BaselineConfig
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaselineRegistry:
    def __init__(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
        doc_embeddings: torch.Tensor,
        config: BaselineConfig,
        device: torch.device,
    ) -> None:
        self._doc_ids = doc_ids
        self._doc_texts = doc_texts
        self._config = config
        self.device = device

        self._bm25: Optional[BM25Selector] = None
        self._dense: Optional[DenseSelector] = None
        self._hybrid: Optional[HybridSelector] = None
        self._dpr: Optional[DPRSelector] = None
        self._ranker: Optional[LearnedRanker] = None
        self._bge: Optional[ModernDenseSelector] = None
        self._e5: Optional[ModernDenseSelector] = None

        if config.use_bm25 or config.use_hybrid:
            self._build_bm25(doc_ids, doc_texts)

        if config.use_dense or config.use_hybrid:
            self._build_dense(doc_ids, doc_embeddings, device)

        if config.use_hybrid and self._bm25 is not None and self._dense is not None:
            self._hybrid = HybridSelector(self._bm25, self._dense, config.hybrid_bm25_weight)
            logger.info("HybridSelector initialized (bm25_weight=%.2f)", config.hybrid_bm25_weight)

        if config.use_dpr:
            self._dpr = DPRSelector(doc_ids, doc_texts, device, cache_path=config.dpr_cache_path)

        if config.use_bge_small:
            self._bge = ModernDenseSelector(
                config.bge_small_model, doc_ids, doc_texts, device,
                cache_path=config.bge_small_cache_path, label="BGE-small-en-v1.5",
            )

        if config.use_e5_small:
            self._e5 = ModernDenseSelector(
                config.e5_small_model, doc_ids, doc_texts, device,
                cache_path=config.e5_small_cache_path, label="E5-small-v2",
            )

        if config.use_learned_ranker:
            self._ranker = LearnedRanker(device=device)
            if config.learned_ranker_checkpoint:
                try:
                    self._ranker.load(config.learned_ranker_checkpoint)
                    logger.info("LearnedRanker loaded from %s", config.learned_ranker_checkpoint)
                except FileNotFoundError:
                    logger.warning("Learned ranker checkpoint not found: %s", config.learned_ranker_checkpoint)

    def _build_bm25(self, doc_ids: List[str], doc_texts: List[str]) -> None:
        try:
            self._bm25 = BM25Selector(doc_texts, doc_ids)
            logger.info("BM25Selector ready")
        except ImportError as e:
            logger.warning("BM25 unavailable: %s", e)

    def _build_dense(self, doc_ids: List[str], doc_embeddings: torch.Tensor, device: torch.device) -> None:
        self._dense = DenseSelector(doc_embeddings, doc_ids, device)
        logger.info("DenseSelector ready")

    def available(self) -> List[str]:
        # "oracle" and "random" are handled directly by SelectorEvaluator, not
        # through BaselineRegistry, because they require reward-matrix data that
        # isn't available here.  Listing "oracle" here caused it to be evaluated
        # with baseline_selector=None and no oracle_rewards, making it identical
        # to random selection.
        names = ["random"]
        if self._bm25 is not None:
            names.append("bm25")
        if self._dense is not None:
            names.append("dense")
        if self._hybrid is not None:
            names.append("hybrid")
        if self._dpr is not None and self._dpr.is_ready:
            names.append("dpr")
        if self._bge is not None and self._bge.is_ready:
            names.append("bge_small")
        if self._e5 is not None and self._e5.is_ready:
            names.append("e5_small")
        if self._ranker is not None and self._ranker._trained:
            names.append("learned_ranker")
        return names

    def get(self, name: str):
        return {
            "bm25": self._bm25,
            "dense": self._dense,
            "hybrid": self._hybrid,
            "dpr": self._dpr,
            "bge_small": self._bge,
            "e5_small": self._e5,
            "learned_ranker": self._ranker,
        }.get(name)

    def get_bm25(self) -> Optional[BM25Selector]:
        return self._bm25

    def get_dense(self) -> Optional[DenseSelector]:
        return self._dense

    def get_hybrid(self) -> Optional[HybridSelector]:
        return self._hybrid

    def get_ranker(self) -> Optional[LearnedRanker]:
        return self._ranker

    def train_ranker(
        self,
        q_embs: List[torch.Tensor],
        d_embs_list: List[torch.Tensor],
        rewards_list: List[List[float]],
    ) -> None:
        if self._ranker is None:
            raise RuntimeError("LearnedRanker is not enabled in config")
        self._ranker.train(
            q_embs, d_embs_list, rewards_list,
            epochs=self._config.learned_ranker_epochs,
            lr=self._config.learned_ranker_lr,
        )
