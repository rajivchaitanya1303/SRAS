from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sras.rl.rewards import relaxed_f1, exact_match
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MetricsComputer:
    def __init__(
        self,
        use_bertscore: bool = True,
        device: Optional[torch.device] = None,
        bertscore_model_type: str = "roberta-large",
    ) -> None:
        self.use_bertscore = use_bertscore
        self.device = device or torch.device("cpu")
        self.bertscore_model_type = bertscore_model_type
        self._bertscore = None

    def _get_bertscore(self):
        if self._bertscore is None:
            from bert_score import BERTScorer
            self._bertscore = BERTScorer(
                model_type=self.bertscore_model_type,
                lang="en",
                device=self.device,
                rescale_with_baseline=False,
                use_fast_tokenizer=False,
            )
            # Newer transformers removed build_inputs_with_special_tokens as a
            # public method; bert_score still calls it directly.  Patch it back.
            tok = getattr(self._bertscore, "_tokenizer", None)
            if tok is not None and not hasattr(tok, "build_inputs_with_special_tokens"):
                _bos = tok.bos_token_id
                _eos = tok.eos_token_id
                def _build_inputs(token_ids_0, token_ids_1=None):
                    if token_ids_1 is None:
                        return [_bos] + token_ids_0 + [_eos]
                    return [_bos] + token_ids_0 + [_eos, _eos] + token_ids_1 + [_eos]
                tok.build_inputs_with_special_tokens = _build_inputs
                logger.info("Patched tokenizer: restored build_inputs_with_special_tokens")
        return self._bertscore

    def compute_single(self, pred: str, gold: str) -> Dict[str, float]:
        result: Dict[str, float] = {
            "relaxed_f1": relaxed_f1(pred, gold),
            "exact_match": exact_match(pred, gold),
        }
        if self.use_bertscore:
            try:
                scorer = self._get_bertscore()
                _, _, F1 = scorer.score([pred], [gold])
                result["bertscore_f1"] = max(0.0, min(1.0, float(F1[0].item())))
            except Exception as e:
                logger.warning("BERTScore single failed: %s", e)
                result["bertscore_f1"] = result["relaxed_f1"]
        return result

    def compute_batch(
        self,
        preds: List[str],
        golds: List[str],
    ) -> Dict[str, List[float]]:
        if len(preds) != len(golds):
            raise ValueError("preds and golds must have equal length")
        if not preds:
            return {"relaxed_f1": [], "exact_match": [], "bertscore_f1": []}

        f1_scores = [relaxed_f1(p, g) for p, g in zip(preds, golds)]
        em_scores = [exact_match(p, g) for p, g in zip(preds, golds)]

        result: Dict[str, List[float]] = {
            "relaxed_f1": f1_scores,
            "exact_match": em_scores,
        }

        if self.use_bertscore:
            try:
                scorer = self._get_bertscore()
                _, _, F1 = scorer.score(preds, golds)
                result["bertscore_f1"] = [max(0.0, min(1.0, float(v.item()))) for v in F1]
            except Exception as e:
                logger.warning("BERTScore batch failed: %s. Using relaxed_f1 as fallback.", e)
                result["bertscore_f1"] = f1_scores

        return result

    def aggregate(self, per_sample: Dict[str, List[float]]) -> Dict[str, float]:
        return {
            metric: float(np.mean(values)) if values else 0.0
            for metric, values in per_sample.items()
        }

    def compute_selection_precision(
        self,
        selected_ids: List[List[str]],
        best_ids: List[str],
    ) -> float:
        if not selected_ids:
            return 0.0
        hits = sum(
            1 for sel, best in zip(selected_ids, best_ids)
            if best in sel
        )
        return hits / len(selected_ids)
