from __future__ import annotations

import re
import string
from typing import List, Optional, Tuple

import torch

from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _ARTICLE_RE.sub(" ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def relaxed_f1(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def _pairwise_cosine_mean(embs: torch.Tensor) -> float:
    """
    Compute the mean *off-diagonal* cosine similarity for a set of embeddings.
    Returns a float in [0, 1].  Higher ⟹ more redundant.
    """
    k = embs.shape[0]
    if k <= 1:
        return 1.0   # single doc: trivially "redundant with itself"
    normed = torch.nn.functional.normalize(embs.float(), dim=-1)
    sim = (normed @ normed.T)                        # [k, k]
    mask = 1.0 - torch.eye(k, device=embs.device)   # off-diagonal mask
    return float((sim * mask).sum() / (k * (k - 1)))


def _answer_coverage(answer: str, doc_texts: List[str], threshold: float = 0.1) -> float:
    """
    Returns 1.0 if any selected document shares ≥ threshold token overlap with
    the answer, else 0.0.  Provides a sparse coverage signal to the reward.
    """
    if not answer.strip():
        return 0.0
    a_toks = set(normalize_text(answer).split())
    if not a_toks:
        return 0.0
    for doc in doc_texts:
        d_toks = set(normalize_text(doc).split())
        if not d_toks:
            continue
        overlap = len(a_toks & d_toks) / len(a_toks)
        if overlap >= threshold:
            return 1.0
    return 0.0


class RewardComputer:
    """
    Computes the composite reward for the SRAS selector.

    Base reward (conference paper):
        r_base = α · relaxed_F1(pred, gold) + β · BERTScore(pred, gold)

    Extended reward (journal, DAR):
        r_dar = r_base
              + λ_d · diversity(selected_doc_embs)    # encourages non-redundant selection
              + λ_c · coverage(answer, selected_docs)  # sparse coverage bonus

    where:
        diversity  = 1 − mean_pairwise_cosine_sim(selected_doc_embs) ∈ [0, 1]
        coverage   ∈ {0, 1}: whether the answer appears in any selected doc

    The diversity bonus discourages the selector from picking near-duplicate
    documents, which typically hurts the generator by providing repetitive
    context.  The coverage bonus provides a direct sparse signal that the
    selector successfully retrieved at least one answer-bearing document.
    """

    def __init__(
        self,
        f1_weight: float = 0.6,
        bertscore_weight: float = 0.4,
        use_reward_shaping: bool = True,
        device: Optional[torch.device] = None,
        bertscore_model_type: str = "roberta-large",
        # DAR parameters
        diversity_weight: float = 0.0,
        coverage_weight: float = 0.0,
    ) -> None:
        if abs(f1_weight + bertscore_weight - 1.0) > 1e-6:
            raise ValueError("f1_weight + bertscore_weight must equal 1.0")
        self.f1_weight = f1_weight
        self.bertscore_weight = bertscore_weight
        self.use_reward_shaping = use_reward_shaping
        self.device = device or torch.device("cpu")
        self.bertscore_model_type = bertscore_model_type
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
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
            # public method; bert_score still calls it directly.  Patch it back
            # onto the tokenizer instance using the standard RoBERTa behaviour.
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
            logger.info("BERTScorer initialized (model=%s)", self.bertscore_model_type)
        return self._bertscore

    def compute(self, pred: str, gold: str) -> float:
        if not pred or not gold:
            return 0.0
        f1_score = relaxed_f1(pred, gold)
        if not self.use_reward_shaping:
            return f1_score
        try:
            scorer = self._get_bertscore()
            _, _, F1 = scorer.score([pred], [gold])
            bs_score = float(F1[0].item())
            bs_score = max(0.0, min(1.0, bs_score))
        except Exception as e:
            logger.warning("BERTScore failed, falling back to F1-only: %s", e)
            bs_score = f1_score
        return self.f1_weight * f1_score + self.bertscore_weight * bs_score

    def compute_with_diversity(
        self,
        pred: str,
        gold: str,
        selected_doc_embs: Optional[torch.Tensor] = None,
        selected_doc_texts: Optional[List[str]] = None,
    ) -> float:
        """
        Extended reward for DAR: base_reward + diversity_bonus + coverage_bonus.

        Parameters
        ----------
        pred               : model-generated answer string
        gold               : ground-truth answer string
        selected_doc_embs  : [k, emb_dim] tensor of selected doc embeddings
        selected_doc_texts : list of selected doc texts (for coverage check)
        """
        base = self.compute(pred, gold)

        diversity_bonus = 0.0
        if self.diversity_weight > 0 and selected_doc_embs is not None:
            mean_sim = _pairwise_cosine_mean(selected_doc_embs)
            diversity = 1.0 - mean_sim
            diversity_bonus = self.diversity_weight * diversity

        coverage_bonus = 0.0
        if self.coverage_weight > 0 and selected_doc_texts:
            cov = _answer_coverage(gold, selected_doc_texts)
            coverage_bonus = self.coverage_weight * cov

        return base + diversity_bonus + coverage_bonus

    def compute_batch_with_diversity(
        self,
        preds: List[str],
        golds: List[str],
        selected_doc_embs_list: Optional[List[torch.Tensor]] = None,
        selected_doc_texts_list: Optional[List[List[str]]] = None,
    ) -> List[float]:
        """
        Batch version of ``compute_with_diversity``.
        """
        base_rewards = self.compute_batch(preds, golds)

        results = []
        for i, (base, pred, gold) in enumerate(zip(base_rewards, preds, golds)):
            embs  = selected_doc_embs_list[i]  if selected_doc_embs_list  else None
            texts = selected_doc_texts_list[i] if selected_doc_texts_list else None

            diversity_bonus = 0.0
            if self.diversity_weight > 0 and embs is not None:
                diversity_bonus = self.diversity_weight * (1.0 - _pairwise_cosine_mean(embs))

            coverage_bonus = 0.0
            if self.coverage_weight > 0 and texts:
                coverage_bonus = self.coverage_weight * _answer_coverage(gold, texts)

            results.append(base + diversity_bonus + coverage_bonus)

        return results

    def compute_batch(self, preds: List[str], golds: List[str]) -> List[float]:
        if len(preds) != len(golds):
            raise ValueError("preds and golds must have equal length")
        if not preds:
            return []

        f1_scores = [relaxed_f1(p, g) for p, g in zip(preds, golds)]

        if not self.use_reward_shaping:
            return f1_scores

        try:
            scorer = self._get_bertscore()
            _, _, F1 = scorer.score(preds, golds)
            bs_scores = [max(0.0, min(1.0, float(v.item()))) for v in F1]
        except Exception as e:
            logger.warning("Batch BERTScore failed, using F1-only: %s", e)
            bs_scores = f1_scores

        return [
            self.f1_weight * f1 + self.bertscore_weight * bs
            for f1, bs in zip(f1_scores, bs_scores)
        ]
