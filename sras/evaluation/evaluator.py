from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from sras.config.schema import EvaluationConfig, SRASConfig
from sras.data.corpus import CorpusStore
from sras.data.datasets import QADataset, RewardDataset, SquadDataset, SquadEvalDataset
from sras.data.embeddings import EmbeddingStore, encode_queries_batch
from sras.evaluation.failure_analysis import _classify_question
from sras.evaluation.metrics import MetricsComputer
from sras.generator.interface import GeneratorInterface
from sras.models.selector import CrossAttentionSelector, load_selector
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger
from sras.utils.reproducibility import get_device, seed_everything

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Gold-answer normalisation
# ---------------------------------------------------------------------------

def _short_gold(text: str, max_tokens: int = 20) -> str:
    """Return a concise gold answer for F1 evaluation.

    Strategy (in order):
      1. Take the first sentence (split on '.', '?', '!').
      2. If that is still longer than max_tokens words, keep only the first
         max_tokens words.

    This avoids penalising correct short predictions against paragraph-length
    gold answers.  The full gold text is still stored in per_sample_predictions
    for reference.
    """
    if not text:
        return text
    # First sentence
    import re
    sentence_end = re.search(r"[.?!]", text)
    if sentence_end:
        short = text[: sentence_end.start() + 1].strip()
    else:
        short = text.strip()
    # Token cap
    tokens = short.split()
    if len(tokens) > max_tokens:
        short = " ".join(tokens[:max_tokens])
    return short if short else text


class SelectorEvaluator:
    def __init__(self, config: SRASConfig) -> None:
        self.config = config
        self.ecfg = config.evaluation
        self.dcfg = config.data
        self.mcfg = config.model

        self.device = get_device(self.ecfg.device)
        seed_everything(self.ecfg.seed)

        self.corpus = CorpusStore(self.dcfg.corpus_metadata_path, max_docs=self.dcfg.max_corpus_docs)
        self.embed_store = EmbeddingStore(
            self.dcfg.doc_embeddings_path, self.corpus.doc_ids, self.device
        )
        self.metrics = MetricsComputer(
            use_bertscore="bertscore_f1" in self.ecfg.metrics,
            device=self.device,
        )
        self.generator = GeneratorInterface(
            model_name=self.ecfg.generator_model,
            device=self.device,
            max_input_len=self.ecfg.generator_max_input_len,
            max_output_len=self.ecfg.generator_max_output_len,
        )

        self._baselines: Optional[Any] = None

        ensure_dir(self.ecfg.results_dir)
        ensure_dir(self.ecfg.figures_dir)

    def _build_baselines(self) -> Any:
        from sras.baselines.registry import BaselineRegistry
        doc_ids = self.corpus.doc_ids
        doc_texts = [self.corpus.get_text(doc_id) for doc_id in doc_ids]
        doc_embeddings = self.embed_store.get_batch_by_ids(doc_ids)
        return BaselineRegistry(doc_ids, doc_texts, doc_embeddings, self.ecfg.baseline, self.device)

    def _get_baselines(self) -> Any:
        if self._baselines is None:
            self._baselines = self._build_baselines()
        return self._baselines

    def _select_top_k_model(
        self,
        model: CrossAttentionSelector,
        q_emb: torch.Tensor,
        pool_embs: torch.Tensor,
        k: int,
    ) -> List[int]:
        model.eval()
        with torch.no_grad():
            scores = model(q_emb, pool_embs)
        return torch.topk(scores, min(k, scores.shape[0])).indices.tolist()

    def _rank_all_model(
        self,
        model: CrossAttentionSelector,
        q_emb: torch.Tensor,
        pool_embs: torch.Tensor,
    ) -> List[int]:
        """Full descending-score ranking of every pool index (not just top-k).

        Used to compute MRR/R@1 in addition to the top-k selection, without
        a second forward pass.
        """
        model.eval()
        with torch.no_grad():
            scores = model(q_emb, pool_embs)
        return torch.argsort(scores, descending=True).tolist()

    @staticmethod
    def _rank_of(target_id: str, ranked_ids: List[str]) -> int:
        """1-based rank of target_id within ranked_ids; len(ranked_ids)+1 if absent."""
        try:
            return ranked_ids.index(target_id) + 1
        except ValueError:
            return len(ranked_ids) + 1

    def _select_random(self, pool_size: int, k: int, rng: random.Random) -> List[int]:
        indices = list(range(pool_size))
        rng.shuffle(indices)
        return indices[:k]

    def _select_oracle(self, rewards: List[float], k: int) -> List[int]:
        return sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:k]

    def _inject_distractors(
        self,
        pool_doc_ids: List[str],
        pool_embs: torch.Tensor,
        rng: random.Random,
    ) -> Tuple[List[str], torch.Tensor]:
        n = len(pool_doc_ids)
        augmented_ids = list(pool_doc_ids)
        augmented_embs = pool_embs

        if self.ecfg.noise_distractor_rate > 0:
            n_noise = max(1, int(n * self.ecfg.noise_distractor_rate))
            noise_ids = self.corpus.sample_doc_ids(n_noise, exclude=pool_doc_ids, rng=rng)
            noise_embs = self.embed_store.get_batch_by_ids(noise_ids)
            augmented_ids += noise_ids
            augmented_embs = torch.cat([augmented_embs, noise_embs], dim=0)

        if self.ecfg.redundant_distractor_rate > 0:
            n_red = max(1, int(n * self.ecfg.redundant_distractor_rate))
            rep_ids = rng.choices(pool_doc_ids, k=n_red)
            rep_embs = self.embed_store.get_batch_by_ids(rep_ids)
            augmented_ids += rep_ids
            augmented_embs = torch.cat([augmented_embs, rep_embs], dim=0)

        if self.ecfg.adversarial_distractor_rate > 0:
            n_adv = max(1, int(n * self.ecfg.adversarial_distractor_rate))
            noise = torch.randn(n_adv, augmented_embs.shape[1], device=self.device) * 0.1
            adv_embs = augmented_embs[torch.randperm(len(augmented_ids))[:n_adv]] + noise
            adv_ids = [f"__adversarial_{i}__" for i in range(n_adv)]
            augmented_ids += adv_ids
            for adv_id, adv_emb in zip(adv_ids, adv_embs):
                self.corpus._id_to_doc[adv_id] = {
                    "id": adv_id,
                    "text": "",
                    "category": "adversarial",
                }
            augmented_embs = torch.cat([augmented_embs, adv_embs], dim=0)

        return augmented_ids, augmented_embs

    def _eval_dataset(
        self,
        questions: List[str],
        golds: List[str],
        selector_name: str,
        model: Optional[CrossAttentionSelector],
        pool_size: int,
        oracle_rewards: Optional[List[List[float]]] = None,
        oracle_doc_ids: Optional[List[List[str]]] = None,
        baseline_selector: Optional[Any] = None,
    ) -> Dict:
        rng = random.Random(self.ecfg.seed)
        k = self.ecfg.top_k
        has_oracle = oracle_rewards is not None and oracle_doc_ids is not None

        all_preds: List[str] = []
        all_golds: List[str] = []
        all_selected_ids: List[List[str]] = []
        failure_cases: List[Dict] = []

        by_qtype_f1: Dict[str, List[float]] = defaultdict(list)
        by_qtype_sel_fail: Dict[str, int] = defaultdict(int)
        reciprocal_ranks: List[float] = []
        r1_flags: List[float] = []
        rk_flags: List[float] = []

        q_embs = encode_queries_batch(
            questions,
            self.dcfg.embedding_model,
            self.device,
            batch_size=self.ecfg.batch_size,
        )

        for idx in tqdm(range(len(questions)), desc=f"Eval [{selector_name}]", leave=False):
            question = questions[idx]
            gold = golds[idx]
            q_emb = q_embs[idx]
            qtype = _classify_question(question)

            # Gold document identity for retrieval metrics (MRR/R@1/R@k): the
            # doc_id with the highest reward in the precomputed reward matrix,
            # captured before any distractor augmentation below.
            gold_doc_id: Optional[str] = None
            if has_oracle and idx < len(oracle_doc_ids) and idx < len(oracle_rewards):
                orig_ids, orig_rewards = oracle_doc_ids[idx], oracle_rewards[idx]
                if orig_ids and orig_rewards:
                    gold_doc_id = orig_ids[max(range(len(orig_rewards)), key=lambda i: orig_rewards[i])]

            if oracle_doc_ids is not None and idx < len(oracle_doc_ids):
                pool_doc_ids = oracle_doc_ids[idx]
            else:
                pool_doc_ids = self.corpus.sample_doc_ids(pool_size, rng=rng)

            pool_embs = self.embed_store.get_batch_by_ids(pool_doc_ids)

            if any([
                self.ecfg.noise_distractor_rate,
                self.ecfg.redundant_distractor_rate,
                self.ecfg.adversarial_distractor_rate,
            ]):
                pool_doc_ids, pool_embs = self._inject_distractors(pool_doc_ids, pool_embs, rng)

            full_ranked_doc_ids: Optional[List[str]] = None  # for MRR/R@1, when computable

            if selector_name == "random":
                shuffled = list(range(len(pool_doc_ids)))
                rng.shuffle(shuffled)
                sel_indices = shuffled[:k]
                full_ranked_doc_ids = [pool_doc_ids[i] for i in shuffled]
            elif selector_name == "oracle" and oracle_rewards is not None and idx < len(oracle_rewards):
                sel_indices = self._select_oracle(oracle_rewards[idx], k)
                # Oracle selects strictly by descending reward, so it always
                # ranks the gold document (max-reward doc) first by construction.
                full_ranked_doc_ids = [pool_doc_ids[i] for i in
                                        sorted(range(len(oracle_rewards[idx])),
                                               key=lambda i: oracle_rewards[idx][i], reverse=True)]
            elif baseline_selector is not None:
                from sras.baselines.bm25_selector import BM25Selector
                from sras.baselines.dense_selector import DenseSelector
                from sras.baselines.dpr_selector import DPRSelector
                from sras.baselines.hybrid_selector import HybridSelector
                from sras.baselines.modern_dense_selector import ModernDenseSelector
                clean_pool_ids = [d for d in pool_doc_ids if not d.startswith("__")]
                full_k = len(clean_pool_ids)
                if isinstance(baseline_selector, HybridSelector):
                    full_ranked_doc_ids = baseline_selector.select_top_k(question, q_emb, full_k, candidate_doc_ids=clean_pool_ids)
                elif isinstance(baseline_selector, (DPRSelector, ModernDenseSelector)):
                    # DPR / BGE / E5 use their own question encoder; pass the raw question string
                    full_ranked_doc_ids = baseline_selector.select_top_k(question, full_k, candidate_doc_ids=clean_pool_ids)
                elif isinstance(baseline_selector, DenseSelector):
                    full_ranked_doc_ids = baseline_selector.select_top_k(q_emb, full_k, candidate_doc_ids=clean_pool_ids)
                elif isinstance(baseline_selector, BM25Selector):
                    full_ranked_doc_ids = baseline_selector.select_top_k(question, full_k, candidate_doc_ids=clean_pool_ids)
                else:
                    full_ranked_doc_ids = list(clean_pool_ids)
                sel_doc_ids = full_ranked_doc_ids[:k]
                _pool_id_to_idx = {d: i for i, d in enumerate(pool_doc_ids)}
                sel_indices = [_pool_id_to_idx[d] for d in sel_doc_ids if d in _pool_id_to_idx]
            elif model is not None:
                full_rank_indices = self._rank_all_model(model, q_emb, pool_embs)
                full_ranked_doc_ids = [pool_doc_ids[i] for i in full_rank_indices]
                sel_indices = full_rank_indices[:k]
            else:
                shuffled = list(range(len(pool_doc_ids)))
                rng.shuffle(shuffled)
                sel_indices = shuffled[:k]
                full_ranked_doc_ids = [pool_doc_ids[i] for i in shuffled]

            if gold_doc_id is not None and full_ranked_doc_ids is not None:
                rank = self._rank_of(gold_doc_id, full_ranked_doc_ids)
                reciprocal_ranks.append(1.0 / rank)
                r1_flags.append(1.0 if rank == 1 else 0.0)
                rk_flags.append(1.0 if rank <= k else 0.0)

            selected_doc_ids = [pool_doc_ids[i] for i in sel_indices if i < len(pool_doc_ids)]
            selected_texts = []
            for doc_id in selected_doc_ids:
                try:
                    selected_texts.append(self.corpus.get_text(doc_id))
                except (KeyError, Exception):
                    pass

            try:
                pred = self.generator.generate(question, selected_texts)
            except Exception as e:
                logger.warning("Generator failed for question %d: %s", idx, e)
                pred = ""

            all_preds.append(pred)
            all_golds.append(gold)
            all_selected_ids.append(selected_doc_ids)

            item_metrics = self.metrics.compute_single(pred, gold)
            f1 = item_metrics.get("relaxed_f1", 0.0)
            by_qtype_f1[qtype].append(f1)

            if f1 < 0.1:
                by_qtype_sel_fail[qtype] += 1
                if self.ecfg.failure_analysis and len(failure_cases) < self.ecfg.failure.max_failure_cases:
                    failure_cases.append({
                        "idx": idx,
                        "question": question,
                        "question_type": qtype,
                        "gold": gold,
                        "pred": pred,
                        "selected_doc_ids": selected_doc_ids,
                        "metrics": item_metrics,
                    })

        per_sample = self.metrics.compute_batch(all_preds, all_golds)
        aggregated = self.metrics.aggregate(per_sample)

        per_qtype: List[Dict] = []
        if self.ecfg.per_question_type_analysis:
            for qtype in sorted(by_qtype_f1.keys()):
                f1s = by_qtype_f1[qtype]
                per_qtype.append({
                    "question_type": qtype,
                    "count": len(f1s),
                    "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
                    "failure_count": by_qtype_sel_fail[qtype],
                })

        result = {
            "selector": selector_name,
            "n_questions": len(questions),
            "metrics": aggregated,
            "per_question_type": per_qtype,
            "per_sample_predictions": [
                {"question": q, "gold": g, "pred": p, "selected_doc_ids": sids}
                for q, g, p, sids in zip(questions, all_golds, all_preds, all_selected_ids)
            ],
        }
        if reciprocal_ranks:
            # Only populated when oracle_rewards/oracle_doc_ids were provided
            # (the internal-corpus evaluation path), matching Table 1's MRR/R@1/R@k.
            result["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
            result["r_at_1"] = sum(r1_flags) / len(r1_flags)
            result[f"r_at_{k}"] = sum(rk_flags) / len(rk_flags)
        if self.ecfg.failure_analysis:
            result["failure_cases"] = failure_cases
            result["failure_rate"] = len(failure_cases) / max(len(questions), 1)

        return result

    def _eval_baselines_on_dataset(
        self,
        questions: List[str],
        golds: List[str],
        pool_size: int,
    ) -> Dict[str, Dict]:
        if not self.ecfg.run_baselines:
            return {}

        baselines = self._get_baselines()
        results: Dict[str, Dict] = {}

        for name in baselines.available():
            logger.info("Evaluating baseline: %s", name)
            try:
                selector = baselines.get(name)
                result = self._eval_dataset(
                    questions, golds, name, None,
                    pool_size=pool_size,
                    baseline_selector=selector,
                )
                results[name] = result
            except Exception as e:
                logger.error("Baseline eval failed for %s: %s", name, e)

        return results

    def evaluate_on_internal(self, selector_name: str, model: Optional[CrossAttentionSelector] = None) -> Dict:
        reward_dataset = RewardDataset(self.dcfg.reward_matrix_path)
        questions = reward_dataset.questions()
        oracle_doc_ids = [reward_dataset.get_doc_ids(q) for q in questions]
        oracle_rewards = [reward_dataset.get_rewards(q) for q in questions]

        qa_dataset = QADataset(self.dcfg.qa_pairs_path)
        qa_map = {item["question"]: item["answer"] for item in qa_dataset}
        raw_golds = [qa_map.get(q, "") for q in questions]

        # Use first-sentence truncation to avoid penalising short predictions
        # against paragraph-length gold answers. Full gold is kept for display.
        golds = [_short_gold(g) for g in raw_golds]

        valid = [(q, g, od, orw) for q, g, od, orw in zip(questions, golds, oracle_doc_ids, oracle_rewards) if g]
        if not valid:
            raise ValueError("No valid internal evaluation samples found.")

        questions, golds, oracle_doc_ids, oracle_rewards = zip(*valid)

        return self._eval_dataset(
            list(questions), list(golds), selector_name, model,
            pool_size=self.ecfg.candidate_pool_size,
            oracle_rewards=list(oracle_rewards),
            oracle_doc_ids=list(oracle_doc_ids),
        )

    def evaluate_on_squad(self, selector_name: str, model: Optional[CrossAttentionSelector] = None) -> Dict:
        """Legacy SQuAD evaluation (no context; broken by design).
        Kept for backward compatibility. Use evaluate_on_squad_eval instead."""
        squad = SquadDataset(self.dcfg.squad_subset_path)
        return self._eval_dataset(
            squad.questions(), squad.answers(), selector_name, model,
            pool_size=self.ecfg.candidate_pool_size,
        )

    def evaluate_on_squad_eval(
        self,
        selector_name: str,
        model: Optional[CrossAttentionSelector] = None,
        baseline_selector: Optional[Any] = None,
    ) -> Dict:
        """Context-aware SQuAD evaluation.

        Uses the SQuAD corpus built by setup_squad_eval.py.  For each question
        the candidate pool is constructed as:
          - 1 guaranteed correct context passage (the Wikipedia paragraph that
            contains the answer)
          - (pool_size - 1) randomly sampled docs from the internal corpus

        This tests whether the selector can identify the relevant passage in a
        mixed pool, which is the standard open-domain RAG evaluation setting.

        ``baseline_selector`` routes BM25/Dense/DPR/Hybrid/modern-dense
        baselines (and "random") through this same SQuAD-aware pool
        construction, so that baseline results on squad_eval are directly
        comparable to the neural-selector results rather than being evaluated
        against a pool sampled purely from the internal corpus (which would
        never contain the correct passage at all).

        In addition to relaxed-F1 and hit-rate (= R@top_k), this also reports
        MRR and R@1 by ranking the *entire* candidate pool rather than just
        the top-k selection, matching the internal-corpus metric set
        (Table 1) so the two are directly comparable.
        """
        import torch as _torch

        squad_eval = SquadEvalDataset(self.dcfg.squad_eval_pairs_path)
        questions       = squad_eval.questions()
        golds           = squad_eval.answers()   # already short spans
        context_doc_ids = squad_eval.context_doc_ids()

        # ── Load the SQuAD context corpus ──────────────────────────────────
        from sras.data.corpus import CorpusStore
        from sras.data.embeddings import EmbeddingStore

        squad_corpus = CorpusStore(self.dcfg.squad_contexts_path)
        squad_embeds = EmbeddingStore(
            self.dcfg.squad_embeddings_path,
            squad_corpus.doc_ids,
            self.device,
        )

        rng = random.Random(self.ecfg.seed)
        k   = self.ecfg.top_k

        all_preds: List[str] = []
        all_golds: List[str] = []
        all_selected_ids: List[List[str]] = []
        failure_cases: List[Dict] = []
        by_qtype_f1: Dict[str, List[float]] = defaultdict(list)
        by_qtype_sel_fail: Dict[str, int]   = defaultdict(int)
        reciprocal_ranks: List[float] = []
        r1_flags: List[float] = []

        q_embs = encode_queries_batch(
            questions,
            self.dcfg.embedding_model,
            self.device,
            batch_size=self.ecfg.batch_size,
        )

        pool_size = self.ecfg.candidate_pool_size

        for idx in tqdm(range(len(questions)), desc=f"Eval SQuAD [{selector_name}]", leave=False):
            question        = questions[idx]
            gold            = golds[idx]
            q_emb           = q_embs[idx]
            qtype           = _classify_question(question)
            correct_doc_id  = context_doc_ids[idx]

            # Build mixed pool: 1 correct SQuAD passage + (pool_size-1) internal docs
            n_distractors = max(0, pool_size - 1)
            distractor_ids = self.corpus.sample_doc_ids(n_distractors, rng=rng)
            pool_doc_ids   = [correct_doc_id] + distractor_ids

            # Gather embeddings from their respective stores
            try:
                correct_emb = squad_embeds.get_batch_by_ids([correct_doc_id])
            except Exception:
                correct_emb = _torch.zeros(1, self.embed_store._embeddings.shape[1],
                                           device=self.device)

            distractor_embs = self.embed_store.get_batch_by_ids(distractor_ids)
            pool_embs = _torch.cat([correct_emb, distractor_embs], dim=0)

            # ── Rank the full pool (needed for MRR/R@1) then take the top-k ──
            if selector_name == "random":
                full_rank_indices = list(range(len(pool_doc_ids)))
                rng.shuffle(full_rank_indices)
                ranked_doc_ids = [pool_doc_ids[i] for i in full_rank_indices]
            elif baseline_selector is not None:
                from sras.baselines.bm25_selector import BM25Selector
                from sras.baselines.dense_selector import DenseSelector
                from sras.baselines.dpr_selector import DPRSelector
                from sras.baselines.hybrid_selector import HybridSelector
                from sras.baselines.modern_dense_selector import ModernDenseSelector
                full_k = len(pool_doc_ids)
                if isinstance(baseline_selector, HybridSelector):
                    ranked_doc_ids = baseline_selector.select_top_k(
                        question, q_emb, full_k, candidate_doc_ids=pool_doc_ids)
                elif isinstance(baseline_selector, (DPRSelector, ModernDenseSelector, BM25Selector)):
                    ranked_doc_ids = baseline_selector.select_top_k(
                        question, full_k, candidate_doc_ids=pool_doc_ids)
                elif isinstance(baseline_selector, DenseSelector):
                    ranked_doc_ids = baseline_selector.select_top_k(
                        q_emb, full_k, candidate_doc_ids=pool_doc_ids)
                else:
                    ranked_doc_ids = list(pool_doc_ids)
            elif model is not None:
                full_rank_indices = self._rank_all_model(model, q_emb, pool_embs)
                ranked_doc_ids = [pool_doc_ids[i] for i in full_rank_indices]
            else:
                full_rank_indices = list(range(len(pool_doc_ids)))
                rng.shuffle(full_rank_indices)
                ranked_doc_ids = [pool_doc_ids[i] for i in full_rank_indices]

            selected_doc_ids = ranked_doc_ids[:k]
            rank = self._rank_of(correct_doc_id, ranked_doc_ids)
            reciprocal_ranks.append(1.0 / rank)
            r1_flags.append(1.0 if rank == 1 else 0.0)

            selected_texts   = []
            for doc_id in selected_doc_ids:
                try:
                    # correct passage is in squad_corpus; distractors are in internal corpus
                    if doc_id.startswith("squad_ctx_"):
                        selected_texts.append(squad_corpus.get_text(doc_id))
                    else:
                        selected_texts.append(self.corpus.get_text(doc_id))
                except Exception:
                    pass

            try:
                pred = self.generator.generate(question, selected_texts)
            except Exception as e:
                logger.warning("Generator failed for SQuAD question %d: %s", idx, e)
                pred = ""

            all_preds.append(pred)
            all_golds.append(gold)
            all_selected_ids.append(selected_doc_ids)

            item_metrics = self.metrics.compute_single(pred, gold)
            f1 = item_metrics.get("relaxed_f1", 0.0)
            by_qtype_f1[qtype].append(f1)

            # Also track whether the correct passage was selected (= R@top_k)
            selector_hit = correct_doc_id in selected_doc_ids
            if not selector_hit:
                by_qtype_sel_fail[qtype] += 1
                if self.ecfg.failure_analysis and len(failure_cases) < self.ecfg.failure.max_failure_cases:
                    failure_cases.append({
                        "idx": idx,
                        "question": question,
                        "question_type": qtype,
                        "gold": gold,
                        "pred": pred,
                        "correct_doc_id": correct_doc_id,
                        "selected_doc_ids": selected_doc_ids,
                        "selector_hit": selector_hit,
                        "rank": rank,
                        "metrics": item_metrics,
                    })

        per_sample = self.metrics.compute_batch(all_preds, all_golds)
        aggregated = self.metrics.aggregate(per_sample)

        # Selector hit rate (fraction of questions where the correct passage was selected)
        hit_rate = 1.0 - (sum(by_qtype_sel_fail.values()) / max(len(questions), 1))
        mrr = sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)
        r_at_1 = sum(r1_flags) / max(len(r1_flags), 1)

        per_qtype: List[Dict] = []
        if self.ecfg.per_question_type_analysis:
            for qtype in sorted(by_qtype_f1.keys()):
                f1s = by_qtype_f1[qtype]
                per_qtype.append({
                    "question_type": qtype,
                    "count": len(f1s),
                    "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
                    "failure_count": by_qtype_sel_fail[qtype],
                })

        result = {
            "selector":        selector_name,
            "n_questions":     len(questions),
            "metrics":         aggregated,
            "selector_hit_rate": hit_rate,
            "mrr":             mrr,
            "r_at_1":          r_at_1,
            f"r_at_{k}":       hit_rate,
            "per_question_type": per_qtype,
            "per_sample_predictions": [
                {"question": q, "gold": g, "pred": p, "selected_doc_ids": sids}
                for q, g, p, sids in zip(questions, all_golds, all_preds, all_selected_ids)
            ],
        }
        if self.ecfg.failure_analysis:
            result["failure_cases"] = failure_cases
            result["failure_rate"]  = len(failure_cases) / max(len(questions), 1)

        return result

    def evaluate_on_external(
        self,
        path: str,
        selector_name: str,
        model: Optional[CrossAttentionSelector] = None,
    ) -> Dict:
        from sras.data.datasets import ExternalQADataset
        ext = ExternalQADataset(path)
        questions = [ext[i]["question"] for i in range(len(ext))]
        golds = [ext[i]["answer"] for i in range(len(ext))]
        return self._eval_dataset(
            questions, golds, selector_name, model,
            pool_size=self.ecfg.candidate_pool_size,
        )

    def run_full_evaluation(
        self,
        selector_model_registry: Dict[str, str],
    ) -> Dict[str, Dict]:
        all_results: Dict[str, Dict] = {}

        for variant_name, checkpoint_path in selector_model_registry.items():
            logger.info("Evaluating variant: %s", variant_name)
            try:
                model = load_selector(
                    checkpoint_path, self.device,
                    model_kwargs={
                        "doc_emb_dim": self.mcfg.doc_emb_dim,
                        "hidden_dim": self.mcfg.hidden_dim,
                        "dropout": 0.0,
                        "use_layer_norm": self.mcfg.use_layer_norm,
                        "use_residual": self.mcfg.use_residual,
                    },
                )
            except FileNotFoundError:
                logger.warning("Checkpoint not found for %s: %s", variant_name, checkpoint_path)
                continue

            variant_results: Dict[str, Dict] = {}

            for dataset_name in self.ecfg.datasets:
                try:
                    if dataset_name == "internal":
                        result = self.evaluate_on_internal(variant_name, model)
                    elif dataset_name == "squad_eval":
                        result = self.evaluate_on_squad_eval(variant_name, model)
                    elif dataset_name == "squad":
                        result = self.evaluate_on_squad(variant_name, model)
                    else:
                        result = self.evaluate_on_external(dataset_name, variant_name, model)
                    variant_results[dataset_name] = result
                except Exception as e:
                    logger.error("Eval failed for %s on %s: %s", variant_name, dataset_name, e)

            if variant_results:
                all_results[variant_name] = variant_results
                out_dir = os.path.join(self.ecfg.figures_dir, variant_name)
                ensure_dir(out_dir)
                for ds_name, ds_result in variant_results.items():
                    save_json(ds_result.get("metrics", {}), os.path.join(out_dir, f"eval_metrics_{ds_name}.json"))
                    save_json(ds_result, os.path.join(self.ecfg.results_dir, f"{variant_name}_{ds_name}_results.json"))
                logger.info(
                    "Results for %s: %s",
                    variant_name,
                    {ds: r.get("metrics") for ds, r in variant_results.items()},
                )

        for dataset_name in self.ecfg.datasets:
            try:
                if dataset_name == "squad_eval":
                    # squad_eval requires the SQuAD-aware pool construction (1
                    # guaranteed-correct passage + internal-corpus distractors).
                    # Routing baselines/random through the generic _eval_dataset
                    # path here would sample pools purely from the internal
                    # corpus, which structurally never contains the correct
                    # SQuAD passage, making the comparison meaningless. See
                    # evaluate_on_squad_eval() docstring.
                    random_result = self.evaluate_on_squad_eval("random", model=None)
                    all_results.setdefault("random", {})[dataset_name] = random_result
                    save_json(
                        random_result,
                        os.path.join(self.ecfg.results_dir, f"random_{dataset_name}_results.json"),
                    )
                    if self.ecfg.run_baselines:
                        baselines = self._get_baselines()
                        for bname in baselines.available():
                            if bname == "random":
                                continue
                            try:
                                selector = baselines.get(bname)
                                bresult = self.evaluate_on_squad_eval(
                                    bname, model=None, baseline_selector=selector,
                                )
                                all_results.setdefault(bname, {})[dataset_name] = bresult
                                save_json(
                                    bresult,
                                    os.path.join(self.ecfg.results_dir, f"{bname}_{dataset_name}_results.json"),
                                )
                            except Exception as e:
                                logger.error("Baseline eval failed for %s on squad_eval: %s", bname, e)
                    continue

                questions, golds = self._load_dataset_pairs(dataset_name)
                random_result = self._eval_dataset(
                    questions, golds, "random", None,
                    pool_size=self.ecfg.candidate_pool_size,
                )
                all_results.setdefault("random", {})[dataset_name] = random_result
                save_json(
                    random_result,
                    os.path.join(self.ecfg.results_dir, f"random_{dataset_name}_results.json"),
                )

                if self.ecfg.run_baselines:
                    baseline_results = self._eval_baselines_on_dataset(
                        questions, golds, self.ecfg.candidate_pool_size
                    )
                    for bname, bresult in baseline_results.items():
                        all_results.setdefault(bname, {})[dataset_name] = bresult
                        save_json(
                            bresult,
                            os.path.join(self.ecfg.results_dir, f"{bname}_{dataset_name}_results.json"),
                        )
            except Exception as e:
                logger.error("Baseline/random eval failed on %s: %s", dataset_name, e)

        # ── Oracle evaluation ────────────────────────────────────────────────
        # Oracle must be evaluated via evaluate_on_internal() because it needs
        # the per-question reward matrix (oracle_rewards + oracle_doc_ids).
        # Evaluating through BaselineRegistry gave None selector → random fallback.
        if "internal" in self.ecfg.datasets:
            try:
                logger.info("Evaluating oracle selector on internal dataset")
                oracle_result = self.evaluate_on_internal("oracle", model=None)
                all_results.setdefault("oracle", {})["internal"] = oracle_result
                save_json(
                    oracle_result,
                    os.path.join(self.ecfg.results_dir, "oracle_internal_results.json"),
                )
                logger.info(
                    "Oracle internal F1=%.4f",
                    oracle_result.get("metrics", {}).get("relaxed_f1", 0.0),
                )
            except Exception as e:
                logger.error("Oracle eval failed on internal: %s", e)

        return all_results

    def _load_dataset_pairs(self, dataset_name: str) -> Tuple[List[str], List[str]]:
        if dataset_name == "internal":
            qa = QADataset(self.dcfg.qa_pairs_path)
            questions = [item["question"] for item in qa]
            # Use first-sentence gold for consistent evaluation
            golds = [_short_gold(item["answer"]) for item in qa]
            return questions, golds
        elif dataset_name == "squad_eval":
            sq = SquadEvalDataset(self.dcfg.squad_eval_pairs_path)
            return sq.questions(), sq.answers()
        elif dataset_name == "squad":
            sq = SquadDataset(self.dcfg.squad_subset_path)
            return sq.questions(), sq.answers()
        else:
            from sras.data.datasets import ExternalQADataset
            ext = ExternalQADataset(dataset_name)
            return [ext[i]["question"] for i in range(len(ext))], [ext[i]["answer"] for i in range(len(ext))]
