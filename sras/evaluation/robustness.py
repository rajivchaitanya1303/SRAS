from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from sras.config.schema import RobustnessConfig
from sras.evaluation.metrics import MetricsComputer
from sras.rl.rewards import relaxed_f1 as _relaxed_f1
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _noise_doc(doc: str, rate: float, rng: random.Random) -> str:
    if rate <= 0.0:
        return doc
    words = doc.split()
    if not words:
        return doc
    n_corrupt = max(1, int(len(words) * rate))
    indices = rng.sample(range(len(words)), min(n_corrupt, len(words)))
    for idx in indices:
        words[idx] = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=len(words[idx])))
    return " ".join(words)


def _make_redundant_doc(doc: str, rng: random.Random) -> str:
    sentences = [s.strip() for s in doc.split(".") if s.strip()]
    if not sentences:
        return doc
    repeated = rng.choice(sentences)
    return doc + " " + repeated + "."


def _make_adversarial_doc(question: str, corpus_docs: List[str], rng: random.Random) -> str:
    q_words = set(question.lower().split())
    scored = []
    for d in corpus_docs:
        overlap = len(q_words & set(d.lower().split()))
        scored.append((overlap, d))
    scored.sort(key=lambda x: -x[0])
    if scored:
        return scored[0][1]
    return rng.choice(corpus_docs) if corpus_docs else ""


def _inject_distractors(
    candidate_pool: List[Dict],
    question: str,
    all_docs: List[Dict],
    noise_rate: float,
    redundant_rate: float,
    adversarial_rate: float,
    rng: random.Random,
) -> List[Dict]:
    pool = []
    all_texts = [d["content"] for d in all_docs]

    for doc in candidate_pool:
        content = doc["content"]
        if noise_rate > 0.0 and rng.random() < noise_rate:
            content = _noise_doc(content, noise_rate, rng)
        if redundant_rate > 0.0 and rng.random() < redundant_rate:
            content = _make_redundant_doc(content, rng)
        pool.append({**doc, "content": content})

    n_adversarial = int(len(candidate_pool) * adversarial_rate)
    for _ in range(n_adversarial):
        adv_text = _make_adversarial_doc(question, all_texts, rng)
        pool.append({"content": adv_text, "doc_id": f"adv_{rng.randint(0, 99999)}", "category": "adversarial"})

    return pool


class RobustnessEvaluator:
    def __init__(self, config: RobustnessConfig) -> None:
        self.config = config
        self.metrics = MetricsComputer()
        ensure_dir(config.results_dir)

    def _select_candidates(
        self,
        model: Any,
        q_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        k: int,
    ) -> List[int]:
        if doc_embs.shape[0] == 0:
            return []
        k = min(k, doc_embs.shape[0])
        with torch.no_grad():
            scores = model(q_emb.unsqueeze(0) if q_emb.dim() == 1 else q_emb, doc_embs)
        return torch.topk(scores, k).indices.tolist()

    def _eval_single(
        self,
        model: Any,
        qa_pairs: List[Dict],
        corpus_docs: List[Dict],
        doc_embs: torch.Tensor,
        q_embs: List[torch.Tensor],
        noise_rate: float,
        redundant_rate: float,
        adversarial_rate: float,
        pool_size: int,
        top_k: int,
        seed: int,
    ) -> Dict[str, float]:
        rng = random.Random(seed)
        all_f1: List[float] = []

        for idx, qa in enumerate(qa_pairs):
            question = qa["question"]
            answer = qa["answer"]

            n_pool = min(pool_size, len(corpus_docs))
            pool_indices = rng.sample(range(len(corpus_docs)), n_pool)
            pool_docs = [corpus_docs[i] for i in pool_indices]
            pool_embs = doc_embs[pool_indices]

            if noise_rate > 0.0 or redundant_rate > 0.0 or adversarial_rate > 0.0:
                pool_docs = _inject_distractors(
                    pool_docs, question, corpus_docs,
                    noise_rate, redundant_rate, adversarial_rate, rng,
                )
                if len(pool_docs) > len(pool_indices):
                    n_extra = len(pool_docs) - len(pool_indices)
                    extra_embs = torch.zeros(n_extra, doc_embs.shape[1], device=pool_embs.device)
                    pool_embs = torch.cat([pool_embs, extra_embs], dim=0)

            q_emb = q_embs[idx] if idx < len(q_embs) else q_embs[0]
            selected = self._select_candidates(model, q_emb, pool_embs, top_k)

            selected_texts = [pool_docs[i]["content"] for i in selected if i < len(pool_docs)]
            context = " ".join(selected_texts)
            f1 = _relaxed_f1(context, answer)
            all_f1.append(f1)

        return {
            "mean_f1": float(sum(all_f1) / len(all_f1)) if all_f1 else 0.0,
            "n_samples": len(all_f1),
        }

    def sweep_noise(
        self,
        model: Any,
        qa_pairs: List[Dict],
        corpus_docs: List[Dict],
        doc_embs: torch.Tensor,
        q_embs: List[torch.Tensor],
        pool_size: int = 30,
        top_k: int = 3,
    ) -> List[Dict]:
        results = []
        for rate in self.config.noise_rates:
            trial_scores = []
            for trial in range(self.config.n_trials):
                stats = self._eval_single(
                    model, qa_pairs, corpus_docs, doc_embs, q_embs,
                    noise_rate=rate, redundant_rate=0.0, adversarial_rate=0.0,
                    pool_size=pool_size, top_k=top_k, seed=42 + trial,
                )
                trial_scores.append(stats["mean_f1"])
            avg = sum(trial_scores) / len(trial_scores)
            results.append({"noise_rate": rate, "mean_f1": avg, "trials": trial_scores})
            logger.info("Noise rate %.2f | mean_f1=%.4f", rate, avg)
        return results

    def sweep_redundant(
        self,
        model: Any,
        qa_pairs: List[Dict],
        corpus_docs: List[Dict],
        doc_embs: torch.Tensor,
        q_embs: List[torch.Tensor],
        pool_size: int = 30,
        top_k: int = 3,
    ) -> List[Dict]:
        results = []
        for rate in self.config.redundant_rates:
            trial_scores = []
            for trial in range(self.config.n_trials):
                stats = self._eval_single(
                    model, qa_pairs, corpus_docs, doc_embs, q_embs,
                    noise_rate=0.0, redundant_rate=rate, adversarial_rate=0.0,
                    pool_size=pool_size, top_k=top_k, seed=42 + trial,
                )
                trial_scores.append(stats["mean_f1"])
            avg = sum(trial_scores) / len(trial_scores)
            results.append({"redundant_rate": rate, "mean_f1": avg, "trials": trial_scores})
            logger.info("Redundant rate %.2f | mean_f1=%.4f", rate, avg)
        return results

    def sweep_adversarial(
        self,
        model: Any,
        qa_pairs: List[Dict],
        corpus_docs: List[Dict],
        doc_embs: torch.Tensor,
        q_embs: List[torch.Tensor],
        pool_size: int = 30,
        top_k: int = 3,
    ) -> List[Dict]:
        results = []
        for rate in self.config.adversarial_rates:
            trial_scores = []
            for trial in range(self.config.n_trials):
                stats = self._eval_single(
                    model, qa_pairs, corpus_docs, doc_embs, q_embs,
                    noise_rate=0.0, redundant_rate=0.0, adversarial_rate=rate,
                    pool_size=pool_size, top_k=top_k, seed=42 + trial,
                )
                trial_scores.append(stats["mean_f1"])
            avg = sum(trial_scores) / len(trial_scores)
            results.append({"adversarial_rate": rate, "mean_f1": avg, "trials": trial_scores})
            logger.info("Adversarial rate %.2f | mean_f1=%.4f", rate, avg)
        return results

    def sweep_domain_shift(
        self,
        model: Any,
        qa_pairs_by_category: Dict[str, List[Dict]],
        corpus_docs_by_category: Dict[str, List[Dict]],
        doc_embs_by_category: Dict[str, torch.Tensor],
        q_embs_by_category: Dict[str, List[torch.Tensor]],
        pool_size: int = 30,
        top_k: int = 3,
    ) -> List[Dict]:
        results = []
        for cat in self.config.domain_shift_categories:
            if cat not in qa_pairs_by_category:
                logger.warning("Domain shift category '%s' not found, skipping", cat)
                continue
            trial_scores = []
            for trial in range(self.config.n_trials):
                stats = self._eval_single(
                    model,
                    qa_pairs_by_category[cat],
                    corpus_docs_by_category.get(cat, []),
                    doc_embs_by_category.get(cat, torch.zeros(1, 384)),
                    q_embs_by_category.get(cat, [torch.zeros(384)]),
                    noise_rate=0.0, redundant_rate=0.0, adversarial_rate=0.0,
                    pool_size=pool_size, top_k=top_k, seed=42 + trial,
                )
                trial_scores.append(stats["mean_f1"])
            avg = sum(trial_scores) / len(trial_scores)
            results.append({"category": cat, "mean_f1": avg, "trials": trial_scores})
            logger.info("Domain shift '%s' | mean_f1=%.4f", cat, avg)
        return results

    def run_full(
        self,
        model: Any,
        qa_pairs: List[Dict],
        corpus_docs: List[Dict],
        doc_embs: torch.Tensor,
        q_embs: List[torch.Tensor],
        pool_size: int = 30,
        top_k: int = 3,
        qa_pairs_by_category: Optional[Dict[str, List[Dict]]] = None,
        corpus_docs_by_category: Optional[Dict[str, List[Dict]]] = None,
        doc_embs_by_category: Optional[Dict[str, torch.Tensor]] = None,
        q_embs_by_category: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Dict:
        output: Dict = {}

        logger.info("Robustness sweep: noise")
        output["noise"] = self.sweep_noise(model, qa_pairs, corpus_docs, doc_embs, q_embs, pool_size, top_k)

        logger.info("Robustness sweep: redundant")
        output["redundant"] = self.sweep_redundant(model, qa_pairs, corpus_docs, doc_embs, q_embs, pool_size, top_k)

        logger.info("Robustness sweep: adversarial")
        output["adversarial"] = self.sweep_adversarial(model, qa_pairs, corpus_docs, doc_embs, q_embs, pool_size, top_k)

        if all(x is not None for x in [qa_pairs_by_category, corpus_docs_by_category, doc_embs_by_category, q_embs_by_category]):
            logger.info("Robustness sweep: domain shift")
            output["domain_shift"] = self.sweep_domain_shift(
                model, qa_pairs_by_category, corpus_docs_by_category,
                doc_embs_by_category, q_embs_by_category, pool_size, top_k,
            )

        out_path = os.path.join(self.config.results_dir, "robustness_results.json")
        save_json(output, out_path)
        logger.info("Robustness results saved to %s", out_path)
        return output
