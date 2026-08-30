from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from sras.config.schema import FailureAnalysisConfig
from sras.evaluation.metrics import MetricsComputer
from sras.rl.rewards import relaxed_f1 as _relaxed_f1
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)

_QUESTION_TYPE_PREFIXES = ["what", "who", "when", "where", "how", "why", "which", "is", "are", "does", "do", "can"]


def _classify_question(question: str) -> str:
    first = question.strip().lower().split()[0] if question.strip() else "other"
    if first in _QUESTION_TYPE_PREFIXES:
        return first
    return "other"


def _answer_in_doc(doc_text: str, answer: str, threshold: float = 0.1) -> bool:
    if not answer.strip() or not doc_text.strip():
        return False
    answer_tokens = set(answer.lower().split())
    doc_tokens = set(doc_text.lower().split())
    if not answer_tokens:
        return False
    overlap = len(answer_tokens & doc_tokens) / len(answer_tokens)
    return overlap >= threshold


class FailureAnalyzer:
    def __init__(self, config: FailureAnalysisConfig) -> None:
        self.config = config
        self.metrics = MetricsComputer()
        ensure_dir(config.results_dir)

    def _select(
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
            scores = model(q_emb, doc_embs)
        return torch.topk(scores, k).indices.tolist()

    def _is_selector_failure(
        self,
        selected_docs: List[str],
        answer: str,
    ) -> bool:
        for doc in selected_docs:
            if _answer_in_doc(doc, answer, self.config.selector_hit_threshold):
                return False
        return True

    def _is_generator_failure(
        self,
        selected_docs: List[str],
        answer: str,
        generated_answer: str,
    ) -> bool:
        answer_reachable = any(
            _answer_in_doc(d, answer, self.config.selector_hit_threshold) for d in selected_docs
        )
        if not answer_reachable:
            return False
        f1 = _relaxed_f1(generated_answer, answer)
        return f1 < self.config.f1_failure_threshold

    def analyze(
        self,
        model: Any,
        qa_pairs: List[Dict],
        corpus_docs: List[Dict],
        doc_embs: torch.Tensor,
        q_embs: List[torch.Tensor],
        pool_size: int = 30,
        top_k: int = 3,
        generated_answers: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Dict:
        rng = random.Random(seed)

        selector_failures: List[Dict] = []
        generator_failures: List[Dict] = []

        by_qtype_total: Dict[str, int] = defaultdict(int)
        by_qtype_selector_fail: Dict[str, int] = defaultdict(int)
        by_qtype_generator_fail: Dict[str, int] = defaultdict(int)
        by_qtype_f1: Dict[str, List[float]] = defaultdict(list)

        all_f1: List[float] = []

        for idx, qa in enumerate(qa_pairs):
            question = qa["question"]
            answer = qa["answer"]
            qtype = _classify_question(question)

            n_pool = min(pool_size, len(corpus_docs))
            pool_indices = rng.sample(range(len(corpus_docs)), n_pool)
            pool_docs = [corpus_docs[i] for i in pool_indices]
            pool_embs = doc_embs[pool_indices]

            q_emb = q_embs[idx] if idx < len(q_embs) else q_embs[0]
            selected_indices = self._select(model, q_emb, pool_embs, top_k)
            selected_docs = [pool_docs[i]["content"] for i in selected_indices if i < len(pool_docs)]

            context = " ".join(selected_docs)
            f1 = _relaxed_f1(context, answer)
            all_f1.append(f1)

            by_qtype_total[qtype] += 1
            by_qtype_f1[qtype].append(f1)

            sel_fail = self._is_selector_failure(selected_docs, answer)

            gen_answer = generated_answers[idx] if generated_answers and idx < len(generated_answers) else context
            gen_fail = self._is_generator_failure(selected_docs, answer, gen_answer)

            case = {
                "question": question,
                "answer": answer,
                "question_type": qtype,
                "selected_docs": selected_docs,
                "generated_answer": gen_answer,
                "f1": f1,
            }

            if sel_fail:
                by_qtype_selector_fail[qtype] += 1
                if len(selector_failures) < self.config.max_failure_cases:
                    selector_failures.append({**case, "failure_type": "selector"})

            if gen_fail:
                by_qtype_generator_fail[qtype] += 1
                if len(generator_failures) < self.config.max_failure_cases:
                    generator_failures.append({**case, "failure_type": "generator"})

        per_qtype: List[Dict] = []
        for qtype in sorted(by_qtype_total.keys()):
            total = by_qtype_total[qtype]
            sfail = by_qtype_selector_fail[qtype]
            gfail = by_qtype_generator_fail[qtype]
            f1s = by_qtype_f1[qtype]
            per_qtype.append({
                "question_type": qtype,
                "total": total,
                "selector_failures": sfail,
                "generator_failures": gfail,
                "selector_failure_rate": sfail / total if total > 0 else 0.0,
                "generator_failure_rate": gfail / total if total > 0 else 0.0,
                "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            })

        overall_mean_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
        total_n = len(qa_pairs)

        summary = {
            "total_samples": total_n,
            "overall_mean_f1": overall_mean_f1,
            "selector_failure_count": len(selector_failures),
            "generator_failure_count": len(generator_failures),
            "selector_failure_rate": len(selector_failures) / total_n if total_n > 0 else 0.0,
            "generator_failure_rate": len(generator_failures) / total_n if total_n > 0 else 0.0,
        }

        output = {
            "summary": summary,
            "per_question_type": per_qtype,
            "selector_failure_cases": selector_failures,
            "generator_failure_cases": generator_failures,
        }

        out_path = os.path.join(self.config.results_dir, "failure_analysis.json")
        save_json(output, out_path)
        logger.info(
            "Failure analysis complete | selector_fail=%d | generator_fail=%d | mean_f1=%.4f",
            len(selector_failures), len(generator_failures), overall_mean_f1,
        )
        return output

    def compare_failure_rates(
        self,
        results_by_model: Dict[str, Dict],
    ) -> List[Dict]:
        comparison = []
        for label, result in results_by_model.items():
            summary = result.get("summary", {})
            comparison.append({
                "model": label,
                "overall_mean_f1": summary.get("overall_mean_f1", 0.0),
                "selector_failure_rate": summary.get("selector_failure_rate", 0.0),
                "generator_failure_rate": summary.get("generator_failure_rate", 0.0),
                "total_samples": summary.get("total_samples", 0),
            })
        comparison.sort(key=lambda x: -x["overall_mean_f1"])
        out_path = os.path.join(self.config.results_dir, "failure_comparison.json")
        save_json(comparison, out_path)
        return comparison
