from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import torch

from sras.evaluation.metrics import MetricsComputer
from sras.rl.rewards import relaxed_f1 as _relaxed_f1
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class E2EComparison:
    """Bounded end-to-end retriever-generator comparison.

    Compares SRAS selector against retrieval baselines in a full RAG pipeline
    where a shared generator is used for all methods. The selector is the only
    variable; this isolates the selector's contribution to end-to-end quality.
    """

    def __init__(self, output_dir: str = "results/e2e_comparison") -> None:
        self.output_dir = output_dir
        self.metrics = MetricsComputer()
        ensure_dir(output_dir)

    def _select_with_model(
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

    def _select_random(self, n: int, k: int, rng: random.Random) -> List[int]:
        indices = list(range(n))
        rng.shuffle(indices)
        return indices[:k]

    def _run_pipeline(
        self,
        selector_name: str,
        selector: Any,
        questions: List[str],
        golds: List[str],
        corpus_docs: List[Dict],
        doc_embs: torch.Tensor,
        q_embs: List[torch.Tensor],
        generator: Any,
        pool_size: int,
        top_k: int,
        seed: int,
        baseline_selector: Optional[Any] = None,
    ) -> Dict:
        rng = random.Random(seed)
        all_preds: List[str] = []
        all_f1s: List[float] = []

        for idx, (question, gold) in enumerate(zip(questions, golds)):
            n_pool = min(pool_size, len(corpus_docs))
            pool_indices = rng.sample(range(len(corpus_docs)), n_pool)
            pool_docs = [corpus_docs[i] for i in pool_indices]
            pool_embs = doc_embs[pool_indices]

            q_emb = q_embs[idx] if idx < len(q_embs) else q_embs[0]

            if selector_name == "random":
                sel_indices = self._select_random(n_pool, top_k, rng)
            elif baseline_selector is not None:
                pool_texts = [d["content"] for d in pool_docs]
                pool_embs_list = [pool_embs[i] for i in range(n_pool)]
                sel_indices = baseline_selector.select_top_k(question, pool_texts, pool_embs_list, top_k)
            else:
                sel_indices = self._select_with_model(selector, q_emb, pool_embs, top_k)

            selected_texts = [pool_docs[i]["content"] for i in sel_indices if i < len(pool_docs)]

            try:
                pred = generator.generate(question, selected_texts)
            except Exception as e:
                logger.warning("Generator failed on question %d: %s", idx, e)
                pred = ""

            all_preds.append(pred)
            f1 = _relaxed_f1(pred, gold)
            all_f1s.append(f1)

        mean_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
        return {
            "selector": selector_name,
            "mean_f1": mean_f1,
            "n_samples": len(questions),
            "per_sample": [
                {"question": q, "gold": g, "pred": p, "f1": f}
                for q, g, p, f in zip(questions, golds, all_preds, all_f1s)
            ],
        }

    def compare(
        self,
        sras_model: Any,
        questions: List[str],
        golds: List[str],
        corpus_docs: List[Dict],
        doc_embs: torch.Tensor,
        q_embs: List[torch.Tensor],
        generator: Any,
        pool_size: int = 30,
        top_k: int = 3,
        seed: int = 42,
        baselines: Optional[Dict[str, Any]] = None,
        oracle_selector: Optional[Any] = None,
    ) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}

        logger.info("E2E comparison: running SRAS")
        results["sras"] = self._run_pipeline(
            "sras", sras_model, questions, golds, corpus_docs, doc_embs, q_embs,
            generator, pool_size, top_k, seed,
        )

        logger.info("E2E comparison: running random baseline")
        results["random"] = self._run_pipeline(
            "random", None, questions, golds, corpus_docs, doc_embs, q_embs,
            generator, pool_size, top_k, seed,
        )

        if baselines:
            for name, bselector in baselines.items():
                logger.info("E2E comparison: running baseline %s", name)
                results[name] = self._run_pipeline(
                    name, None, questions, golds, corpus_docs, doc_embs, q_embs,
                    generator, pool_size, top_k, seed, baseline_selector=bselector,
                )

        summary = []
        for name, result in results.items():
            summary.append({
                "selector": name,
                "mean_f1": result["mean_f1"],
                "n_samples": result["n_samples"],
                "gain_vs_random": result["mean_f1"] - results["random"]["mean_f1"],
            })
        summary.sort(key=lambda x: -x["mean_f1"])

        output = {"results": results, "summary": summary}
        out_path = os.path.join(self.output_dir, "e2e_comparison.json")
        save_json(output, out_path)
        logger.info("E2E comparison saved to %s", out_path)

        for entry in summary:
            logger.info(
                "  %s | mean_f1=%.4f | gain_vs_random=%+.4f",
                entry["selector"], entry["mean_f1"], entry["gain_vs_random"],
            )

        return output
