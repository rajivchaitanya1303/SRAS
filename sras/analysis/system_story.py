from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SystemStoryLogger:
    def __init__(self, output_dir: str = "results/system_story") -> None:
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self._claims: List[Dict] = []

    def _record(self, claim_type: str, data: Dict) -> None:
        self._claims.append({"claim_type": claim_type, **data})

    def log_edge_friendliness(
        self,
        model_params: int,
        model_size_mb: float,
        p50_latency_ms: float,
        p99_latency_ms: float,
        ram_mb: float,
        device: str,
        is_raspberry_pi: bool = False,
    ) -> None:
        data = {
            "model_params": model_params,
            "model_size_mb": model_size_mb,
            "p50_latency_ms": p50_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "ram_mb": ram_mb,
            "device": device,
            "is_raspberry_pi": is_raspberry_pi,
            "qualifies_edge": p50_latency_ms < 100.0 and ram_mb < 512.0,
        }
        logger.info(
            "EdgeFriendliness | params=%d | size=%.2f MB | p50=%.2f ms | RAM=%.1f MB | Pi=%s",
            model_params, model_size_mb, p50_latency_ms, ram_mb, is_raspberry_pi,
        )
        self._record("edge_friendliness", data)

    def log_sparse_reward_behavior(
        self,
        reward_history: List[float],
        n_zero_reward_steps: int,
        n_total_steps: int,
        final_avg_reward: float,
    ) -> None:
        sparsity_ratio = n_zero_reward_steps / max(n_total_steps, 1)
        data = {
            "n_zero_reward_steps": n_zero_reward_steps,
            "n_total_steps": n_total_steps,
            "sparsity_ratio": sparsity_ratio,
            "final_avg_reward": final_avg_reward,
            "reward_mean": sum(reward_history) / len(reward_history) if reward_history else 0.0,
            "reward_min": min(reward_history) if reward_history else 0.0,
            "reward_max": max(reward_history) if reward_history else 0.0,
        }
        logger.info(
            "SparseReward | sparsity=%.2f | final_avg_reward=%.4f",
            sparsity_ratio, final_avg_reward,
        )
        self._record("sparse_reward_behavior", data)

    def log_rag_bottleneck_analysis(
        self,
        selector_f1: float,
        oracle_f1: float,
        random_f1: float,
        generator_ceiling_f1: float,
        selector_failure_rate: float,
        generator_failure_rate: float,
    ) -> None:
        selection_gap = oracle_f1 - selector_f1
        generation_gap = generator_ceiling_f1 - selector_f1
        bottleneck = "selection" if selector_failure_rate > generator_failure_rate else "generation"
        data = {
            "selector_f1": selector_f1,
            "oracle_f1": oracle_f1,
            "random_f1": random_f1,
            "generator_ceiling_f1": generator_ceiling_f1,
            "selector_failure_rate": selector_failure_rate,
            "generator_failure_rate": generator_failure_rate,
            "selection_gap": selection_gap,
            "generation_gap": generation_gap,
            "identified_bottleneck": bottleneck,
            "selector_vs_random_gain": selector_f1 - random_f1,
        }
        logger.info(
            "RAGBottleneck | selector_f1=%.4f | oracle=%.4f | bottleneck=%s | sel_fail=%.3f | gen_fail=%.3f",
            selector_f1, oracle_f1, bottleneck, selector_failure_rate, generator_failure_rate,
        )
        self._record("rag_bottleneck_analysis", data)

    def log_curriculum_impact(
        self,
        with_curriculum_reward: float,
        without_curriculum_reward: float,
        curriculum_schedule: str,
        start_docs: int,
        end_docs: int,
    ) -> None:
        gain = with_curriculum_reward - without_curriculum_reward
        data = {
            "with_curriculum_reward": with_curriculum_reward,
            "without_curriculum_reward": without_curriculum_reward,
            "curriculum_schedule": curriculum_schedule,
            "start_docs": start_docs,
            "end_docs": end_docs,
            "reward_gain": gain,
            "curriculum_helps": gain > 0.0,
        }
        logger.info(
            "CurriculumImpact | w=%.4f | wo=%.4f | gain=%.4f | schedule=%s",
            with_curriculum_reward, without_curriculum_reward, gain, curriculum_schedule,
        )
        self._record("curriculum_impact", data)

    def log_compression_tradeoff(
        self,
        original_f1: float,
        compressed_f1: float,
        original_latency_ms: float,
        compressed_latency_ms: float,
        compression_method: str,
        speedup: float,
        f1_drop: float,
    ) -> None:
        data = {
            "original_f1": original_f1,
            "compressed_f1": compressed_f1,
            "original_latency_ms": original_latency_ms,
            "compressed_latency_ms": compressed_latency_ms,
            "compression_method": compression_method,
            "speedup": speedup,
            "f1_drop": f1_drop,
            "acceptable_tradeoff": f1_drop < 0.05 and speedup > 1.2,
        }
        logger.info(
            "CompressionTradeoff | method=%s | speedup=%.2fx | f1_drop=%.4f | acceptable=%s",
            compression_method, speedup, f1_drop, data["acceptable_tradeoff"],
        )
        self._record("compression_tradeoff", data)

    def log_multilingual_generalization(
        self,
        results_by_language: Dict[str, float],
        base_language: str = "en",
    ) -> None:
        base_f1 = results_by_language.get(base_language, 0.0)
        cross_lingual = {lang: f1 for lang, f1 in results_by_language.items() if lang != base_language}
        avg_cross = sum(cross_lingual.values()) / len(cross_lingual) if cross_lingual else 0.0
        data = {
            "results_by_language": results_by_language,
            "base_language": base_language,
            "base_f1": base_f1,
            "avg_cross_lingual_f1": avg_cross,
            "cross_lingual_drop": base_f1 - avg_cross,
        }
        logger.info(
            "MultilingualGen | base_f1=%.4f | avg_cross=%.4f | drop=%.4f",
            base_f1, avg_cross, base_f1 - avg_cross,
        )
        self._record("multilingual_generalization", data)

    def log_selector_contribution(
        self,
        sras_f1: float,
        bm25_f1: float,
        dense_f1: float,
        hybrid_f1: float,
        random_f1: float,
        oracle_f1: float,
    ) -> None:
        data = {
            "sras_f1": sras_f1,
            "bm25_f1": bm25_f1,
            "dense_f1": dense_f1,
            "hybrid_f1": hybrid_f1,
            "random_f1": random_f1,
            "oracle_f1": oracle_f1,
            "gain_over_bm25": sras_f1 - bm25_f1,
            "gain_over_dense": sras_f1 - dense_f1,
            "gain_over_hybrid": sras_f1 - hybrid_f1,
            "gain_over_random": sras_f1 - random_f1,
            "oracle_gap": oracle_f1 - sras_f1,
            "sras_ranks_first": sras_f1 == max(sras_f1, bm25_f1, dense_f1, hybrid_f1),
        }
        logger.info(
            "SelectorContribution | sras=%.4f | bm25=%.4f | dense=%.4f | hybrid=%.4f | oracle=%.4f",
            sras_f1, bm25_f1, dense_f1, hybrid_f1, oracle_f1,
        )
        self._record("selector_contribution", data)

    def save(self, filename: str = "system_story.json") -> str:
        out_path = os.path.join(self.output_dir, filename)
        save_json(self._claims, out_path)
        logger.info("System story saved to %s (%d claims)", out_path, len(self._claims))
        return out_path

    def summary(self) -> Dict[str, Any]:
        return {
            "n_claims": len(self._claims),
            "claim_types": list({c["claim_type"] for c in self._claims}),
        }
