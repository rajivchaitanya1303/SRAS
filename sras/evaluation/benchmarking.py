from __future__ import annotations

import os
import random
import time
import tracemalloc
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sras.config.schema import BenchmarkConfig, SRASConfig
from sras.data.corpus import CorpusStore
from sras.data.embeddings import EmbeddingStore, encode_query
from sras.deployment.profiler import DeploymentProfiler
from sras.models.selector import CrossAttentionSelector, load_selector
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger
from sras.utils.reproducibility import get_device, seed_everything

logger = get_logger(__name__)

_DUMMY_QUESTION = "What is the principal mechanism of action of transformer-based language models?"


class EdgeBenchmarker:
    def __init__(self, config: SRASConfig) -> None:
        self.config = config
        self.bcfg = config.benchmark
        self.mcfg = config.model

        self.device = get_device(self.bcfg.device)
        seed_everything(self.bcfg.seed)

        ensure_dir(self.bcfg.results_dir)

        self.corpus = CorpusStore(self.bcfg.corpus_metadata_path)
        self.embed_store = EmbeddingStore(
            config.data.doc_embeddings_path, self.corpus.doc_ids, self.device
        )

    def _get_model_file_size(self, path: str) -> float:
        if not os.path.exists(path):
            return 0.0
        return os.path.getsize(path) / (1024 ** 2)

    def _profile_model(
        self,
        model: CrossAttentionSelector,
        q_emb: torch.Tensor,
        pool_embs: torch.Tensor,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        for _ in range(self.bcfg.n_warmup):
            with torch.no_grad():
                model(q_emb, pool_embs)

        latencies: List[float] = []
        mem_deltas: List[float] = []

        tracemalloc.start()
        for _ in range(self.bcfg.n_iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            mem_before = tracemalloc.get_traced_memory()[0]
            t0 = time.perf_counter()

            with torch.no_grad():
                model(q_emb, pool_embs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            mem_after = tracemalloc.get_traced_memory()[0]

            latencies.append((t1 - t0) * 1000.0)
            mem_deltas.append((mem_after - mem_before) / (1024 ** 2))

        tracemalloc.stop()

        return {
            "avg_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "avg_delta_mem_mb": float(np.mean(mem_deltas)),
            "num_params": model.count_parameters(),
            "model_size_mb": self._get_model_file_size(checkpoint_path) if checkpoint_path else 0.0,
            "n_iterations": self.bcfg.n_iterations,
        }

    def benchmark_pool_size(
        self,
        model: CrossAttentionSelector,
        q_emb: torch.Tensor,
        pool_size: int,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        n = min(pool_size, self.embed_store.size())
        pool_embs = self.embed_store.get_batch_by_indices(list(range(n)))
        result = self._profile_model(model, q_emb, pool_embs, checkpoint_path)
        result["pool_size"] = n
        return result

    def run_deployment_profiling(
        self,
        model_registry: Dict[str, Any],
        q_embs: List[torch.Tensor],
        doc_embs: torch.Tensor,
    ) -> List[Dict]:
        profiler = DeploymentProfiler(self.config)
        return profiler.run_full(model_registry, q_embs, doc_embs)

    def run(self) -> List[Dict]:
        q_emb = encode_query(_DUMMY_QUESTION, self.config.data.embedding_model, self.device)

        all_results: List[Dict] = []
        loaded_models: Dict[str, Any] = {}

        for variant_name, checkpoint_path in self.bcfg.model_registry.items():
            logger.info("Benchmarking: %s", variant_name)
            if not os.path.exists(checkpoint_path):
                logger.warning("Checkpoint missing: %s -- skipping", checkpoint_path)
                continue

            try:
                model = load_selector(
                    checkpoint_path,
                    self.device,
                    model_kwargs={
                        "doc_emb_dim": self.mcfg.doc_emb_dim,
                        "hidden_dim": self.mcfg.hidden_dim,
                        "dropout": 0.0,
                        "use_layer_norm": self.mcfg.use_layer_norm,
                        "use_residual": self.mcfg.use_residual,
                    },
                )
                loaded_models[variant_name] = model
            except Exception as e:
                logger.error("Failed to load %s: %s", variant_name, e)
                continue

            for pool_size in self.bcfg.candidate_pool_sizes:
                result = self.benchmark_pool_size(model, q_emb, pool_size, checkpoint_path)
                result["variant"] = variant_name
                all_results.append(result)
                logger.info(
                    "%s | pool=%d | p50=%.2f ms | p95=%.2f ms | params=%d",
                    variant_name, pool_size,
                    result["p50_latency_ms"], result["p95_latency_ms"], result["num_params"],
                )

        for pool_size in self.bcfg.candidate_pool_sizes:
            n = min(pool_size, self.embed_store.size())
            indices = list(range(n))
            latencies = []
            tracemalloc.start()
            for _ in range(self.bcfg.n_iterations):
                t0 = time.perf_counter()
                random.sample(indices, min(3, n))
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
            tracemalloc.stop()
            all_results.append({
                "variant": "random",
                "pool_size": n,
                "avg_latency_ms": float(np.mean(latencies)),
                "std_latency_ms": float(np.std(latencies)),
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies)),
                "avg_delta_mem_mb": 0.0,
                "num_params": 0,
                "model_size_mb": 0.0,
                "n_iterations": self.bcfg.n_iterations,
            })

        if loaded_models:
            logger.info("Running deployment profiling with psutil metrics")
            try:
                max_pool = max(self.bcfg.candidate_pool_sizes)
                n_docs = min(max_pool, self.embed_store.size())
                doc_embs = self.embed_store.get_batch_by_indices(list(range(n_docs)))
                # Deployment profiler always runs on CPU (edge scenario); move models accordingly
                cpu_models = {k: v.cpu().eval() for k, v in loaded_models.items()}
                q_embs_list = [q_emb.cpu()]
                deploy_results = self.run_deployment_profiling(cpu_models, q_embs_list, doc_embs.cpu())
                out_deploy = os.path.join(self.bcfg.results_dir, "deployment_profile.json")
                save_json(deploy_results, out_deploy)
                logger.info("Deployment profile saved to %s", out_deploy)
            except Exception as e:
                logger.error("Deployment profiling failed: %s", e)

        out_path = os.path.join(self.bcfg.results_dir, "edge_benchmark.json")
        save_json(all_results, out_path)
        logger.info("Benchmark results saved to %s", out_path)
        return all_results
