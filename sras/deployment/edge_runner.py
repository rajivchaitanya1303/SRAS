from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict, List, Optional

import torch

from sras.models.selector import CrossAttentionSelector, load_selector
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)

_PI_MARKERS = ("aarch64", "armv7l", "armv6l")


def is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    if any(m in machine for m in _PI_MARKERS):
        try:
            with open("/proc/cpuinfo") as f:
                return "raspberry pi" in f.read().lower()
        except OSError:
            return True
    return False


def get_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "is_raspberry_pi": is_raspberry_pi(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total / (1024 ** 3)
        info["ram_available_gb"] = vm.available / (1024 ** 3)
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return info


class EdgeRunner:
    def __init__(
        self,
        checkpoint_path: str,
        model_kwargs: Optional[Dict] = None,
        force_cpu: bool = True,
    ) -> None:
        self.device = torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device("cuda")
        self.model = load_selector(checkpoint_path, self.device, model_kwargs)
        self.model.eval()
        self._hw_info = get_hardware_info()
        logger.info("EdgeRunner initialized on %s | Pi=%s", self.device, self._hw_info["is_raspberry_pi"])

    def select(
        self,
        q_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        k: int = 3,
    ) -> List[int]:
        if doc_embs.shape[0] == 0:
            return []
        k = min(k, doc_embs.shape[0])
        q = q_emb.float().to(self.device)
        d = doc_embs.float().to(self.device)
        with torch.no_grad():
            scores = self.model(q, d)
        return torch.topk(scores, k).indices.tolist()

    def select_safe(
        self,
        q_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        k: int = 3,
    ) -> Dict:
        try:
            t0 = time.perf_counter()
            indices = self.select(q_emb, doc_embs, k)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {"indices": indices, "latency_ms": latency_ms, "error": None}
        except Exception as e:
            logger.error("EdgeRunner.select_safe failed: %s", e)
            return {"indices": [], "latency_ms": 0.0, "error": str(e)}

    def batch_select(
        self,
        q_embs: List[torch.Tensor],
        doc_embs: torch.Tensor,
        k: int = 3,
    ) -> List[List[int]]:
        results = []
        for q_emb in q_embs:
            results.append(self.select(q_emb, doc_embs, k))
        return results

    def warmup(self, doc_embs: torch.Tensor, n: int = 5) -> None:
        dummy_q = torch.randn(doc_embs.shape[1])
        for _ in range(n):
            self.select(dummy_q, doc_embs[:10] if doc_embs.shape[0] >= 10 else doc_embs)

    def hardware_info(self) -> Dict:
        return self._hw_info

    def model_info(self) -> Dict:
        return {
            "num_params": self.model.count_parameters(),
            "device": str(self.device),
            **self.model.get_init_kwargs(),
        }

    def quick_benchmark(
        self,
        q_embs: List[torch.Tensor],
        doc_embs: torch.Tensor,
        n_iters: int = 50,
    ) -> Dict:
        import numpy as np

        if not q_embs:
            raise ValueError("q_embs must not be empty")

        d = doc_embs.float().to(self.device)
        q = q_embs[0].float().to(self.device)

        self.warmup(d, n=5)

        latencies: List[float] = []
        for i in range(n_iters):
            qi = q_embs[i % len(q_embs)].float().to(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                self.model(qi, d)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        return {
            "n_docs": doc_embs.shape[0],
            "avg_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "queries_per_sec": 1000.0 / max(float(np.mean(latencies)), 1e-9),
            "hardware": self._hw_info,
            "model": self.model_info(),
        }
