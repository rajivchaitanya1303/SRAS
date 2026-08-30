from __future__ import annotations

import os
import platform
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from sras.config.schema import DeploymentConfig, SRASConfig
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_ram_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return 0.0


def _get_cpu_time_s() -> float:
    try:
        import psutil
        t = psutil.Process(os.getpid()).cpu_times()
        return t.user + t.system
    except ImportError:
        return 0.0


class _CPUPercentMonitor(threading.Thread):
    """Samples psutil process-level CPU utilization (%) at a fixed interval.

    Reports direct CPU utilization, complementing the existing RAM/throughput/
    thermal measurements. ``psutil.Process.cpu_percent()`` reports the percentage of
    one core's worth of time consumed since the previous call, matching what
    tools like `top`/`htop` show per-process.
    """

    def __init__(self, interval_s: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.interval = interval_s
        self.readings: List[float] = []
        self._stop_event = threading.Event()
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(interval=None)  # prime the internal counter
        except ImportError:
            pass

    def run(self) -> None:
        if self._proc is None:
            return
        while not self._stop_event.is_set():
            try:
                self.readings.append(self._proc.cpu_percent(interval=None))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_event.set()


def _read_thermal_celsius() -> Optional[float]:
    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    vcgencmd_path = "/usr/bin/vcgencmd"
    try:
        if os.path.exists(vcgencmd_path):
            import subprocess
            out = subprocess.check_output([vcgencmd_path, "measure_temp"], text=True)
            return float(out.strip().replace("temp=", "").replace("'C", ""))
        if os.path.exists(thermal_path):
            with open(thermal_path) as f:
                return int(f.read().strip()) / 1000.0
    except Exception:
        pass
    return None


class _ThermalMonitor(threading.Thread):
    def __init__(self, interval_s: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.interval = interval_s
        self.readings: List[float] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            t = _read_thermal_celsius()
            if t is not None:
                self.readings.append(t)
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_event.set()


class DeploymentProfiler:
    def __init__(self, config: SRASConfig) -> None:
        self.dcfg = config.benchmark.deployment
        ensure_dir(self.dcfg.results_dir)

    def _run_latency(
        self,
        fn: Callable,
        n_warmup: int,
        n_iters: int,
    ) -> Dict[str, float]:
        for _ in range(n_warmup):
            fn()
        latencies: List[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            fn()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return {
            "avg_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "std_ms": float(np.std(latencies)),
        }

    def _run_throughput(
        self,
        model: Any,
        q_embs: List[torch.Tensor],
        doc_embs: torch.Tensor,
        batch_size: int,
    ) -> float:
        n = len(q_embs)
        if n == 0:
            return 0.0
        t0 = time.perf_counter()
        for i in range(0, n, batch_size):
            batch = q_embs[i:i + batch_size]
            for q in batch:
                with torch.no_grad():
                    model(q.cpu(), doc_embs.cpu())
        elapsed = time.perf_counter() - t0
        return n / elapsed

    def _run_energy(self, fn: Callable, n_iters: int) -> Dict[str, float]:
        cpu_before = _get_cpu_time_s()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn()
        elapsed = time.perf_counter() - t0
        cpu_after = _get_cpu_time_s()
        cpu_time = cpu_after - cpu_before
        return {
            "total_cpu_time_s": cpu_time,
            "cpu_time_per_query_ms": cpu_time / max(n_iters, 1) * 1000.0,
            "wall_time_s": elapsed,
        }

    def profile_model(
        self,
        model: Any,
        q_embs: List[torch.Tensor],
        doc_embs: torch.Tensor,
        label: str,
        pool_size: int,
    ) -> Dict:
        if not q_embs:
            raise ValueError("q_embs must not be empty")

        d = doc_embs[:pool_size].cpu()
        q = q_embs[0].cpu()

        def _single():
            with torch.no_grad():
                model(q, d)

        result: Dict = {
            "label": label,
            "pool_size": pool_size,
            "platform": platform.platform(),
            "cpu": platform.processor(),
        }

        result.update(self._run_latency(_single, self.dcfg.n_warmup, self.dcfg.n_iterations))

        if self.dcfg.measure_ram:
            ram_before = _get_ram_mb()
            _single()
            ram_after = _get_ram_mb()
            result["ram_delta_mb"] = ram_after - ram_before
            result["ram_total_mb"] = ram_after

        if self.dcfg.measure_throughput:
            for bs in self.dcfg.throughput_batch_sizes:
                n_q = min(bs * 10, len(q_embs))
                tput = self._run_throughput(model, q_embs[:n_q], d, bs)
                result[f"throughput_bs{bs}_qps"] = tput

        if self.dcfg.measure_energy:
            result.update(self._run_energy(_single, self.dcfg.n_iterations))

        if self.dcfg.measure_thermal:
            monitor = _ThermalMonitor(self.dcfg.thermal_interval_s)
            monitor.start()
            for _ in range(self.dcfg.n_iterations):
                _single()
            monitor.stop()
            monitor.join(timeout=2.0)
            if monitor.readings:
                result["thermal_avg_c"] = float(np.mean(monitor.readings))
                result["thermal_max_c"] = float(np.max(monitor.readings))
                result["thermal_min_c"] = float(np.min(monitor.readings))
            else:
                result["thermal_avg_c"] = None
                result["thermal_max_c"] = None
                result["thermal_min_c"] = None

        if getattr(self.dcfg, "measure_cpu_percent", True):
            cpu_monitor = _CPUPercentMonitor(interval_s=0.2)
            cpu_monitor.start()
            for _ in range(self.dcfg.n_iterations):
                _single()
            cpu_monitor.stop()
            cpu_monitor.join(timeout=2.0)
            if cpu_monitor.readings:
                result["cpu_percent_avg"] = float(np.mean(cpu_monitor.readings))
                result["cpu_percent_max"] = float(np.max(cpu_monitor.readings))
            else:
                result["cpu_percent_avg"] = None
                result["cpu_percent_max"] = None

        return result

    def run_full(
        self,
        model_registry: Dict[str, Any],
        q_embs: List[torch.Tensor],
        doc_embs: torch.Tensor,
    ) -> List[Dict]:
        all_results: List[Dict] = []

        for label, model in model_registry.items():
            logger.info("Profiling deployment: %s", label)
            for pool_size in self.dcfg.candidate_pool_sizes:
                try:
                    result = self.profile_model(model, q_embs, doc_embs, label, pool_size)
                    all_results.append(result)
                    logger.info(
                        "%s | pool=%d | p50=%.2f ms | tput_bs1=%.1f qps",
                        label, pool_size,
                        result.get("p50_ms", 0),
                        result.get("throughput_bs1_qps", 0),
                    )
                except Exception as e:
                    logger.error("Profiling failed for %s pool=%d: %s", label, pool_size, e)

        out_path = os.path.join(self.dcfg.results_dir, "deployment_profile.json")
        save_json(all_results, out_path)
        logger.info("Deployment profile saved to %s", out_path)
        return all_results
