from __future__ import annotations

import copy
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sras.compression.pruning import get_sparsity, prune_selector
from sras.compression.quantization import QuantizedSelector, measure_quantization_overhead, quantize_selector
from sras.config.schema import CompressionConfig, SRASConfig
from sras.evaluation.metrics import MetricsComputer
from sras.models.selector import CrossAttentionSelector, load_selector
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _latency_ms(fn: Callable, n_iters: int = 50) -> float:
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1000.0


def _model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2) if os.path.exists(path) else 0.0


class CompressionEvaluator:
    def __init__(self, config: SRASConfig) -> None:
        self.config = config
        self.ccfg = config.benchmark.compression
        self.device = torch.device("cpu")
        ensure_dir(self.ccfg.output_dir)

    def _profile_model(
        self,
        model: Any,
        q_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        label: str,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        q = q_emb.cpu()
        d = doc_embs.cpu()

        def _run():
            with torch.no_grad():
                model(q, d)

        latency = _latency_ms(_run)
        if isinstance(model, CrossAttentionSelector):
            n_params = model.count_parameters()
            sparsity = get_sparsity(model)
        elif isinstance(model, QuantizedSelector):
            n_params = model.count_parameters()
            sparsity = 0.0
        else:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            sparsity = 0.0

        size_mb = _model_size_mb(checkpoint_path) if checkpoint_path else 0.0

        return {
            "label": label,
            "latency_ms": latency,
            "num_params": n_params,
            "sparsity": sparsity,
            "model_size_mb": size_mb,
        }

    def run(
        self,
        base_model: CrossAttentionSelector,
        q_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        base_checkpoint_path: Optional[str] = None,
        eval_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
    ) -> List[Dict]:
        results: List[Dict] = []
        base = copy.deepcopy(base_model).cpu().eval()

        entry = self._profile_model(base, q_emb, doc_embs, "original", base_checkpoint_path)
        if eval_fn:
            entry.update(eval_fn(base))
        results.append(entry)

        if self.ccfg.use_quantization:
            qpath = os.path.join(self.ccfg.output_dir, "sras_selector_int8.pt")
            quantized = quantize_selector(base_model, dtype=self.ccfg.quantization_dtype, output_path=qpath)
            entry = self._profile_model(quantized, q_emb, doc_embs, f"quantized_{self.ccfg.quantization_dtype}", qpath)
            if eval_fn:
                entry.update(eval_fn(quantized))
            results.append(entry)
            logger.info("Quantization evaluated")

        if self.ccfg.use_pruning:
            for amount in [self.ccfg.pruning_amount, self.ccfg.pruning_amount * 1.5]:
                amt = min(float(amount), 0.9)
                pruned = prune_selector(
                    copy.deepcopy(base_model),
                    amount=amt,
                    method=self.ccfg.pruning_method,
                )
                pruned = pruned.cpu().eval()
                ppath = os.path.join(self.ccfg.output_dir, f"sras_selector_pruned_{int(amt*100)}.pt")
                torch.save({"model_state_dict": pruned.state_dict(), "model_kwargs": pruned.get_init_kwargs()}, ppath)
                entry = self._profile_model(pruned, q_emb, doc_embs, f"pruned_{int(amt*100)}pct", ppath)
                entry["sparsity"] = get_sparsity(pruned)
                if eval_fn:
                    entry.update(eval_fn(pruned))
                results.append(entry)
            logger.info("Pruning evaluated")

        out_path = os.path.join(self.ccfg.output_dir, "compression_results.json")
        save_json(results, out_path)
        logger.info("Compression results saved to %s", out_path)
        return results
