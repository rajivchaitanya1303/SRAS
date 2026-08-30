from __future__ import annotations

import copy
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from sras.models.selector import CrossAttentionSelector, save_selector
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class QuantizedSelector:
    def __init__(self, quantized_model: nn.Module, original_kwargs: Dict) -> None:
        self._model = quantized_model
        self._kwargs = original_kwargs

    def __call__(self, q_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        return self._model(q_embedding.cpu(), doc_embeddings.cpu())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def get_init_kwargs(self) -> Dict:
        return self._kwargs


def quantize_selector(
    model: CrossAttentionSelector,
    dtype: str = "int8",
    output_path: Optional[str] = None,
) -> QuantizedSelector:
    if dtype not in ("int8", "float16"):
        raise ValueError(f"Unsupported quantization dtype: {dtype}. Use 'int8' or 'float16'.")

    cpu_model = copy.deepcopy(model).cpu().eval()

    if dtype == "int8":
        quantized = torch.quantization.quantize_dynamic(
            cpu_model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        logger.info("Applied dynamic INT8 quantization")
    else:
        quantized = cpu_model.half()
        logger.info("Applied FP16 quantization")

    wrapped = QuantizedSelector(quantized, model.get_init_kwargs())

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(
            {
                "quantized_state": quantized.state_dict(),
                "model_kwargs": model.get_init_kwargs(),
                "dtype": dtype,
            },
            output_path,
        )
        logger.info("Quantized model saved to %s", output_path)

    return wrapped


def load_quantized(path: str, dtype: str = "int8") -> QuantizedSelector:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Quantized model not found: {path}")
    state = torch.load(path, map_location="cpu")
    kwargs = state["model_kwargs"]
    model = CrossAttentionSelector(**kwargs).cpu().eval()
    if state.get("dtype", dtype) == "int8":
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    else:
        model = model.half()
    model.load_state_dict(state["quantized_state"])
    return QuantizedSelector(model, kwargs)


def measure_quantization_overhead(
    original: CrossAttentionSelector,
    quantized: QuantizedSelector,
    q_emb: torch.Tensor,
    doc_embs: torch.Tensor,
    n_iters: int = 100,
) -> Dict[str, float]:
    q_cpu = q_emb.cpu()
    d_cpu = doc_embs.cpu()

    original_cpu = copy.deepcopy(original).cpu().eval()
    for _ in range(5):
        with torch.no_grad():
            original_cpu(q_cpu, d_cpu)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            original_cpu(q_cpu, d_cpu)
    orig_time = (time.perf_counter() - t0) / n_iters * 1000.0

    for _ in range(5):
        with torch.no_grad():
            quantized(q_cpu, d_cpu)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            quantized(q_cpu, d_cpu)
    quant_time = (time.perf_counter() - t0) / n_iters * 1000.0

    return {
        "original_latency_ms": orig_time,
        "quantized_latency_ms": quant_time,
        "speedup": orig_time / max(quant_time, 1e-9),
        "original_params": original.count_parameters(),
        "quantized_params": quantized.count_parameters(),
    }
