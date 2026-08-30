from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from sras.models.selector import CrossAttentionSelector
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.weight.numel()
            zeros += (module.weight == 0).sum().item()
    if total == 0:
        return 0.0
    return zeros / total


def _get_linear_params(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    params = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            params.append((module, "weight"))
    return params


def prune_selector(
    model: CrossAttentionSelector,
    amount: float = 0.3,
    method: str = "magnitude",
    make_permanent: bool = True,
) -> CrossAttentionSelector:
    if not (0.0 < amount < 1.0):
        raise ValueError(f"pruning amount must be in (0, 1), got {amount}")
    if method not in ("magnitude", "random"):
        raise ValueError(f"pruning method must be 'magnitude' or 'random', got {method}")

    pruned = copy.deepcopy(model)
    params_to_prune = _get_linear_params(pruned)

    if not params_to_prune:
        logger.warning("No linear layers found for pruning")
        return pruned

    if method == "magnitude":
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    else:
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=amount,
        )

    if make_permanent:
        for module, param_name in params_to_prune:
            prune.remove(module, param_name)

    sparsity = get_sparsity(pruned)
    logger.info("Pruning complete | method=%s | target=%.2f | actual_sparsity=%.4f", method, amount, sparsity)
    return pruned


def iterative_prune(
    model: CrossAttentionSelector,
    target_amount: float,
    n_steps: int = 5,
    method: str = "magnitude",
) -> Tuple[CrossAttentionSelector, List[float]]:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    step_amount = 1.0 - (1.0 - target_amount) ** (1.0 / n_steps)
    current = copy.deepcopy(model)
    sparsity_history: List[float] = []

    for step in range(n_steps):
        current = prune_selector(current, amount=step_amount, method=method, make_permanent=True)
        s = get_sparsity(current)
        sparsity_history.append(s)
        logger.info("Iterative pruning step %d/%d | sparsity=%.4f", step + 1, n_steps, s)

    return current, sparsity_history


def prune_sensitivity_analysis(
    model: CrossAttentionSelector,
    amounts: Optional[List[float]] = None,
    method: str = "magnitude",
) -> List[Dict]:
    if amounts is None:
        amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []
    for amount in amounts:
        pruned = prune_selector(copy.deepcopy(model), amount=amount, method=method)
        sparsity = get_sparsity(pruned)
        n_params = sum(p.numel() for p in pruned.parameters() if p.requires_grad)
        results.append({
            "target_amount": amount,
            "actual_sparsity": sparsity,
            "num_params": n_params,
        })
    return results
