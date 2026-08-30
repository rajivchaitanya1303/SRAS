"""
SRAS Selector Models
====================
Contains two model classes:

CrossAttentionSelector
    Original model from the conference paper.  Query-document cross-attention
    fusion with a single scoring head.

AdaptiveCrossAttentionSelector  (journal extension: ADB)
    Extends CrossAttentionSelector with an Adaptive Document Budget (ADB) head
    that learns to predict the optimal number of documents to retrieve per query.

    Motivation
    ----------
    A fixed top-k forces the model to always retrieve the same number of
    documents regardless of query difficulty.  Simple factoid queries (e.g.
    "Who wrote Hamlet?") need only one precise document; complex causal queries
    (e.g. "Why did the Roman Empire fall?") benefit from broader coverage.
    ADB adds a lightweight budget-prediction head that takes the query
    representation and outputs a soft budget scalar b ∈ [0,1], mapping to an
    integer k = min_k + round(b × (max_k − min_k)).

    Training
    --------
    The budget head has no explicit supervision signal; it is trained
    implicitly through PPO.  Selecting fewer documents reduces noise for easy
    queries (reward increases); selecting more documents increases coverage for
    hard queries (reward increases).  The PPO reward gradient propagates
    through the budget head naturally.

    Backward compatibility
    ----------------------
    AdaptiveCrossAttentionSelector inherits CrossAttentionSelector and adds
    only the ``budget_head`` parameter group.  Checkpoints are saved with
    ``model_kwargs["use_adaptive_budget"]=True``, so ``load_selector`` routes
    them to the correct class automatically.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionSelector(nn.Module):
    def __init__(
        self,
        doc_emb_dim: int = 384,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.doc_emb_dim = doc_emb_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        self.q_proj = nn.Linear(doc_emb_dim, hidden_dim)
        self.d_proj = nn.Linear(doc_emb_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

        if use_layer_norm:
            self.q_norm = nn.LayerNorm(hidden_dim)
            self.d_norm = nn.LayerNorm(hidden_dim)

        if use_residual and doc_emb_dim != hidden_dim:
            self.res_proj = nn.Linear(doc_emb_dim, hidden_dim)
        else:
            self.res_proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.q_proj, self.d_proj, self.attn]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        if self.res_proj is not None:
            nn.init.xavier_uniform_(self.res_proj.weight)
            nn.init.zeros_(self.res_proj.bias)

    def forward(
        self,
        q_embedding: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if q_embedding.dim() == 1:
            q_embedding = q_embedding.unsqueeze(0)
        if doc_embeddings.dim() == 1:
            doc_embeddings = doc_embeddings.unsqueeze(0)

        if q_embedding.shape[-1] != self.doc_emb_dim:
            raise ValueError(
                f"Expected q_embedding dim {self.doc_emb_dim}, got {q_embedding.shape[-1]}"
            )
        if doc_embeddings.shape[-1] != self.doc_emb_dim:
            raise ValueError(
                f"Expected doc_embeddings dim {self.doc_emb_dim}, got {doc_embeddings.shape[-1]}"
            )

        q = self.q_proj(q_embedding)
        d = self.d_proj(doc_embeddings)

        if self.use_layer_norm:
            q = self.q_norm(q)
            d = self.d_norm(d)

        q = self.dropout(q)
        d = self.dropout(d)

        n_docs = d.shape[0]
        q_expanded = q.expand(n_docs, -1)

        fused = torch.tanh(q_expanded + d)

        if self.use_residual:
            if self.res_proj is not None:
                res = self.res_proj(doc_embeddings)
            else:
                res = doc_embeddings if doc_embeddings.shape[-1] == self.hidden_dim else d
            fused = fused + res

        scores = self.attn(fused).squeeze(-1)
        return scores

    def get_init_kwargs(self) -> Dict:
        return {
            "doc_emb_dim": self.doc_emb_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout.p,
            "use_layer_norm": self.use_layer_norm,
            "use_residual": self.use_residual,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_quantizable_modules(self) -> Dict[str, nn.Module]:
        return {
            "q_proj": self.q_proj,
            "d_proj": self.d_proj,
            "attn": self.attn,
        }


class AdaptiveCrossAttentionSelector(CrossAttentionSelector):
    """
    CrossAttentionSelector extended with an Adaptive Document Budget (ADB) head.

    The budget head is a two-layer MLP that takes the projected query
    representation and outputs a scalar b ∈ (0, 1).  During a forward pass the
    predicted budget is stored in ``self.last_budget`` so that calling code
    can use it for adaptive top-k selection without re-running the model.

    Parameters
    ----------
    min_k : int
        Minimum number of documents to select (inclusive). Default 1.
    max_k : int
        Maximum number of documents to select (inclusive). Default 5.
    All other parameters are identical to CrossAttentionSelector.
    """

    def __init__(
        self,
        doc_emb_dim: int = 384,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        min_k: int = 1,
        max_k: int = 5,
    ) -> None:
        super().__init__(
            doc_emb_dim=doc_emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
        )
        self.min_k = min_k
        self.max_k = max_k

        # Budget head: query_repr → learned temperature τ ∈ [τ_min, τ_max]
        # The temperature scales the raw scores before the PPO softmax:
        #   high τ (uncertain query)  → flat distribution  → more docs contribute
        #   low  τ (confident query)  → sharp distribution → fewer docs dominate
        # Because τ is part of the forward computation, gradients from the PPO
        # policy loss naturally flow through τ → budget_head parameters.
        self.tau_min: float = 0.3
        self.tau_max: float = 1.5
        self.budget_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),     # output b ∈ (0, 1)
        )
        # Initialise to output ~0.5 → τ ≈ (τ_min + τ_max) / 2
        nn.init.xavier_uniform_(self.budget_head[0].weight)
        nn.init.zeros_(self.budget_head[0].bias)
        nn.init.xavier_uniform_(self.budget_head[2].weight)
        nn.init.constant_(self.budget_head[2].bias, 0.0)

        self.last_budget: float = 0.5  # fractional budget from last forward

    # ── Override forward: learned temperature scaling (fully trainable) ───────
    def forward(
        self,
        q_embedding: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns temperature-scaled scores.

        The budget head predicts τ ∈ [τ_min, τ_max] from the query representation.
        Raw scores are divided by τ before being returned.  Because τ is part of
        the differentiable forward pass, PPO policy gradients naturally update the
        budget_head weights; no separate loss term required.

        The last fractional budget b (before temperature conversion) is stored in
        ``self.last_budget`` for adaptive top-k selection at inference time.
        """
        if q_embedding.dim() == 1:
            q_embedding = q_embedding.unsqueeze(0)

        # ── Shared query projection ─────────────────────────────────────────
        q = self.q_proj(q_embedding)   # [1, hidden]
        if self.use_layer_norm:
            q = self.q_norm(q)

        # ── Budget / temperature prediction (in-graph, fully differentiable) ─
        b = self.budget_head(q)                        # [1, 1] in (0, 1)
        tau = self.tau_min + b * (self.tau_max - self.tau_min)   # [1, 1]
        self.last_budget = float(b.detach().squeeze().item())    # store for inference

        # ── Document projection ─────────────────────────────────────────────
        d = self.d_proj(doc_embeddings)
        if self.use_layer_norm:
            d = self.d_norm(d)

        q_drop = self.dropout(q)
        d_drop = self.dropout(d)

        n_docs     = d_drop.shape[0]
        q_expanded = q_drop.expand(n_docs, -1)
        fused      = torch.tanh(q_expanded + d_drop)

        if self.use_residual:
            if self.res_proj is not None:
                res = self.res_proj(doc_embeddings)
            else:
                res = doc_embeddings \
                    if doc_embeddings.shape[-1] == self.hidden_dim else d_drop
            fused = fused + res

        scores_raw = self.attn(fused).squeeze(-1)          # [n_docs]
        scores     = scores_raw / tau.squeeze()            # temperature-scaled
        return scores

    def predicted_k(self) -> int:
        """Return the adaptive k predicted during the last forward pass."""
        k = self.min_k + round(self.last_budget * (self.max_k - self.min_k))
        return max(self.min_k, min(self.max_k, k))

    def get_init_kwargs(self) -> Dict:
        d = super().get_init_kwargs()
        d.update({
            "use_adaptive_budget": True,
            "min_k": self.min_k,
            "max_k": self.max_k,
        })
        return d

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_selector(
    checkpoint_path: str,
    device: torch.device,
    model_kwargs: Optional[Dict] = None,
) -> CrossAttentionSelector:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Selector checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        kwargs = state.get("model_kwargs", model_kwargs or {})
    else:
        kwargs = model_kwargs or {}

    # Route to AdaptiveCrossAttentionSelector if the checkpoint requests it
    use_adb = kwargs.pop("use_adaptive_budget", False)
    if use_adb:
        model = AdaptiveCrossAttentionSelector(**kwargs)
    else:
        model = CrossAttentionSelector(**kwargs)

    state_dict = state["model_state_dict"] \
        if isinstance(state, dict) and "model_state_dict" in state \
        else state
    # Allow partial load when budget_head keys are missing (e.g. loading a
    # standard checkpoint into AdaptiveCrossAttentionSelector for transfer)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        import warnings
        warnings.warn(
            f"load_selector: missing keys (may be expected for ADB init): {missing}"
        )

    model.to(device)
    model.eval()
    return model


def save_selector(
    model: CrossAttentionSelector,
    checkpoint_path: str,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    payload: Dict = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.get_init_kwargs(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
