from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from sras.models.selector import CrossAttentionSelector
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class _Transition:
    __slots__ = ("q_emb", "doc_embs", "actions", "reward")

    def __init__(
        self,
        q_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        actions: torch.Tensor,
        reward: float,
    ) -> None:
        self.q_emb = q_emb
        self.doc_embs = doc_embs
        self.actions = actions
        self.reward = reward


class PPOAgent:
    def __init__(
        self,
        model: CrossAttentionSelector,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        update_epochs: int = 4,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.grad_clip = grad_clip
        self._memory: List[_Transition] = []

    def clear_memory(self) -> None:
        self._memory.clear()

    def store(
        self,
        q_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        actions: torch.Tensor,
        reward: float,
    ) -> None:
        self._memory.append(
            _Transition(
                q_emb=q_emb.detach(),
                doc_embs=doc_embs.detach(),
                actions=actions.detach(),
                reward=float(reward),
            )
        )

    def store_batch(
        self,
        q_embs: List[torch.Tensor],
        doc_embs_list: List[torch.Tensor],
        actions_list: List[torch.Tensor],
        rewards: List[float],
    ) -> None:
        for q, d, a, r in zip(q_embs, doc_embs_list, actions_list, rewards):
            self.store(q, d, a, float(r) if not isinstance(r, float) else r)

    def _compute_loss(self, transition: _Transition) -> Tuple[torch.Tensor, torch.Tensor]:
        q_emb = transition.q_emb
        doc_embs = transition.doc_embs
        actions = transition.actions
        reward = transition.reward

        logits = self.model(q_emb, doc_embs)
        with torch.no_grad():
            old_logits = self.old_model(q_emb, doc_embs)

        probs = torch.softmax(logits, dim=-1)
        old_probs = torch.softmax(old_logits, dim=-1)

        probs = probs.clamp(min=1e-8)
        old_probs = old_probs.clamp(min=1e-8)

        valid_actions = actions[actions < probs.shape[0]]
        if valid_actions.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), torch.tensor(0.0, device=logits.device)

        sel_probs = probs[valid_actions]
        old_sel_probs = old_probs[valid_actions].detach()

        ratio = sel_probs / old_sel_probs
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_loss = -torch.min(ratio * reward, clipped * reward).mean()

        entropy = -(probs * torch.log(probs)).sum()

        return policy_loss, entropy

    def update(self) -> Dict[str, float]:
        if not self._memory:
            return {"policy_loss": 0.0, "entropy": 0.0}

        total_policy_loss = 0.0
        total_entropy = 0.0
        n = len(self._memory)

        for _ in range(self.update_epochs):
            epoch_policy_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
            epoch_entropy = torch.tensor(0.0, device=next(self.model.parameters()).device)

            for t in self._memory:
                pl, ent = self._compute_loss(t)
                epoch_policy_loss = epoch_policy_loss + pl
                epoch_entropy = epoch_entropy + ent

            total_loss = epoch_policy_loss / n - self.entropy_coef * (epoch_entropy / n)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_policy_loss += (epoch_policy_loss / n).item()
            total_entropy += (epoch_entropy / n).item()

        self.old_model.load_state_dict(self.model.state_dict())
        self.clear_memory()

        return {
            "policy_loss": total_policy_loss / self.update_epochs,
            "entropy": total_entropy / self.update_epochs,
        }

    def state_dict(self) -> Dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.old_model.load_state_dict(state["model"])
