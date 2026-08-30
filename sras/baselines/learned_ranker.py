from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class _RankerMLP(nn.Module):
    def __init__(self, emb_dim: int = 384, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        n = d_emb.shape[0]
        q_expanded = q_emb.expand(n, -1)
        interaction = q_expanded * d_emb
        x = torch.cat([q_expanded, d_emb, interaction], dim=-1)
        return self.net(x).squeeze(-1)


class LearnedRanker:
    def __init__(
        self,
        emb_dim: int = 384,
        hidden_dim: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self._model = _RankerMLP(emb_dim, hidden_dim).to(self.device)
        self._trained = False

    def train(
        self,
        q_embs: List[torch.Tensor],
        d_embs_list: List[torch.Tensor],
        rewards_list: List[List[float]],
        epochs: int = 10,
        lr: float = 1e-4,
    ) -> Dict[str, List[float]]:
        if not q_embs:
            raise ValueError("No training data provided to LearnedRanker")

        optimizer = Adam(self._model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for epoch in range(1, epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            n = 0
            for q_emb, d_embs, rewards in zip(q_embs, d_embs_list, rewards_list):
                if not rewards:
                    continue
                q = q_emb.float().to(self.device)
                d = d_embs.float().to(self.device)
                r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                if r.shape[0] != d.shape[0]:
                    continue
                scores = self._model(q, d)
                loss = loss_fn(scores, r)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n += 1

            avg = epoch_loss / max(n, 1)
            history.append(avg)
            if epoch % 5 == 0:
                logger.info("LearnedRanker epoch %d | loss=%.4f", epoch, avg)

        self._trained = True
        return {"train_loss": history}

    def score_candidates(
        self,
        q_emb: torch.Tensor,
        d_embs: torch.Tensor,
    ) -> List[float]:
        self._model.eval()
        with torch.no_grad():
            scores = self._model(q_emb.float().to(self.device), d_embs.float().to(self.device))
        return [float(s.item()) for s in scores]

    def select_top_k(
        self,
        q_emb: torch.Tensor,
        d_embs: torch.Tensor,
        doc_ids: List[str],
        k: int,
    ) -> List[str]:
        if not doc_ids:
            return []
        scores = self.score_candidates(q_emb, d_embs)
        ranked = sorted(zip(scores, doc_ids), key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in ranked[:k]]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model_state_dict": self._model.state_dict()}, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Learned ranker checkpoint not found: {path}")
        state = torch.load(path, map_location=self.device)
        self._model.load_state_dict(state["model_state_dict"])
        self._trained = True
