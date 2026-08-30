from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from sras.config.schema import CompressionConfig
from sras.data.embeddings import EmbeddingStore
from sras.models.selector import CrossAttentionSelector, save_selector
from sras.utils.io import save_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DistillationTrainer:
    def __init__(
        self,
        teacher: CrossAttentionSelector,
        config: CompressionConfig,
        device: torch.device,
    ) -> None:
        self._teacher = copy.deepcopy(teacher).to(device).eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        self._config = config
        self.device = device

        teacher_kwargs = teacher.get_init_kwargs()
        student_kwargs = dict(teacher_kwargs)
        student_kwargs["hidden_dim"] = config.student_hidden_dim
        self._student = CrossAttentionSelector(**student_kwargs).to(device)
        logger.info(
            "DistillationTrainer | teacher_hidden=%d | student_hidden=%d",
            teacher_kwargs["hidden_dim"],
            config.student_hidden_dim,
        )

    @property
    def student(self) -> CrossAttentionSelector:
        return self._student

    def _distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        rewards: Optional[torch.Tensor],
    ) -> torch.Tensor:
        T = self._config.distillation_temperature
        alpha = self._config.distillation_alpha

        s_soft = F.log_softmax(student_logits / T, dim=-1)
        t_soft = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T ** 2)

        if rewards is not None and rewards.numel() > 0:
            task_loss = F.mse_loss(
                torch.softmax(student_logits, dim=-1),
                F.normalize(rewards.clamp(min=0), p=1, dim=-1),
            )
        else:
            task_loss = torch.tensor(0.0, device=self.device)

        return alpha * kl_loss + (1.0 - alpha) * task_loss

    def train(
        self,
        q_embs: List[torch.Tensor],
        doc_embs_list: List[torch.Tensor],
        rewards_list: Optional[List[List[float]]] = None,
    ) -> Dict[str, List[float]]:
        if not q_embs:
            raise ValueError("No training data provided to DistillationTrainer")

        optimizer = Adam(
            self._student.parameters(),
            lr=self._config.distillation_lr,
            weight_decay=1e-4,
        )
        history: List[float] = []

        for epoch in range(1, self._config.distillation_epochs + 1):
            self._student.train()
            epoch_loss = 0.0
            n = 0

            indices = list(range(len(q_embs)))
            for idx in indices:
                q = q_embs[idx].float().to(self.device)
                d = doc_embs_list[idx].float().to(self.device)

                if q.dim() == 1:
                    q = q.unsqueeze(0)

                with torch.no_grad():
                    teacher_logits = self._teacher(q, d)

                student_logits = self._student(q, d)

                rewards_tensor = None
                if rewards_list is not None and idx < len(rewards_list):
                    r = rewards_list[idx]
                    if len(r) == d.shape[0]:
                        rewards_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)

                loss = self._distill_loss(student_logits, teacher_logits, rewards_tensor)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._student.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n += 1

            avg = epoch_loss / max(n, 1)
            history.append(avg)
            if epoch % 5 == 0:
                logger.info("Distillation epoch %d/%d | loss=%.4f", epoch, self._config.distillation_epochs, avg)

        return {"distillation_loss": history}

    def save_student(self, path: Optional[str] = None) -> str:
        if path is None:
            os.makedirs(self._config.output_dir, exist_ok=True)
            path = os.path.join(self._config.output_dir, "sras_selector_distilled.pt")
        save_selector(self._student, path, {"distillation": True})
        logger.info("Student model saved to %s", path)
        return path
