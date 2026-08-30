from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from sras.config.schema import SRASConfig
from sras.data.datasets import RewardDataset
from sras.data.embeddings import EmbeddingStore, encode_query
from sras.models.selector import CrossAttentionSelector, load_selector, save_selector
from sras.utils.io import ensure_dir, save_json
from sras.utils.logging_utils import get_logger
from sras.utils.reproducibility import seed_everything

logger = get_logger(__name__)


class SupervisedTrainer:
    def __init__(self, config: SRASConfig) -> None:
        self.config = config
        self.tcfg = config.training
        self.dcfg = config.data
        self.mcfg = config.model

        from sras.utils.reproducibility import get_device
        self.device = get_device(self.tcfg.device)
        seed_everything(self.tcfg.seed)

        ensure_dir(self.tcfg.checkpoint_dir)
        ensure_dir(self.tcfg.log_dir)

    def _build_model(self) -> CrossAttentionSelector:
        return CrossAttentionSelector(
            doc_emb_dim=self.mcfg.doc_emb_dim,
            hidden_dim=self.mcfg.hidden_dim,
            dropout=self.mcfg.dropout,
            use_layer_norm=self.mcfg.use_layer_norm,
            use_residual=self.mcfg.use_residual,
        ).to(self.device)

    def train(self) -> CrossAttentionSelector:
        reward_dataset = RewardDataset(self.dcfg.reward_matrix_path)

        from sras.data.corpus import CorpusStore
        corpus = CorpusStore(self.dcfg.corpus_metadata_path, max_docs=self.dcfg.max_corpus_docs)

        embed_store = EmbeddingStore(
            self.dcfg.doc_embeddings_path,
            corpus.doc_ids,
            self.device,
        )

        model = self._build_model()

        # If contrastive pre-training was run, initialise projection weights from it
        csp_ckpt = self.tcfg.contrastive_checkpoint
        if os.path.exists(csp_ckpt):
            try:
                csp_model = load_selector(csp_ckpt, self.device, model_kwargs=model.get_init_kwargs())
                csp_sd = csp_model.state_dict()
                cur_sd = model.state_dict()
                proj_keys = [k for k in csp_sd if any(
                    k.startswith(p) for p in ("q_proj", "d_proj", "q_norm", "d_norm")
                )]
                for k in proj_keys:
                    if k in cur_sd and cur_sd[k].shape == csp_sd[k].shape:
                        cur_sd[k] = csp_sd[k]
                model.load_state_dict(cur_sd)
                logger.info("Supervised: initialised projections from CSP checkpoint %s", csp_ckpt)
            except Exception as e:
                logger.debug("Supervised: CSP checkpoint load skipped (%s)", e)

        optimizer = Adam(
            model.parameters(),
            lr=self.tcfg.supervised_lr,
            weight_decay=self.tcfg.supervised_weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        best_loss = float("inf")
        log_history: List[Dict] = []

        for epoch in range(1, self.tcfg.supervised_epochs + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0

            for question, candidates in tqdm(
                reward_dataset.items(),
                desc=f"Supervised epoch {epoch}/{self.tcfg.supervised_epochs}",
                leave=False,
            ):
                if len(candidates) < 2:
                    continue

                try:
                    q_emb = encode_query(question, self.dcfg.embedding_model, self.device)

                    doc_ids = [c["candidate_doc_id"] for c in candidates]
                    doc_embs = embed_store.get_batch_by_ids(doc_ids)

                    rewards = [c["reward"] for c in candidates]
                    best_idx = int(np.argmax(rewards))
                    target = torch.tensor([best_idx], dtype=torch.long, device=self.device)

                    logits = model(q_emb, doc_embs).unsqueeze(0)
                    loss = loss_fn(logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    total_samples += 1

                except (KeyError, RuntimeError, ValueError) as e:
                    logger.warning("Supervised training: skipping sample (%s)", e)
                    continue

            if total_samples == 0:
                logger.error("No valid training samples in epoch %d", epoch)
                continue

            avg_loss = total_loss / total_samples
            log_history.append({"epoch": epoch, "avg_loss": avg_loss, "samples": total_samples})
            logger.info("Supervised epoch %d | loss=%.4f | samples=%d", epoch, avg_loss, total_samples)

            if self.tcfg.save_best and avg_loss < best_loss:
                best_loss = avg_loss
                save_selector(
                    model,
                    self.tcfg.supervised_checkpoint,
                    {"epoch": epoch, "avg_loss": avg_loss},
                )
                logger.info("Best model saved (loss=%.4f)", best_loss)

        if not self.tcfg.save_best:
            save_selector(model, self.tcfg.supervised_checkpoint)

        log_path = os.path.join(self.tcfg.log_dir, "supervised_training_log.json")
        save_json(log_history, log_path)
        logger.info("Supervised training complete. Log saved to %s", log_path)

        return model
