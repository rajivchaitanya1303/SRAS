from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from sras.utils.io import load_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


class QADataset:
    def __init__(self, qa_pairs_path: str) -> None:
        raw = load_json(qa_pairs_path)
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"QA dataset at {qa_pairs_path} must be a non-empty list.")
        self._pairs: List[Dict] = []
        for i, item in enumerate(raw):
            if "question" not in item:
                raise ValueError(f"QA item {i} missing 'question' field.")
            if "answer" not in item:
                raise ValueError(f"QA item {i} missing 'answer' field.")
            self._pairs.append({
                "question": str(item["question"]).strip(),
                "answer": str(item["answer"]).strip(),
                "doc_id": item.get("doc_id", None),
                "topic": item.get("topic", "unknown"),
            })
        logger.info("QA dataset loaded: %d pairs", len(self._pairs))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict:
        return self._pairs[idx]

    def shuffled(self, seed: Optional[int] = None) -> List[Dict]:
        pairs = list(self._pairs)
        rng = random.Random(seed)
        rng.shuffle(pairs)
        return pairs

    def get_batches(self, batch_size: int, shuffle: bool = True, seed: Optional[int] = None) -> List[List[Dict]]:
        pairs = self.shuffled(seed) if shuffle else list(self._pairs)
        return [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

    def get_topics(self) -> List[str]:
        return sorted(set(p["topic"] for p in self._pairs))

    def filter_by_topic(self, topic: str) -> List[Dict]:
        return [p for p in self._pairs if p["topic"] == topic]


class RewardDataset:
    def __init__(self, reward_matrix_path: str, min_candidates: int = 2) -> None:
        raw = load_json(reward_matrix_path)
        if not isinstance(raw, dict) or len(raw) == 0:
            raise ValueError(f"Reward matrix at {reward_matrix_path} must be a non-empty dict.")
        self._data: Dict[str, List[Dict]] = {}
        skipped = 0
        for question, candidates in raw.items():
            if not isinstance(candidates, list):
                skipped += 1
                continue
            valid = [
                c for c in candidates
                if isinstance(c, dict) and "candidate_doc_id" in c and "reward" in c
            ]
            if len(valid) < min_candidates:
                skipped += 1
                continue
            self._data[question] = valid
        if skipped > 0:
            logger.warning("RewardDataset: skipped %d entries with insufficient candidates", skipped)
        if len(self._data) == 0:
            raise ValueError("RewardDataset is empty after filtering.")
        logger.info("RewardDataset loaded: %d questions", len(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data.items())

    def items(self):
        return self._data.items()

    def questions(self) -> List[str]:
        return list(self._data.keys())

    def get_candidates(self, question: str) -> List[Dict]:
        if question not in self._data:
            raise KeyError(f"Question not found in reward dataset: {question!r}")
        return self._data[question]

    def get_best_doc_id(self, question: str) -> str:
        candidates = self.get_candidates(question)
        return max(candidates, key=lambda c: c["reward"])["candidate_doc_id"]

    def get_doc_ids(self, question: str) -> List[str]:
        return [c["candidate_doc_id"] for c in self.get_candidates(question)]

    def get_rewards(self, question: str) -> List[float]:
        return [c["reward"] for c in self.get_candidates(question)]


class SquadDataset:
    """Legacy SQuAD dataset loader (question + answer only, no context).
    Used for backward compatibility. Prefer SquadEvalDataset for new runs."""

    def __init__(self, squad_path: str) -> None:
        raw = load_json(squad_path)
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"SQuAD dataset at {squad_path} must be a non-empty list.")
        self._pairs: List[Dict] = []
        for i, item in enumerate(raw):
            if "question" not in item or "answer" not in item:
                raise ValueError(f"SQuAD item {i} missing 'question' or 'answer'.")
            self._pairs.append({
                "question": str(item["question"]).strip(),
                "answer": str(item["answer"]).strip(),
                "id": item.get("id", str(i)),
            })
        logger.info("SQuAD dataset loaded: %d pairs", len(self._pairs))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict:
        return self._pairs[idx]

    def questions(self) -> List[str]:
        return [p["question"] for p in self._pairs]

    def answers(self) -> List[str]:
        return [p["answer"] for p in self._pairs]


class SquadEvalDataset:
    """Context-aware SQuAD evaluation dataset.

    Each item has:
      question       - the question string
      answer         - short answer span (gold)
      context_doc_id - doc ID of the correct Wikipedia passage in the
                       SQuAD context corpus (data/squad_contexts.json)

    Generated by setup_squad_eval.py.
    """

    def __init__(self, squad_eval_path: str) -> None:
        raw = load_json(squad_eval_path)
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(
                f"SQuAD eval dataset at {squad_eval_path} must be a non-empty list.\n"
                "Run:  python setup_squad_eval.py  to generate it."
            )
        self._pairs: List[Dict] = []
        for i, item in enumerate(raw):
            if "question" not in item or "answer" not in item:
                raise ValueError(f"SquadEvalDataset item {i} missing 'question' or 'answer'.")
            self._pairs.append({
                "question":       str(item["question"]).strip(),
                "answer":         str(item["answer"]).strip(),
                "context_doc_id": item.get("context_doc_id", ""),
            })
        logger.info("SquadEvalDataset loaded: %d pairs", len(self._pairs))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict:
        return self._pairs[idx]

    def questions(self) -> List[str]:
        return [p["question"] for p in self._pairs]

    def answers(self) -> List[str]:
        return [p["answer"] for p in self._pairs]

    def context_doc_ids(self) -> List[str]:
        return [p["context_doc_id"] for p in self._pairs]


class ExternalQADataset:
    def __init__(self, path: str, question_key: str = "question", answer_key: str = "answer") -> None:
        raw = load_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"External dataset at {path} must be a list.")
        self._pairs: List[Dict] = []
        for i, item in enumerate(raw):
            if question_key not in item or answer_key not in item:
                logger.warning("External dataset item %d missing keys, skipping.", i)
                continue
            self._pairs.append({
                "question": str(item[question_key]).strip(),
                "answer": str(item[answer_key]).strip(),
            })
        logger.info("External dataset loaded from %s: %d pairs", path, len(self._pairs))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict:
        return self._pairs[idx]
