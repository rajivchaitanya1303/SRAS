from __future__ import annotations

from typing import List, Optional

import torch

from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Prompt template for instruction-tuned models (Flan-T5, etc.).
# Uses a clear instruction format that significantly outperforms the raw
# "question: ... context: ..." prefix used by vanilla T5-base.
_PROMPT_TEMPLATE = (
    "Answer the question based on the context below. "
    "Give a short, direct answer.\n\n"
    "Question: {question}\n"
    "Context: {context}\n\n"
    "Answer:"
)

# Fallback template for non-instruction-tuned models (e.g. t5-base).
_LEGACY_PROMPT_TEMPLATE = "question: {question} context: {context}"

_INSTRUCTION_MODEL_PREFIXES = ("flan-t5", "flan_t5", "google/flan")


def _is_instruction_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(prefix in name_lower for prefix in _INSTRUCTION_MODEL_PREFIXES)


def _build_prompt(model_name: str, question: str, context: str) -> str:
    if _is_instruction_model(model_name):
        return _PROMPT_TEMPLATE.format(question=question, context=context)
    return _LEGACY_PROMPT_TEMPLATE.format(question=question, context=context)


class GeneratorInterface:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[torch.device] = None,
        max_input_len: int = 512,
        max_output_len: int = 128,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        logger.info("Loading generator model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        logger.info("Generator model loaded on %s", self.device)

    def generate(self, question: str, context_docs: List[str]) -> str:
        self._load()
        context = " ".join(doc.strip() for doc in context_docs if doc.strip())
        prompt = _build_prompt(self.model_name, question, context)
        input_ids = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_len,
        ).input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.max_output_len,
                num_beams=4,
                early_stopping=True,
            )

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def generate_batch(self, questions: List[str], contexts: List[List[str]]) -> List[str]:
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have equal length")
        if not questions:
            return []

        self._load()
        prompts = [
            _build_prompt(self.model_name, q, " ".join(c))
            for q, c in zip(questions, contexts)
        ]
        encodings = self._tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_input_len,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                encodings.input_ids,
                attention_mask=encodings.attention_mask,
                max_new_tokens=self.max_output_len,
                num_beams=4,
                early_stopping=True,
            )

        return [
            self._tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output_ids
        ]

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
