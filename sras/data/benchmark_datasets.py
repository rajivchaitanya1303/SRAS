from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

from sras.config.schema import BenchmarkDatasetConfig
from sras.utils.io import ensure_dir, load_json, save_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_hf_dataset(dataset_name: str, config_name: Optional[str], split: str, cache_dir: str):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace 'datasets' is required: pip install datasets") from e
    logger.info("Loading HF dataset: %s / %s / %s", dataset_name, config_name, split)
    return load_dataset(dataset_name, config_name, split=split, cache_dir=cache_dir, trust_remote_code=True)


def _sample(items: List, n: int, seed: int = 42) -> List:
    rng = random.Random(seed)
    if n >= len(items):
        return list(items)
    return rng.sample(list(items), n)


class TriviaQALoader:
    DATASET = "trivia_qa"
    CONFIG = "rc"

    def __init__(self, config: BenchmarkDatasetConfig) -> None:
        self.config = config
        ensure_dir(config.cache_dir)

    def load(self, seed: int = 42) -> List[Dict]:
        cache_path = os.path.join(self.config.cache_dir, f"triviaqa_{self.config.triviaqa_split}_{self.config.triviaqa_subset_n}.json")
        if os.path.exists(cache_path):
            return load_json(cache_path)

        ds = _load_hf_dataset(self.DATASET, self.CONFIG, self.config.triviaqa_split, self.config.cache_dir)
        pairs = []
        for item in ds:
            q = item.get("question", "").strip()
            answers = item.get("answer", {})
            if isinstance(answers, dict):
                a = answers.get("value", "") or (answers.get("aliases", [""])[0])
            else:
                a = str(answers)
            if q and a:
                pairs.append({"question": q, "answer": a.strip(), "dataset": "triviaqa"})

        sampled = _sample(pairs, self.config.triviaqa_subset_n, seed)
        save_json(sampled, cache_path)
        logger.info("TriviaQA loaded: %d pairs cached to %s", len(sampled), cache_path)
        return sampled


class NaturalQuestionsLoader:
    DATASET = "natural_questions"
    CONFIG = "default"

    def __init__(self, config: BenchmarkDatasetConfig) -> None:
        self.config = config
        ensure_dir(config.cache_dir)

    def _extract_short_answer(self, item: Dict) -> str:
        annotations = item.get("annotations", {})
        if isinstance(annotations, dict):
            short = annotations.get("short_answers", [])
            if isinstance(short, list) and short:
                first = short[0]
                if isinstance(first, dict):
                    tokens = first.get("text", [])
                    if tokens:
                        return " ".join(tokens) if isinstance(tokens, list) else str(tokens)
        return ""

    def load(self, seed: int = 42) -> List[Dict]:
        cache_path = os.path.join(self.config.cache_dir, f"nq_{self.config.nq_split}_{self.config.nq_subset_n}.json")
        if os.path.exists(cache_path):
            return load_json(cache_path)

        ds = _load_hf_dataset(self.DATASET, self.CONFIG, self.config.nq_split, self.config.cache_dir)
        pairs = []
        for item in ds:
            q = item.get("question", {})
            if isinstance(q, dict):
                q = q.get("text", "")
            q = str(q).strip()
            a = self._extract_short_answer(item)
            if q and a:
                pairs.append({"question": q, "answer": a.strip(), "dataset": "natural_questions"})

        sampled = _sample(pairs, self.config.nq_subset_n, seed)
        save_json(sampled, cache_path)
        logger.info("NaturalQuestions loaded: %d pairs cached to %s", len(sampled), cache_path)
        return sampled


class MLQALoader:
    DATASET = "mlqa"

    _LANG_CONFIGS = {
        "en": "mlqa.en.en",
        "de": "mlqa.de.de",
        "es": "mlqa.es.es",
        "ar": "mlqa.ar.ar",
        "hi": "mlqa.hi.hi",
        "zh": "mlqa.zh.zh",
        "vi": "mlqa.vi.vi",
    }

    def __init__(self, config: BenchmarkDatasetConfig) -> None:
        self.config = config
        ensure_dir(config.cache_dir)

    def load_language(self, lang: str, seed: int = 42) -> List[Dict]:
        if lang not in self._LANG_CONFIGS:
            raise ValueError(f"Unsupported MLQA language: {lang}. Choose from {list(self._LANG_CONFIGS)}")
        cfg = self._LANG_CONFIGS[lang]
        cache_path = os.path.join(self.config.cache_dir, f"mlqa_{lang}_{self.config.mlqa_subset_n}.json")
        if os.path.exists(cache_path):
            return load_json(cache_path)

        ds = _load_hf_dataset(self.DATASET, cfg, "test", self.config.cache_dir)
        pairs = []
        for item in ds:
            q = item.get("question", "").strip()
            answers = item.get("answers", {})
            if isinstance(answers, dict):
                texts = answers.get("text", [])
                a = texts[0].strip() if texts else ""
            else:
                a = ""
            if q and a:
                pairs.append({"question": q, "answer": a, "language": lang, "dataset": "mlqa"})

        sampled = _sample(pairs, self.config.mlqa_subset_n, seed)
        save_json(sampled, cache_path)
        logger.info("MLQA [%s] loaded: %d pairs cached to %s", lang, len(sampled), cache_path)
        return sampled

    def load_all(self, seed: int = 42) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {}
        for lang in self.config.mlqa_languages:
            try:
                result[lang] = self.load_language(lang, seed)
            except Exception as e:
                logger.error("Failed to load MLQA [%s]: %s", lang, e)
        return result


def load_benchmark_datasets(
    config: BenchmarkDatasetConfig,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    datasets: Dict[str, List[Dict]] = {}

    if config.use_triviaqa:
        try:
            datasets["triviaqa"] = TriviaQALoader(config).load(seed)
        except Exception as e:
            logger.error("TriviaQA load failed: %s", e)

    if config.use_natural_questions:
        try:
            datasets["natural_questions"] = NaturalQuestionsLoader(config).load(seed)
        except Exception as e:
            logger.error("NaturalQuestions load failed: %s", e)

    if config.use_mlqa:
        try:
            mlqa_data = MLQALoader(config).load_all(seed)
            for lang, pairs in mlqa_data.items():
                datasets[f"mlqa_{lang}"] = pairs
        except Exception as e:
            logger.error("MLQA load failed: %s", e)

    return datasets
