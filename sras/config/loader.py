from __future__ import annotations

import copy
import dataclasses
import json
import os
import typing
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

from sras.config.schema import SRASConfig

T = TypeVar("T")


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _from_dict(cls: Type[T], d: Any) -> T:
    if not dataclasses.is_dataclass(cls) or not isinstance(d, dict):
        return d
    hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        hint = hints.get(f.name)
        origin = getattr(hint, "__origin__", None)
        if hint is not None and dataclasses.is_dataclass(hint) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(hint, val)
        elif origin is not None:
            kwargs[f.name] = val
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SRASConfig:
    base = dataclasses.asdict(SRASConfig())

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                loaded = yaml.safe_load(f) or {}
            elif path.suffix == ".json":
                loaded = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        base = _deep_update(base, loaded)

    if overrides:
        base = _deep_update(base, overrides)

    return _from_dict(SRASConfig, base)


def save_config(config: SRASConfig, output_path: str) -> None:
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)
    d = dataclasses.asdict(config)
    with open(path, "w", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            yaml.safe_dump(d, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(d, f, indent=2)
