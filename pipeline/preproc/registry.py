from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

PostprocessorFn = Callable[[dict[str, Any], dict[str, Any], Path], list[Path]]

REGISTRY: dict[str, PostprocessorFn] = {}


def register_postprocessor(kind: str):
    def decorator(fn: PostprocessorFn) -> PostprocessorFn:
        if kind in REGISTRY:
            raise ValueError(f"постпроцессор {kind!r} уже зарегистрирован")
        REGISTRY[kind] = fn
        return fn

    return decorator


def get_postprocessor(kind: str) -> PostprocessorFn:
    if kind not in REGISTRY:
        raise KeyError(
            f"неизвестный kind {kind!r}. Зарегистрировано: {sorted(REGISTRY)}"
        )
    return REGISTRY[kind]
