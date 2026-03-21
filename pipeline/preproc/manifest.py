from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def normalize_recording_tags(raw: Any) -> Any:
    """tags из YAML: словарь или список; нет поля — пустой словарь."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return deepcopy(raw)
    if isinstance(raw, list):
        return deepcopy(raw)
    raise SystemExit(
        "recording 'tags' must be a YAML mapping or list, or omitted (got "
        f"{type(raw).__name__})"
    )


def build_run_manifest(
    ctx: dict[str, Any],
    params: dict[str, Any],
    *,
    kind: str,
    artifacts: list[str],
    **extra: Any,
) -> dict[str, Any]:
    """Тело manifest.yaml; extra — доп. поля под конкретный kind."""
    tags = ctx.get("tags")
    if tags is None:
        tags = {}
    return {
        "recording_id": ctx["recording_id"],
        "dataset": ctx["dataset"],
        "subject_id": ctx["subject_id"],
        "session_date": ctx["session_date"],
        "source_eeg_path": ctx["source_eeg_path"],
        "tags": tags,
        "preproc_id": params.get("_preproc_id"),
        "kind": kind,
        "params": {k: v for k, v in params.items() if not str(k).startswith("_")},
        "artifacts": artifacts,
        **extra,
    }


def write_manifest_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
