from __future__ import annotations

import re
from typing import Any

_LETTERS_RE = re.compile(r"[a-zA-Zа-яА-Я]")


def _strip_description(desc: str) -> str:
    return (desc or "").strip()


def annotation_has_letters(description: str) -> bool:
    return bool(_LETTERS_RE.search(_strip_description(description)))


def filter_dep_annotation_dicts(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Оставить аннотации без букв в описании (ветка DEPD)."""
    out: list[dict[str, Any]] = []
    for ann in annotations:
        desc = _strip_description(str(ann.get("description", "")))
        if not desc:
            continue
        if _LETTERS_RE.search(desc):
            continue
        onset = float(ann["onset"])
        out.append({"onset": onset, "description": desc})
    return out


def filter_dep_annotations_from_raw(raw) -> list[dict[str, Any]]:
    """Из MNE Raw — список {onset, description} после фильтра DEPD."""
    rows: list[dict[str, Any]] = []
    for ann in raw.annotations:
        desc = _strip_description(ann["description"])
        if not desc or _LETTERS_RE.search(desc):
            continue
        rows.append({"onset": float(ann["onset"]), "description": desc})
    return rows
