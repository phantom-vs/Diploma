"""
Загрузка сохранённых сегментов (без исходного .EEG): манифест, meta.json, MNE Raw из FIF.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import mne


def load_export_manifest(export_root: str | Path) -> dict[str, Any]:
    root = Path(export_root).expanduser().resolve()
    path = root / "export_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Нет {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_segment_meta(export_root: str | Path, segment_index: int) -> dict[str, Any]:
    man = load_export_manifest(export_root)
    segs = man.get("segments") or []
    for s in segs:
        if int(s["index"]) == int(segment_index):
            meta_path = Path(export_root).expanduser().resolve() / s["meta"]
            with meta_path.open(encoding="utf-8") as f:
                return json.load(f)
    raise IndexError(f"Нет сегмента с index={segment_index}")


def segment_dir(export_root: str | Path, segment_index: int) -> Path:
    man = load_export_manifest(export_root)
    for s in man.get("segments") or []:
        if int(s["index"]) == int(segment_index):
            return Path(export_root).expanduser().resolve() / s["dir"]
    raise IndexError(f"Нет сегмента с index={segment_index}")


def load_segment_raw(export_root: str | Path, segment_index: int) -> mne.io.BaseRaw:
    """Загрузить только один сегмент из segment_raw.fif (preload=True по умолчанию MNE)."""
    d = segment_dir(export_root, segment_index)
    fif = d / "segment_raw.fif"
    if not fif.is_file():
        raise FileNotFoundError(fif)
    return mne.io.read_raw_fif(str(fif), preload=True, verbose=False)


def iter_segment_indices(export_root: str | Path) -> Iterator[int]:
    man = load_export_manifest(export_root)
    for s in sorted(man.get("segments") or [], key=lambda x: int(x["index"])):
        yield int(s["index"])


def load_all_metas(export_root: str | Path) -> list[dict[str, Any]]:
    """Список meta.json по порядку сегментов."""
    out: list[dict[str, Any]] = []
    for idx in iter_segment_indices(export_root):
        out.append(load_segment_meta(export_root, idx))
    return out
