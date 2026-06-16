"""
Загрузка сохранённых сегментов (без исходного .EEG): манифест, *_meta.json, `.npz`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import mne
import numpy as np


def load_export_manifest(export_root: str | Path) -> dict[str, Any]:
    root = Path(export_root).expanduser().resolve()
    path = root / "export_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Нет {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _segment_entry(man: dict[str, Any], segment_index: int) -> dict[str, Any]:
    for s in man.get("segments") or []:
        if int(s["index"]) == int(segment_index):
            return s
    raise IndexError(f"Нет сегмента с index={segment_index}")


def load_segment_meta(export_root: str | Path, segment_index: int) -> dict[str, Any]:
    root = Path(export_root).expanduser().resolve()
    man = load_export_manifest(export_root)
    s = _segment_entry(man, segment_index)
    meta_path = root / s["meta"]
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def segment_dir(export_root: str | Path, segment_index: int) -> Path:
    """Корень экспорта (сегменты лежат плоско: `*.npz`, `*_meta.json`)."""
    root = Path(export_root).expanduser().resolve()
    man = load_export_manifest(root)
    _segment_entry(man, segment_index)
    return root


def load_segment_npz(export_root: str | Path, segment_index: int) -> dict[str, Any]:
    """Словарь с массивами: data, sfreq, t_start_sec, t_end_sec, ch_names."""
    root = Path(export_root).expanduser().resolve()
    man = load_export_manifest(export_root)
    s = _segment_entry(man, segment_index)
    rel = s.get("npz")
    if not rel:
        raise KeyError(
            f"Сегмент {segment_index}: в манифесте нет поля «npz» "
            f"(ожидается export_manifest version 2, storage npz_bipolar)."
        )
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_segment_raw(export_root: str | Path, segment_index: int) -> mne.io.BaseRaw:
    """Один сегмент как MNE RawArray из bipolar `.npz`."""
    z = load_segment_npz(export_root, segment_index)
    data = np.asarray(z["data"], dtype=np.float64)
    sfreq = float(z["sfreq"])
    chs = [str(x) for x in z["ch_names"].tolist()]
    info = mne.create_info(chs, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def iter_segment_indices(export_root: str | Path) -> Iterator[int]:
    man = load_export_manifest(export_root)
    for s in sorted(man.get("segments") or [], key=lambda x: int(x["index"])):
        yield int(s["index"])


def load_all_metas(export_root: str | Path) -> list[dict[str, Any]]:
    """Список meta по порядку сегментов."""
    out: list[dict[str, Any]] = []
    for idx in iter_segment_indices(export_root):
        out.append(load_segment_meta(export_root, idx))
    return out
