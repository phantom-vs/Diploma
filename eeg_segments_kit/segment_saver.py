"""
Нарезка записи ЭЭГ на сегменты (~по умолчанию 5 мин) и сохранение без зависимости от исходного файла.

Конфиг — обычный dict Python; при необходимости сохраняйте/читайте через save_config_json / load_config_json.
"""
from __future__ import annotations

import json
import uuid
import warnings
from pathlib import Path
from typing import Any

import mne
import numpy as np

from preprocessor import (
    apply_label_pipeline,
    depd_intervals_from_annotations,
    filter_dep_like_annotation_dicts,
    filter_intervals_intersecting_range,
    find_annotated_region_bounds_sec,
    global_intervals_from_markers,
    intervals_for_segment,
    preprocess_annotation_descriptions,
)


def load_config_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open(encoding="utf-8") as f:
        return json.load(f)


def save_config_json(cfg: dict[str, Any], path: str | Path) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def read_raw_nihon(eeg_path: str | Path, *, preload: bool) -> mne.io.BaseRaw:
    eeg_path = Path(eeg_path).expanduser().resolve()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*[Dd]ecode log as utf-8.*",
            category=RuntimeWarning,
        )
        return mne.io.read_raw_nihon(str(eeg_path), preload=preload, verbose=False)


def _raw_annotations_as_dicts(raw: mne.io.BaseRaw) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ann in raw.annotations:
        rows.append(
            {
                "onset": float(ann["onset"]),
                "description": str(ann["description"]),
            }
        )
    return rows


def _ensure_recording_id(cfg: dict[str, Any]) -> str:
    rid = str(cfg.get("recording_id") or "").strip()
    if rid:
        return rid
    return uuid.uuid4().hex[:12]


BIPOLAR_ANODES = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "Fp1",
    "Fp2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
]
BIPOLAR_CATHODES = [
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "O1",
    "O2",
    "Cz",
    "Pz",
]


def bipolar_channel_names() -> list[str]:
    return [f"{a}-{c}" for a, c in zip(BIPOLAR_ANODES, BIPOLAR_CATHODES)]


def cropped_raw_to_bipolar_float32(seg_raw: mne.io.BaseRaw) -> tuple[np.ndarray, list[str]]:
    """Обрезанный монополярный Raw → (data float32 (18, n_times), имена биполярных каналов)."""
    names = bipolar_channel_names()
    data_chunk = seg_raw.get_data()
    info = seg_raw.info.copy()
    tmp_raw = mne.io.RawArray(data_chunk, info, verbose=False)
    raw_bip = mne.set_bipolar_reference(
        tmp_raw,
        anode=BIPOLAR_ANODES,
        cathode=BIPOLAR_CATHODES,
        ch_name=names,
        drop_refs=True,
        copy=True,
    )
    raw_bip.pick(names)
    seg_data = raw_bip.get_data().astype(np.float32)
    return seg_data, names


def export_recording_to_segments(
    eeg_path: str | Path,
    out_dir: str | Path,
    cfg: dict[str, Any],
) -> Path:
    """
    Главная точка входа: прочитать ЭЭГ, (опционально) обрезать по меткам региона,
    нарезать на сегменты, сохранить сжатый `.npz` (биполяр 18 каналов)
    + `*_meta.json` на сегмент + `export_manifest.json`.

    cfg (основные поля):
      - segment_duration_sec: float (по умолчанию 300)
      - label_pipeline: list[str] — имена шагов из preprocessor.LABEL_STEP_REGISTRY
      - depd_mode: "by_channel" | "global" | "none"
      - depd_global_markers: {"start_markers": [...], "end_markers": [...]} для depd_mode=global
      - annotated_region: null | объект с полями:
          start_markers, end_markers — точное совпадение description (после strip и label_pipeline);
          required (bool) — если true и начало не найдено — ошибка;
          require_end_marker (bool) — если true и после начала нет метки конца — ошибка.
        При заданных границах нарезка и окна только внутри [начало, конец]; для by_channel
        интервалы DEPD вне этого окна отбрасываются из meta.
      - export_only_if_dep_intervals: bool (по умолчанию false) — если true, при отсутствии
        интервалов после парсинга и фильтра окна выбросить ValueError и не писать сегменты.
        Для depd_mode=none не применяется.
      - recording_id, dataset, subject_id, session_date, tags — для манифеста
    """
    eeg_path = Path(eeg_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eeg_path.is_file():
        raise FileNotFoundError(eeg_path)

    segment_duration_sec = float(cfg.get("segment_duration_sec", 300.0))
    label_pipeline = cfg.get("label_pipeline") or []
    depd_mode = str(cfg.get("depd_mode", "by_channel"))
    depd_global_cfg = cfg.get("depd_global_markers") or {}
    region_cfg = cfg.get("annotated_region")

    recording_id = _ensure_recording_id(cfg)

    raw = read_raw_nihon(eeg_path, preload=False)
    try:
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        duration_sec = n_times / sfreq
        # n_times/sfreq иногда чуть выше последней метки времени MNE → crop(tmax=...) падает
        t_max_raw = float(raw.times[-1])
        stem = eeg_path.stem

        raw_anns = _raw_annotations_as_dicts(raw)
        piped = preprocess_annotation_descriptions(raw_anns, label_pipeline)

        t_crop_lo: float = 0.0
        t_crop_hi: float = duration_sec
        region_window_active = False
        if region_cfg:
            s_markers = list(region_cfg.get("start_markers") or [])
            e_markers = list(region_cfg.get("end_markers") or [])
            required = bool(region_cfg.get("required", False))
            require_end_marker = bool(region_cfg.get("require_end_marker", False))
            if s_markers and e_markers:
                ts, te, end_found = find_annotated_region_bounds_sec(
                    piped,
                    s_markers,
                    e_markers,
                    total_duration_sec=duration_sec,
                )
                if ts is None:
                    if required:
                        raise ValueError(
                            "annotated_region: не найдена ни одна метка начала разметки "
                            f"из {s_markers!r}"
                        )
                elif require_end_marker and not end_found:
                    raise ValueError(
                        "annotated_region: после метки начала не найдена метка конца разметки "
                        f"из {e_markers!r} (require_end_marker=true)"
                    )
                else:
                    if ts is not None and te is not None:
                        t_crop_lo = max(0.0, float(ts))
                        t_crop_hi = min(duration_sec, float(te))
                        region_window_active = True

        t_crop_hi = min(t_crop_hi, t_max_raw)
        t_crop_lo = min(max(0.0, t_crop_lo), t_max_raw)

        intervals_by_channel: list[dict[str, Any]] = []
        intervals_global: list[dict[str, Any]] = []

        if depd_mode == "by_channel":
            dep_like = filter_dep_like_annotation_dicts(piped)
            intervals_by_channel = depd_intervals_from_annotations(dep_like)
        elif depd_mode == "global":
            sm = list(depd_global_cfg.get("start_markers") or [])
            em = list(depd_global_cfg.get("end_markers") or [])
            if not sm or not em:
                raise ValueError("depd_mode=global требует depd_global_markers.start_markers и end_markers")
            intervals_global = global_intervals_from_markers(piped, sm, em)
        elif depd_mode == "none":
            pass
        else:
            raise ValueError(f"Неизвестный depd_mode: {depd_mode!r}")

        if region_window_active:
            intervals_by_channel = filter_intervals_intersecting_range(
                intervals_by_channel, t_crop_lo, t_crop_hi
            )
            intervals_global = filter_intervals_intersecting_range(
                intervals_global, t_crop_lo, t_crop_hi
            )

        if bool(cfg.get("export_only_if_dep_intervals", False)):
            if depd_mode == "by_channel" and not intervals_by_channel:
                raise ValueError(
                    "export_only_if_dep_intervals: нет интервалов DEP по каналам "
                    "(после парсинга и фильтра окна разметки); сегменты не записываются."
                )
            if depd_mode == "global" and not intervals_global:
                raise ValueError(
                    "export_only_if_dep_intervals: нет глобальных интервалов по маркерам; "
                    "сегменты не записываются."
                )

        seg_len = segment_duration_sec
        segments_meta: list[dict[str, Any]] = []

        t = t_crop_lo
        seg_index = 0
        while t < t_crop_hi - 1e-9:
            t1 = min(t + seg_len, t_crop_hi, t_max_raw)
            if t1 <= t:
                break
            seg_tag = f"{stem}_seg{seg_index:03d}"
            npz_rel = f"{seg_tag}.npz"
            meta_rel = f"{seg_tag}_meta.json"
            npz_abs = out_dir / npz_rel
            meta_abs = out_dir / meta_rel

            seg_raw = raw.copy().crop(tmin=t, tmax=t1).load_data()
            try:
                seg_data, seg_ch_names = cropped_raw_to_bipolar_float32(seg_raw)

                np.savez_compressed(
                    npz_abs,
                    data=seg_data,
                    sfreq=sfreq,
                    t_start_sec=float(t),
                    t_end_sec=float(t1),
                    ch_names=np.array(seg_ch_names, dtype=object),
                )

                seg_intervals_ch = intervals_for_segment(intervals_by_channel, t, t1)
                seg_intervals_gl = intervals_for_segment(intervals_global, t, t1)

                meta = {
                    "version": 1,
                    "segment_index": seg_index,
                    "segment_stem": seg_tag,
                    "npz_file": npz_rel,
                    "montage": "bipolar_18",
                    "tmin_global_sec": t,
                    "tmax_global_sec": t1,
                    "duration_sec": t1 - t,
                    "sfreq": sfreq,
                    "n_channels": len(seg_ch_names),
                    "ch_names": list(seg_ch_names),
                    "recording_id": recording_id,
                    "dataset": cfg.get("dataset"),
                    "subject_id": cfg.get("subject_id"),
                    "session_date": cfg.get("session_date"),
                    "tags": cfg.get("tags") or {},
                    "source_eeg_path": str(eeg_path),
                    "label_pipeline": label_pipeline,
                    "depd_mode": depd_mode,
                    "intervals_by_channel": seg_intervals_ch,
                    "intervals_global": seg_intervals_gl,
                    "annotated_region_sec": [t_crop_lo, t_crop_hi],
                }
                with meta_abs.open("w", encoding="utf-8") as mf:
                    json.dump(meta, mf, ensure_ascii=False, indent=2)

                segments_meta.append(
                    {
                        "index": seg_index,
                        "tmin_global_sec": t,
                        "tmax_global_sec": t1,
                        "n_samples": int(seg_data.shape[1]),
                        "npz": npz_rel,
                        "meta": meta_rel,
                    }
                )
            finally:
                del seg_raw

            seg_index += 1
            t = t1

        manifest = {
            "version": 2,
            "storage": "npz_bipolar",
            "recording_id": recording_id,
            "source_eeg_path": str(eeg_path),
            "eeg_stem": stem,
            "segment_duration_sec": segment_duration_sec,
            "n_segments": len(segments_meta),
            "sfreq": sfreq,
            "crop_region_global_sec": [t_crop_lo, t_crop_hi],
            "segments": segments_meta,
        }
        man_path = out_dir / "export_manifest.json"
        with man_path.open("w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        return man_path
    finally:
        del raw


def build_windows_index_for_segment_meta(
    meta: dict[str, Any],
    *,
    window_sec: float,
    overlap_sec: float,
    n_label_channels: int = 18,
    use_global_intervals: bool = False,
) -> list[dict[str, Any]]:
    """
    Вспомогательно: индекс скользящих окон по одному meta.json (локальные секунды).
    По умолчанию метки по каналам из intervals_by_channel; иначе intervals_global
    даёт один «канал» логики — здесь при use_global_intervals=True все depd_ch_i
    получают одно значение (есть ли пересечение с любым глобальным интервалом).
    """
    sfreq = float(meta["sfreq"])
    dur = float(meta["duration_sec"])
    n_times = int(round(dur * sfreq))

    if use_global_intervals:
        events = []
        for row in meta.get("intervals_global") or []:
            s0 = int(float(row["start"]) * sfreq)
            s1 = int(float(row["end"]) * sfreq)
            events.append({"start": s0, "end": s1})
    else:
        events = []
        for row in meta.get("intervals_by_channel") or []:
            ch = int(row["channel"])
            s0 = int(float(row["start"]) * sfreq)
            s1 = int(float(row["end"]) * sfreq)
            events.append({"channel": ch, "start": s0, "end": s1})

    window_samples = int(window_sec * sfreq)
    step_samples = int((window_sec - overlap_sec) * sfreq)
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_sec и overlap_sec должны давать положительное окно и шаг")

    rows: list[dict[str, Any]] = []
    widx = 0
    for start in range(0, max(0, n_times - window_samples + 1), step_samples):
        end = start + window_samples
        if use_global_intervals:
            hit = False
            for e in events:
                if not (e["end"] <= start or e["start"] >= end):
                    hit = True
                    break
            labs = [1.0 if hit else 0.0] * n_label_channels
        else:
            labels = np.zeros(n_label_channels, dtype=float)
            for ch in range(1, n_label_channels + 1):
                hit = False
                for e in events:
                    if e.get("channel") != ch:
                        continue
                    if not (e["end"] <= start or e["start"] >= end):
                        hit = True
                        break
                labels[ch - 1] = 1.0 if hit else 0.0
            labs = labels.tolist()

        rec: dict[str, Any] = {
            "window_index": widx,
            "start_sample": start,
            "end_sample": end,
            "start_sec": start / sfreq,
            "end_sec": end / sfreq,
        }
        for i, v in enumerate(labs, start=1):
            rec[f"depd_ch_{i}"] = v
        rows.append(rec)
        widx += 1
    return rows


def save_windows_index_csv(
    meta: dict[str, Any],
    csv_path: str | Path,
    *,
    window_sec: float,
    overlap_sec: float,
    n_label_channels: int = 18,
    use_global_intervals: bool = False,
) -> Path:
    import pandas as pd

    rows = build_windows_index_for_segment_meta(
        meta,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        n_label_channels=n_label_channels,
        use_global_intervals=use_global_intervals,
    )
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8")
    return p
