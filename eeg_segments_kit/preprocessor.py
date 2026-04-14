"""Метки из LOG + вытаскивание интервалов DEP (цифры / точка, маркеры и т.д.)."""
from __future__ import annotations

import re
from typing import Any, Callable

_LETTERS_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ]")


def _strip_description(desc: str) -> str:
    return (desc or "").strip()


def label_step_spaces_to_commas(text: str) -> str:
    # врачи любят пробелы вместо запятых
    s = _strip_description(text)
    return s.replace(" ", ",")


def label_step_collapse_duplicate_commas(text: str) -> str:
    s = _strip_description(text)
    while ",," in s:
        s = s.replace(",,", ",")
    return s.strip(",")


def label_step_expand_merged_channels_ge19(
    text: str,
    *,
    min_channel: int = 19,
    max_channel: int = 99,
) -> str:
    """
    Внутри каждого «токена» между запятыми: если строка только из цифр,
    чётной длины >= 4 и разбиение на пары даёт числа в [min_channel, max_channel],
    заменить токен на список через запятую (пары слева направо).
    Иначе токен не менять. 192021 -> 19,20,21 если все пары в диапазоне; иначе не трогаем
    """
    parts = [p for p in text.split(",") if p.strip() != ""]
    out: list[str] = []
    for raw_tok in parts:
        tok = raw_tok.strip().replace(" ", "")
        if not tok.isdigit() or len(tok) < 4 or len(tok) % 2 != 0:
            out.append(tok)
            continue
        pairs: list[int] = []
        ok = True
        for i in range(0, len(tok), 2):
            v = int(tok[i : i + 2])
            if not (min_channel <= v <= max_channel):
                ok = False
                break
            pairs.append(v)
        if ok and len(pairs) >= 2:
            out.extend(str(v) for v in pairs)
        else:
            out.append(tok)
    return ",".join(out)


LABEL_STEP_REGISTRY: dict[str, Callable[[str], str]] = {
    "spaces_to_commas": label_step_spaces_to_commas,
    "collapse_duplicate_commas": label_step_collapse_duplicate_commas,
    "expand_merged_channels_ge19": label_step_expand_merged_channels_ge19,
}


def apply_label_pipeline(description: str, step_names: list[str] | None) -> str:
    # имена из LABEL_STEP_REGISTRY, опечатка в конфиге = KeyError
    s = str(description) if description is not None else ""
    if not step_names:
        return s
    for name in step_names:
        fn = LABEL_STEP_REGISTRY[str(name)]
        s = fn(s)
    return s


def annotation_description_has_letters(description: str) -> bool:
    return bool(_LETTERS_RE.search(_strip_description(description)))


def filter_dep_like_annotation_dicts(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Оставить аннотации без букв в описании"""
    out: list[dict[str, Any]] = []
    for ann in annotations:
        desc = _strip_description(str(ann.get("description", "")))
        if not desc or _LETTERS_RE.search(desc):
            continue
        out.append({"onset": float(ann["onset"]), "description": desc})
    return out


def preprocess_annotation_descriptions(
    annotations: list[dict[str, Any]],
    label_pipeline: list[str] | None,
) -> list[dict[str, Any]]:
    """Копия списка аннотаций с прогоном description через apply_label_pipeline."""
    out: list[dict[str, Any]] = []
    for ann in annotations:
        desc = apply_label_pipeline(str(ann.get("description", "")), label_pipeline)
        out.append({"onset": float(ann["onset"]), "description": desc})
    return out


def parse_depd_channel_values(value_str: str) -> list[int]:
    cleaned = value_str.replace(" ", "").replace(".", "").strip()
    if not cleaned:
        return []
    channels: list[int] = []
    for part in cleaned.split(","):
        if part.isdigit():
            channels.append(int(part))
    return channels


def depd_intervals_from_annotations(
    depd_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Интервалы по каналам: число в описании до следующей точки '.'.
    Каждый интервал: start, end (сек), channel.
    """
    intervals: list[dict[str, Any]] = []
    n = len(depd_annotations)
    for i, ann in enumerate(depd_annotations):
        current_desc = ann["description"]
        current_cleaned = current_desc.replace(",", "").replace(" ", "").replace(".", "")
        if not current_cleaned or not current_cleaned.isdigit() or current_desc == ".":
            continue
        channels = parse_depd_channel_values(current_desc)
        if not channels:
            continue
        for j in range(i + 1, n):
            if depd_annotations[j]["description"] == ".":
                start = float(ann["onset"])
                end = float(depd_annotations[j]["onset"])
                for channel in channels:
                    intervals.append({"start": start, "end": end, "channel": channel})
                break
    return intervals


def global_intervals_from_markers(
    annotations: list[dict[str, Any]],
    start_markers: list[str],
    end_markers: list[str],
) -> list[dict[str, Any]]:
    """
    Пары (start, end) в секундах по текстовым маркерам без привязки к каналу.
    start_markers / end_markers — списки возможных описаний.
    """
    starts = {_strip_description(s) for s in start_markers}
    ends = {_strip_description(s) for s in end_markers}
    anns = sorted(annotations, key=lambda a: float(a["onset"]))
    intervals: list[dict[str, Any]] = []
    open_onset: float | None = None
    for ann in anns:
        desc = _strip_description(str(ann["description"]))
        t = float(ann["onset"])
        if open_onset is None and desc in starts:
            open_onset = t
        elif open_onset is not None and desc in ends:
            intervals.append({"start": open_onset, "end": t, "channel": None})
            open_onset = None
    return intervals


def _merge_touching_intervals(
    intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Склеить пересекающиеся или соприкасающиеся [start, end) в один список."""
    if not intervals:
        return []
    srt = sorted(intervals, key=lambda x: (float(x["start"]), float(x["end"])))
    out: list[dict[str, Any]] = [{"start": float(srt[0]["start"]), "end": float(srt[0]["end"])}]
    for row in srt[1:]:
        s, e = float(row["start"]), float(row["end"])
        last = out[-1]
        le = float(last["end"])
        if s <= le + 1e-9:
            last["end"] = max(le, e)
        else:
            out.append({"start": s, "end": e})
    return out


def sleep_wake_global_intervals(
    annotations: list[dict[str, Any]],
    sleep_start_markers: list[str],
    wake_start_markers: list[str],
    *,
    t_end_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Глобальные интервалы сна и бодрствования (секунды от начала записи).

    sleep_start_markers — метки начала сна (например «Stage 1»).
    wake_start_markers — метки начала бодрствования (например «Waking»).
    Состояние до первого sleep-маркера — бодрствование. Дальше чередование по маркерам.

    Если ни одна аннотация не совпала с маркерами: весь [0, t_end_sec) — бодрствование,
    список сна пустой.
    """
    ss = {_strip_description(s) for s in sleep_start_markers if str(s).strip()}
    ws = {_strip_description(s) for s in wake_start_markers if str(s).strip()}
    te = float(t_end_sec)
    if te <= 0:
        return [], []

    events: list[tuple[float, str]] = []
    anns = sorted(annotations, key=lambda a: float(a["onset"]))
    for ann in anns:
        desc = _strip_description(str(ann["description"]))
        t = float(ann["onset"])
        if t >= te:
            break
        if desc in ss:
            events.append((t, "sleep"))
        elif desc in ws:
            events.append((t, "wake"))

    if not events:
        return [], [{"start": 0.0, "end": te}]

    intervals_sleep: list[dict[str, float]] = []
    intervals_wake: list[dict[str, float]] = []
    state = "wake"
    prev_t = 0.0

    for t, kind in events:
        if t >= te:
            break
        if kind == "sleep" and state == "wake":
            if t > prev_t:
                intervals_wake.append({"start": prev_t, "end": t})
            prev_t = t
            state = "sleep"
        elif kind == "wake" and state == "sleep":
            if t > prev_t:
                intervals_sleep.append({"start": prev_t, "end": t})
            prev_t = t
            state = "wake"

    if state == "wake":
        if te > prev_t:
            intervals_wake.append({"start": prev_t, "end": te})
    else:
        if te > prev_t:
            intervals_sleep.append({"start": prev_t, "end": te})

    return (
        _merge_touching_intervals(intervals_sleep),
        _merge_touching_intervals(intervals_wake),
    )


def find_annotated_region_bounds_sec(
    annotations: list[dict[str, Any]],
    region_start_markers: list[str],
    region_end_markers: list[str],
    *,
    total_duration_sec: float,
) -> tuple[float | None, float | None, bool]:
    # первая start, потом первая end после неё; если конца нет — t_end = длина записи, bool=False
    sset = {_strip_description(s) for s in region_start_markers}
    eset = {_strip_description(s) for s in region_end_markers}
    anns = sorted(annotations, key=lambda a: float(a["onset"]))
    t_start: float | None = None
    for ann in anns:
        desc = _strip_description(str(ann["description"]))
        t = float(ann["onset"])
        if t_start is None and desc in sset:
            t_start = t
            break
    if t_start is None:
        return None, None, False
    t_end: float | None = None
    end_marker_found = False
    for ann in anns:
        desc = _strip_description(str(ann["description"]))
        t = float(ann["onset"])
        if t <= t_start:
            continue
        if desc in eset:
            t_end = t
            end_marker_found = True
            break
    if t_end is None:
        t_end = float(total_duration_sec)
    return t_start, t_end, end_marker_found


def filter_intervals_intersecting_range(
    intervals: list[dict[str, Any]],
    t_lo: float,
    t_hi: float,
) -> list[dict[str, Any]]:
    """Оставить интервалы, пересекающиеся с [t_lo, t_hi) (глобальные секунды)."""
    out: list[dict[str, Any]] = []
    for row in intervals:
        s = float(row["start"])
        e = float(row["end"])
        if e <= t_lo or s >= t_hi:
            continue
        out.append(dict(row))
    return out


def clip_intervals_to_range(
    intervals: list[dict[str, Any]],
    t_lo: float,
    t_hi: float,
) -> list[dict[str, Any]]:
    """
    Пересечь каждый интервал с [t_lo, t_hi) в глобальных секундах (обрезка границ).
    В отличие от filter_intervals_intersecting_range, start/end сужаются до окна —
    суммы длительностей и доли в окне считаются корректно.
    """
    out: list[dict[str, Any]] = []
    for row in intervals:
        s = float(row["start"])
        e = float(row["end"])
        lo = max(s, t_lo)
        hi = min(e, t_hi)
        if hi <= lo:
            continue
        item: dict[str, Any] = {"start": lo, "end": hi}
        if "channel" in row:
            item["channel"] = row["channel"]
        out.append(item)
    return out


def clip_interval_to_segment(
    start_sec: float,
    end_sec: float,
    seg_tmin: float,
    seg_tmax: float,
) -> tuple[float, float] | None:
    """Пересечение [start,end) с [seg_tmin, seg_tmax) в глобальных секундах -> локальные секунды от seg_tmin."""
    lo = max(start_sec, seg_tmin)
    hi = min(end_sec, seg_tmax)
    if hi <= lo:
        return None
    return lo - seg_tmin, hi - seg_tmin


def intervals_for_segment(
    intervals: list[dict[str, Any]],
    seg_tmin: float,
    seg_tmax: float,
) -> list[dict[str, Any]]:
    """Сдвиг start/end в локальное время сегмента [0, seg_tmax-seg_tmin)."""
    local: list[dict[str, Any]] = []
    for row in intervals:
        clip = clip_interval_to_segment(
            float(row["start"]), float(row["end"]), seg_tmin, seg_tmax
        )
        if clip is None:
            continue
        ls, le = clip
        item = {"start": ls, "end": le}
        if "channel" in row and row["channel"] is not None:
            item["channel"] = int(row["channel"])
        else:
            item["channel"] = row.get("channel")
        local.append(item)
    return local


def intervals_for_segment_time_only(
    intervals: list[dict[str, Any]],
    seg_tmin: float,
    seg_tmax: float,
) -> list[dict[str, float]]:
    """Как intervals_for_segment, но только start/end (без channel) — для сна/бодрствования."""
    local: list[dict[str, float]] = []
    for row in intervals:
        clip = clip_interval_to_segment(
            float(row["start"]), float(row["end"]), seg_tmin, seg_tmax
        )
        if clip is None:
            continue
        ls, le = clip
        local.append({"start": ls, "end": le})
    return local
