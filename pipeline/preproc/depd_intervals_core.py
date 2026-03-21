from __future__ import annotations

from typing import Any


def parse_depd_values(value_str: str) -> list[int]:
    """Строка каналов из разметки -> список номеров."""
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
    """Интервалы по каналам: число в описании до следующей точки «.» (как в старом preprocessing.py)."""
    intervals: list[dict[str, Any]] = []
    n = len(depd_annotations)
    for i, ann in enumerate(depd_annotations):
        current_desc = ann["description"]
        current_cleaned = current_desc.replace(",", "").replace(" ", "").replace(".", "")
        if not current_cleaned or not current_cleaned.isdigit() or current_desc == ".":
            continue
        channels = parse_depd_values(current_desc)
        if not channels:
            continue
        for j in range(i + 1, n):
            if depd_annotations[j]["description"] == ".":
                start = float(ann["onset"])
                end = float(depd_annotations[j]["onset"])
                for channel in channels:
                    intervals.append(
                        {"start": start, "end": end, "channel": channel}
                    )
                break
    return intervals
