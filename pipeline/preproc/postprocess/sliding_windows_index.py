from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preproc.manifest import build_run_manifest, write_manifest_yaml
from preproc.registry import register_postprocessor


def _events_as_samples(
    depd_intervals: list[dict[str, Any]], sfreq: float
) -> list[dict[str, Any]]:
    ev = []
    for row in depd_intervals:
        ch = int(row["channel"])
        s0 = int(float(row["start"]) * sfreq)
        s1 = int(float(row["end"]) * sfreq)
        ev.append({"channel": ch, "start": s0, "end": s1})
    return ev


def _window_labels_for_channels(
    start: int,
    end: int,
    events: list[dict[str, Any]],
    n_label_channels: int,
) -> list[float]:
    """1.0 если на канале есть пересечение с любым интервалом DEPD в [start, end)."""
    labels = np.zeros(n_label_channels, dtype=float)
    for ch in range(1, n_label_channels + 1):
        hit = False
        for e in events:
            if e["channel"] != ch:
                continue
            if not (e["end"] <= start or e["start"] >= end):
                hit = True
                break
        labels[ch - 1] = 1.0 if hit else 0.0
    return labels.tolist()


@register_postprocessor("sliding_windows_index")
def run_sliding_windows_index(
    ctx: dict[str, Any], params: dict[str, Any], out_dir: Path
) -> list[Path]:
    """Индекс окон + depd_ch_*; шаг = window_sec - overlap_sec; max_duration_sec опционально."""
    out_dir.mkdir(parents=True, exist_ok=True)
    window_sec = float(params["window_sec"])
    overlap_sec = float(params["overlap_sec"])
    n_ch = int(params.get("n_label_channels", 18))
    max_dur = params.get("max_duration_sec")
    basename = params.get("output_basename", "windows_index.csv")

    sfreq = float(ctx["sfreq"])
    n_times_full = int(ctx["n_times"])
    n_limit = n_times_full
    if max_dur is not None:
        n_limit = min(n_limit, int(float(max_dur) * sfreq))

    window_samples = int(window_sec * sfreq)
    step_samples = int((window_sec - overlap_sec) * sfreq)
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_sec и overlap_sec должны давать положительное окно и шаг")

    events = _events_as_samples(ctx.get("depd_intervals") or [], sfreq)

    rows: list[dict[str, Any]] = []
    widx = 0
    for start in range(0, n_limit - window_samples + 1, step_samples):
        end = start + window_samples
        labs = _window_labels_for_channels(start, end, events, n_ch)
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

    csv_path = out_dir / basename
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")

    manifest_path = out_dir / "manifest.yaml"
    write_manifest_yaml(
        manifest_path,
        build_run_manifest(
            ctx,
            params,
            kind="sliding_windows_index",
            artifacts=[csv_path.name],
        ),
    )
    return [csv_path, manifest_path]
