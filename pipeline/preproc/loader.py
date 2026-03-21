from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import mne

from preproc.depd_intervals_core import depd_intervals_from_annotations
from preproc.preprocessor import filter_dep_annotations_from_raw


def read_raw_nihon_header(eeg_path: Path, preload: bool = False):
    """Nihon через MNE; preload=False — без загрузки сигнала в память."""
    # .LOG часто не в UTF-8, MNE ругается, события всё равно читает
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*[Dd]ecode log as utf-8.*",
            category=RuntimeWarning,
        )
        return mne.io.read_raw_nihon(str(eeg_path), preload=preload, verbose=False)


def build_recording_context(
    *,
    recording_id: str,
    dataset: str,
    subject_id: str,
    session_date: str,
    source_eeg_path: Path,
    raw,
    tags: Any = None,
) -> dict[str, Any]:
    """Контекст одной записи: аннотации, интервалы DEPD, sfreq, без массива сигнала."""
    depd_ann = filter_dep_annotations_from_raw(raw)
    intervals = depd_intervals_from_annotations(depd_ann)
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    if tags is None:
        tags = {}
    return {
        "recording_id": recording_id,
        "dataset": dataset,
        "subject_id": subject_id,
        "session_date": session_date,
        "source_eeg_path": str(source_eeg_path.resolve()),
        "tags": tags,
        "sfreq": sfreq,
        "n_times": n_times,
        "depd_annotations": depd_ann,
        "depd_intervals": intervals,
    }
