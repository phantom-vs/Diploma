from __future__ import annotations

from typing import Any


def path_template_mapping(
    rec: dict[str, Any],
    *,
    recording_id: str,
    preproc_id: str,
) -> dict[str, str]:
    """Словарь для path_template из полей записи и id шага."""
    return {
        "dataset": str(rec["dataset"]),
        "subject_id": str(rec["subject_id"]),
        "session_date": str(rec["session_date"]),
        "recording_id": recording_id,
        "preproc_id": preproc_id,
    }
