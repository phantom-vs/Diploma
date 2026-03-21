from __future__ import annotations

from pathlib import Path


def render_path_template(template: str, mapping: dict) -> Path:
    """Подстановка плейсхолдеров в path_template -> относительный Path."""
    try:
        rel = template.format(**mapping)
    except KeyError as e:
        raise KeyError(
            f"path_template: нет плейсхолдера {e!s}. "
            "Допустимы: dataset, subject_id, session_date, recording_id, preproc_id. "
            f"Передано: {sorted(mapping)}"
        ) from e
    return Path(rel)


def resolve_eeg_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()
