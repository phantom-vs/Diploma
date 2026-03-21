from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from preproc.paths import resolve_eeg_path


def iter_recordings_with_eeg_base(
    cfg: dict[str, Any], config_parent: Path
) -> Iterator[tuple[dict[str, Any], Path]]:
    """Для каждой записи: (словарь записи, база для eeg_path).

    Вариант 1 — datasets: у блока id, root, recordings; eeg_path относительно root.
    Вариант 2 — верхний recordings: у строки есть dataset, путь относительно каталога конфига.
    """
    datasets = cfg.get("datasets")
    flat = cfg.get("recordings")

    if datasets and flat:
        raise SystemExit("в конфиге нельзя одновременно datasets и recordings")

    if datasets:
        if not isinstance(datasets, list):
            raise SystemExit("datasets должен быть списком")
        for block in datasets:
            if not isinstance(block, dict):
                raise SystemExit("каждый элемент datasets — объект YAML")
            ds_id = block.get("id")
            if ds_id is None or str(ds_id).strip() == "":
                raise SystemExit("у каждого датасета должно быть непустое поле id")
            ds_id = str(ds_id)
            root_raw = block.get("root")
            if root_raw is None:
                raise SystemExit(f"у датасета {ds_id!r} нет поля root")
            root = Path(str(root_raw)).expanduser()
            if not root.is_absolute():
                root = (config_parent / root).resolve()
            recs = block.get("recordings") or []
            if not recs:
                continue
            for rec in recs:
                if not isinstance(rec, dict):
                    raise SystemExit("каждая запись в recordings — объект YAML")
                merged = dict(rec)
                merged["dataset"] = ds_id
                yield merged, root
        return

    if flat:
        for rec in flat:
            if not isinstance(rec, dict):
                raise SystemExit("каждая запись в recordings — объект YAML")
            merged = dict(rec)
            if "dataset" not in merged or str(merged["dataset"]).strip() == "":
                raise SystemExit(
                    "в плоском recordings у строки нужно поле dataset, либо перейдите на datasets:"
                )
            yield merged, config_parent
        return

    raise SystemExit("задайте datasets или recordings")


def resolve_recording_eeg_path(rec: dict[str, Any], eeg_base: Path) -> Path:
    return resolve_eeg_path(str(rec["eeg_path"]), eeg_base)
