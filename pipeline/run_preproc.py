#!/usr/bin/env python3
"""Запуск предобработки по YAML. Пример: python run_preproc.py --config configs/preproc.yaml

Пути в конфиге — относительно каталога файла конфигурации (если не абсолютные).
"""
from __future__ import annotations

import argparse
import copy
import uuid
from pathlib import Path

import yaml

from preproc.config_expand import iter_recordings_with_eeg_base, resolve_recording_eeg_path
from preproc.loader import build_recording_context, read_raw_nihon_header
from preproc.manifest import normalize_recording_tags
from preproc.paths import render_path_template
from preproc.recording_fields import path_template_mapping
from preproc.registry import get_postprocessor


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Предобработка EEG по конфигу YAML")
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config",
    )
    args = ap.parse_args()
    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent

    output_root = Path(cfg["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = (base_dir / output_root).resolve()
    path_template = cfg["path_template"]

    preprocessing = cfg.get("preprocessing") or []
    if not preprocessing:
        raise SystemExit("config: no preprocessing jobs")

    recording_jobs = list(iter_recordings_with_eeg_base(cfg, base_dir))
    if not recording_jobs:
        raise SystemExit("config: no recordings under datasets/recordings")

    for rec, eeg_base in recording_jobs:
        dataset = str(rec["dataset"])
        subject_id = str(rec["subject_id"])
        session_date = str(rec["session_date"])
        eeg_path = resolve_recording_eeg_path(rec, eeg_base)

        rid = (rec.get("recording_id") or "").strip()
        if not rid:
            rid = uuid.uuid4().hex[:12]

        tags = normalize_recording_tags(rec.get("tags"))

        raw = read_raw_nihon_header(eeg_path, preload=False)
        try:
            ctx = build_recording_context(
                recording_id=rid,
                dataset=dataset,
                subject_id=subject_id,
                session_date=session_date,
                source_eeg_path=eeg_path,
                raw=raw,
                tags=tags,
            )
        finally:
            del raw

        for job in preprocessing:
            preproc_id = str(job["id"])
            kind = str(job["kind"])
            params = copy.deepcopy(job.get("params") or {})
            params["_preproc_id"] = preproc_id

            rel = render_path_template(
                path_template,
                path_template_mapping(rec, recording_id=rid, preproc_id=preproc_id),
            )
            out_dir = output_root / rel
            fn = get_postprocessor(kind)
            written = fn(ctx, params, out_dir)
            print(f"{kind} -> {out_dir}")
            for p in written:
                print(f"  wrote {p}")


if __name__ == "__main__":
    main()
