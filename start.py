import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent / "eeg_segments_kit"
sys.path.insert(0, str(KIT))

import segment_saver

BASE_CFG = segment_saver.load_config_json(KIT / "export_config.example.json")
OUT_ROOT = Path(__file__).resolve().parent.parent / "exports"

JOBS = [
    {
        "eeg": Path("/Users/vspyatochkin/diploma_dev_git/data/008/NKT/EEG2100/GA15011B.EEG"),
        "subject_id": "008",
        "recording_id": "GA15011B",
    },
    # {
    #     "eeg": Path("/Users/vspyatochkin/diploma_dev/new_data/NKT/EEG2100/FA0183YC.EEG"),
    #     "subject_id": "FA0183YC",
    #     "recording_id": "FA0183YC_run1",
    # },
]

BASE_CFG["session_date"] = "2025-03-21"


def print_recording_labels(eeg_path: Path) -> None:
    raw = segment_saver.read_raw_nihon(eeg_path, preload=False)
    try:
        descs = [str(d) for d in raw.annotations.description]
        unique = sorted(set(descs), key=lambda x: (x.lower(), x))
        print(
            f"\n=== Метки в записи {eeg_path.name} "
            f"(уникальных: {len(unique)}, событий: {len(descs)}) ==="
        )
        for d in unique:
            print(f"  {d}")
    finally:
        del raw


for job in JOBS:
    cfg = {**BASE_CFG, **{k: v for k, v in job.items() if k != "eeg"}}
    eeg = job["eeg"]
    print_recording_labels(eeg)
    out = OUT_ROOT / cfg["subject_id"] / cfg["recording_id"]
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = segment_saver.export_recording_to_segments(eeg, out, cfg)
    print(eeg.name, "->", manifest_path)
