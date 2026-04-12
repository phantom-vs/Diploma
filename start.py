import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent / "eeg_segments_kit"
sys.path.insert(0, str(KIT))

import segment_saver

BASE_CFG = segment_saver.load_config_json(KIT / "export_config.example.json")
OUT_ROOT = Path(__file__).resolve().parent.parent / "exports"

JOBS = [
    {
        "eeg": Path("/Users/vspyatochkin/diploma_dev/new_data/NKT/EEG2100/FA0183YC.EEG"),
        "subject_id": "FA0183YD",
        "recording_id": "FA0183YD_run1",
    },
    # {
    #     "eeg": Path("/Users/vspyatochkin/diploma_dev/new_data/NKT/EEG2100/FA0183YC.EEG"),
    #     "subject_id": "FA0183YC",
    #     "recording_id": "FA0183YC_run1",
    # },
]

BASE_CFG["session_date"] = "2025-03-21"

for job in JOBS:
    cfg = {**BASE_CFG, **{k: v for k, v in job.items() if k != "eeg"}}
    eeg = job["eeg"]
    out = OUT_ROOT / cfg["subject_id"] / cfg["recording_id"]
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = segment_saver.export_recording_to_segments(eeg, out, cfg)
    print(eeg.name, "->", manifest_path)
