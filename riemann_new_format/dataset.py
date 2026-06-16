"""Load new_format_data memmaps and build patient-level train/test splits."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

N_CHANNELS = 18
N_SAMPLES = 200
RAW_FLAT_DIM = N_CHANNELS * N_SAMPLES

# Precomputed feature memmaps (excluding raw_signals).
# Compact set from ablation on patient split (features-only LR, 100k train windows):
# spectral+sync+turbulence gave best F1; morphology/hjorth overlap and hurt when stacked.
DEFAULT_FEATURE_NAMES = (
    "spectral",
    "sync",
    "turbulence",
    "mutual_info",
    "correlation_dynamics",
)

# Previous broad bundle (kept for experiments): morphology,hjorth,spectral,sync,spike_wave,wavelet
LEGACY_FEATURE_NAMES = (
    "morphology",
    "hjorth",
    "spectral",
    "sync",
    "spike_wave",
    "wavelet",
)

ALL_FEATURE_NAMES = (
    "morphology",
    "hjorth",
    "spectral",
    "sync",
    "spike_wave",
    "wavelet",
    "correlation",
    "correlation_dynamics",
    "mutual_info",
    "smart_correlation",
    "rqa",
    "simple",
    "turbulence",
    "inversion_cleaned",
)


def load_metadata(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "metadata.pkl"
    with path.open("rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, list):
        raise TypeError(f"Expected list in metadata.pkl, got {type(meta)}")
    return meta


def open_raw_memmap(data_dir: Path, n_windows: int) -> np.memmap:
    path = data_dir / "raw_signals.memmap"
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n_windows, RAW_FLAT_DIM))


def feature_dim_from_file(path: Path, n_windows: int) -> int:
    nbytes = path.stat().st_size
    if nbytes % (4 * n_windows) != 0:
        raise ValueError(f"Unexpected memmap size for {path.name}: {nbytes} bytes")
    return nbytes // (4 * n_windows)


def open_feature_memmap(data_dir: Path, name: str, n_windows: int) -> np.memmap:
    path = data_dir / f"{name}.memmap"
    if not path.is_file():
        raise FileNotFoundError(f"Feature memmap not found: {path}")
    dim = feature_dim_from_file(path, n_windows)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n_windows, dim))


def parse_feature_names(spec: str | None) -> tuple[str, ...]:
    """Parse --features: none/empty -> (), 'default', 'all', or comma-separated names."""
    if spec is None or spec.strip().lower() in {"", "none", "raw"}:
        return ()
    key = spec.strip().lower()
    if key == "default":
        return DEFAULT_FEATURE_NAMES
    if key in {"legacy", "old_default"}:
        return LEGACY_FEATURE_NAMES
    if key == "all":
        return ALL_FEATURE_NAMES
    names = tuple(s.strip() for s in spec.split(",") if s.strip())
    unknown = set(names) - set(ALL_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    return names


def feature_dim_map(data_dir: Path, feature_names: tuple[str, ...], n_windows: int) -> dict[str, int]:
    dims: dict[str, int] = {}
    for name in feature_names:
        path = data_dir / f"{name}.memmap"
        dims[name] = feature_dim_from_file(path, n_windows)
    return dims


def patient_train_test_split(
    meta: list[dict[str, Any]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split by external_id (patient). No patient appears in both train and test.
    Stratify: patient has at least one window with has_depd=True.
    """
    patient_has_depd: dict[str, bool] = {}
    for row in meta:
        pid = row["external_id"]
        if pid not in patient_has_depd:
            patient_has_depd[pid] = False
        if row.get("has_depd"):
            patient_has_depd[pid] = True

    patients = sorted(patient_has_depd.keys())
    stratify = [int(patient_has_depd[p]) for p in patients]

    train_patients, test_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_set = set(train_patients)
    test_set = set(test_patients)
    if train_set & test_set:
        raise RuntimeError("Patient leak: same patient in train and test")
    return list(train_patients), list(test_patients)


def indices_for_patients(meta: list[dict[str, Any]], patients: set[str] | list[str]) -> np.ndarray:
    patient_set = set(patients)
    return np.array(
        [i for i, row in enumerate(meta) if row["external_id"] in patient_set],
        dtype=np.int64,
    )


def labels_for_indices(meta: list[dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    return np.array([int(meta[i]["has_depd"]) for i in indices], dtype=np.int64)


def patient_ids_for_indices(meta: list[dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    return np.array([meta[i]["external_id"] for i in indices], dtype=object)


def subsample_train_indices(
    indices: np.ndarray,
    y: np.ndarray,
    *,
    neg_per_pos: float = 5.0,
    max_samples: int | None = 200_000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Balance train set: keep all positives, undersample negatives.
    Full train (~800k windows) is too heavy for in-memory Riemann fit on a laptop.
    """
    rng = np.random.default_rng(random_state)
    pos_idx = indices[y == 1]
    neg_idx = indices[y == 0]

    if len(pos_idx) == 0:
        raise ValueError("No positive train windows (has_depd=True)")
    if len(neg_idx) == 0:
        raise ValueError("No negative train windows (has_depd=False)")

    n_pos = len(pos_idx)
    n_neg = min(int(n_pos * neg_per_pos), len(neg_idx))

    if max_samples is not None and n_pos + n_neg > max_samples:
        pos_fraction = n_pos / (n_pos + n_neg)
        n_pos_cap = max(1, int(max_samples * pos_fraction))
        n_pos_cap = min(n_pos_cap, n_pos, max_samples - 1)
        n_neg = min(max_samples - n_pos_cap, len(neg_idx))
        n_pos = n_pos_cap

    pos_sampled = pos_idx if n_pos == len(pos_idx) else rng.choice(pos_idx, size=n_pos, replace=False)
    neg_sampled = neg_idx if n_neg == len(neg_idx) else rng.choice(neg_idx, size=n_neg, replace=False)

    chosen = np.concatenate([pos_sampled, neg_sampled])
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def save_split(
    data_dir: Path,
    train_patients: list[str],
    test_patients: list[str],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    out_dir = data_dir / "splits"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "test_size": test_size,
        "random_state": random_state,
        "n_train_patients": len(train_patients),
        "n_test_patients": len(test_patients),
        "train_patients": sorted(train_patients),
        "test_patients": sorted(test_patients),
    }
    path = out_dir / "patient_split_80_20.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_split(data_dir: Path) -> tuple[list[str], list[str]]:
    path = data_dir / "splits" / "patient_split_80_20.json"
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload["train_patients"], payload["test_patients"]


class NewFormatDataset:
    """Indexed access to raw_signals.memmap + optional feature memmaps + metadata."""

    def __init__(self, data_dir: str | Path, feature_names: tuple[str, ...] = ()):
        self.data_dir = Path(data_dir)
        self.meta = load_metadata(self.data_dir)
        self.n_windows = len(self.meta)
        self.feature_names = tuple(feature_names)
        self._raw = open_raw_memmap(self.data_dir, self.n_windows)
        self._features: dict[str, np.memmap] = {}
        self._feature_dims: dict[str, int] = {}
        self._nan_cols: dict[str, np.ndarray] = {}
        for name in self.feature_names:
            mmap = open_feature_memmap(self.data_dir, name, self.n_windows)
            self._features[name] = mmap
            sample = np.asarray(mmap[: min(10_000, self.n_windows)])
            all_nan = np.isnan(sample).all(axis=0)
            self._nan_cols[name] = np.flatnonzero(all_nan)
            self._feature_dims[name] = mmap.shape[1] - int(all_nan.sum())

    @property
    def n_features(self) -> int:
        return sum(self._feature_dims.values())

    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        """Return X shape (n, n_channels, n_samples)."""
        flat = np.asarray(self._raw[indices], dtype=np.float32)
        return flat.reshape(len(indices), N_CHANNELS, N_SAMPLES)

    def get_features_batch(self, indices: np.ndarray) -> np.ndarray | None:
        """Return stacked memmap features, shape (n, n_features)."""
        if not self._features:
            return None
        parts = []
        for name in self.feature_names:
            block = np.asarray(self._features[name][indices], dtype=np.float32)
            drop = self._nan_cols.get(name)
            if drop is not None and drop.size:
                block = np.delete(block, drop, axis=1)
            parts.append(block)
        return np.hstack(parts)

    def get_labels(self, indices: np.ndarray) -> np.ndarray:
        return labels_for_indices(self.meta, indices)
