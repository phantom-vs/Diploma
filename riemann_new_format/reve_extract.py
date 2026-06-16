"""Extract REVE embeddings from new_format_data memmaps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .dataset import N_CHANNELS, N_SAMPLES

REVE_DIR = Path(__file__).resolve().parents[1] / "diploma_new" / "eeg_toolkit" / "REVE"
CH_NAMES = [
    "Fp1-F3", "Fp2-F4", "F3-C3", "F4-C4", "C3-P3", "C4-P4",
    "P3-O1", "P4-O2", "Fp1-F7", "Fp2-F8", "F7-T3", "F8-T4",
    "T3-T5", "T4-T6", "T5-O1", "T6-O2", "Fz-Cz", "Cz-Pz",
]


def normalize_ch_name(raw: str) -> str:
    s = str(raw).strip()
    return s.split("-")[-1] if "-" in s else s


def load_reve_models(reve_dir: Path | None = None):
    import torch
    from transformers import AutoModel

    reve_dir = reve_dir or REVE_DIR
    base_path = reve_dir / "reve-base"
    pos_path = reve_dir / "reve-positions"

    if not base_path.is_dir():
        raise FileNotFoundError(f"REVE base model not found: {base_path}")

    if pos_path.is_dir() and (pos_path / "config.json").is_file():
        pos_bank = AutoModel.from_pretrained(str(pos_path), trust_remote_code=True)
    else:
        print("Downloading REVE position bank from HuggingFace (brain-bzh/reve-positions)...")
        pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)

    reve_model = AutoModel.from_pretrained(str(base_path), trust_remote_code=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    reve_model.to(device)
    reve_model.eval()
    print(f"REVE loaded on {device}")
    return reve_model, pos_bank, device


def get_positions_for_ch_names(ch_names: list[str], pos_bank, device):
    import torch

    norm = [normalize_ch_name(ch) for ch in ch_names]
    positions = pos_bank(norm)
    if not isinstance(positions, torch.Tensor):
        positions = torch.as_tensor(positions, dtype=torch.float32)
    return positions.to(device)


def _pool_reve_output(out) -> "torch.Tensor":
    import torch

    if out.dim() == 4:
        return out.squeeze(2).mean(dim=1)
    if out.dim() == 3:
        return out.mean(dim=1)
    if out.dim() == 2:
        return out
    raise RuntimeError(f"Unexpected REVE output shape: {tuple(out.shape)}")


def _pool_reve_patches(out) -> "torch.Tensor":
    """Mean over channels; keep temporal patches -> (B, n_patches, D)."""
    import torch

    if out.dim() == 4:
        return out.mean(dim=1)
    if out.dim() == 3:
        return out.unsqueeze(1)
    if out.dim() == 2:
        return out.unsqueeze(1)
    raise RuntimeError(f"Unexpected REVE output shape: {tuple(out.shape)}")


def extract_reve_from_memmap(
    raw_memmap: np.ndarray,
    indices: np.ndarray,
    *,
    reve_model,
    pos_bank,
    device,
    batch_size: int = 32,
    ch_names: list[str] | None = None,
) -> np.ndarray:
    """Return REVE patch embeddings flattened: (n, n_patches * 512)."""
    import torch

    ch_names = ch_names or CH_NAMES
    positions = get_positions_for_ch_names(ch_names, pos_bank, device)
    n = len(indices)
    out_batches: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        flat = np.asarray(raw_memmap[batch_idx], dtype=np.float32)
        data = flat.reshape(len(batch_idx), N_CHANNELS, N_SAMPLES)
        data_t = torch.from_numpy(data).float().to(device)
        pos_batch = positions.unsqueeze(0).expand(data_t.size(0), -1, -1)

        with torch.no_grad():
            out = reve_model(data_t, pos_batch)
            pooled = _pool_reve_output(out)

        if start == 0:
            print(
                f"  REVE out {tuple(out.shape)} -> pooled {tuple(pooled.shape)}",
                flush=True,
            )

        out_batches.append(pooled.cpu().numpy())

        if start == 0 or (start // batch_size) % 100 == 0:
            print(f"    REVE extracted {min(start + batch_size, n):,}/{n:,}", flush=True)

    return np.concatenate(out_batches, axis=0).astype(np.float32)


def extract_reve_patches_from_windows(
    windows: np.ndarray,
    *,
    reve_model,
    pos_bank,
    device,
    batch_size: int = 32,
    ch_names: list[str] | None = None,
) -> np.ndarray:
    """Return patch embeddings (n, n_patches, 512) for legacy 2 s @ 500 Hz windows."""
    import torch

    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError(f"Expected (n, C, T), got {windows.shape}")

    ch_names = ch_names or CH_NAMES
    positions = get_positions_for_ch_names(ch_names, pos_bank, device)
    n = windows.shape[0]
    out_batches: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        batch = windows[start : start + batch_size]
        data_t = torch.from_numpy(batch).float().to(device)
        pos_batch = positions.unsqueeze(0).expand(data_t.size(0), -1, -1)

        with torch.no_grad():
            out = reve_model(data_t, pos_batch)
            pooled = _pool_reve_patches(out)

        if start == 0:
            print(
                f"  REVE patches {tuple(out.shape)} -> {tuple(pooled.shape)}",
                flush=True,
            )

        out_batches.append(pooled.cpu().numpy())
        if start == 0 or (start // batch_size) % 100 == 0:
            print(f"    REVE extracted {min(start + batch_size, n):,}/{n:,}", flush=True)

    return np.concatenate(out_batches, axis=0).astype(np.float32)


def cache_path(cache_dir: Path, name: str, indices: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{name}_{len(indices)}_{int(indices[0])}_{int(indices[-1])}"
    return cache_dir / f"{key}.npy"


def cache_path_n(cache_dir: Path, name: str, n: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}_{n}.npy"


def load_or_extract_reve(
    raw_memmap: np.ndarray,
    indices: np.ndarray,
    *,
    cache_dir: Path,
    name: str,
    reve_model,
    pos_bank,
    device,
    batch_size: int,
    reuse_cache: bool,
) -> np.ndarray:
    path = cache_path(cache_dir, name, indices)
    if reuse_cache and path.is_file():
        print(f"  Loading cached REVE: {path.name}")
        return np.load(path)
    print(f"  Extracting REVE -> {path.name}")
    emb = extract_reve_from_memmap(
        raw_memmap,
        indices,
        reve_model=reve_model,
        pos_bank=pos_bank,
        device=device,
        batch_size=batch_size,
    )
    np.save(path, emb)
    return emb


def load_or_extract_reve_patches(
    windows: np.ndarray,
    *,
    cache_dir: Path,
    name: str,
    reve_model,
    pos_bank,
    device,
    batch_size: int,
    reuse_cache: bool,
) -> np.ndarray:
    path = cache_path_n(cache_dir, name, len(windows))
    if reuse_cache and path.is_file():
        print(f"  Loading cached REVE patches: {path.name}")
        return np.load(path)
    print(f"  Extracting REVE patches -> {path.name}")
    emb = extract_reve_patches_from_windows(
        windows,
        reve_model=reve_model,
        pos_bank=pos_bank,
        device=device,
        batch_size=batch_size,
    )
    np.save(path, emb)
    return emb
