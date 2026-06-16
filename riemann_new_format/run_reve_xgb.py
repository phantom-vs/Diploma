#!/usr/bin/env python3
"""
REVE + Mutual Information feature selection + XGBoost on new_format_data.

Protocol (matches pipline_with_all_features):
  raw flat EEG + top-k REVE dims (MI) -> StandardScaler -> XGBoost
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .dataset import (
    RAW_FLAT_DIM,
    indices_for_patients,
    labels_for_indices,
    load_split,
    subsample_train_indices,
)
from .dataset import NewFormatDataset
from .pipeline import evaluate_binary
from .reve_extract import load_or_extract_reve, load_reve_models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REVE + MI + XGBoost on new_format_data")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "new_format_data")
    p.add_argument("--reve-dir", type=Path, default=None)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--mi-k", type=int, default=256, help="Top-k REVE dims by mutual information")
    p.add_argument("--max-train-samples", type=int, default=200_000, help="0 = no cap (all DEPD + ratio)")
    p.add_argument("--full-train", action="store_true")
    p.add_argument("--neg-per-pos", type=float, default=5.0)
    p.add_argument("--output-suffix", default="", help="e.g. _full_1to2 for reve_xgb_mi{suffix}.json")
    p.add_argument("--reve-batch-size", type=int, default=32)
    p.add_argument("--reuse-cache", action="store_true", help="Load cached REVE .npy if present")
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


def load_raw_batch(raw_memmap: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(raw_memmap[indices], dtype=np.float32)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir or (data_dir / "reve_cache")

    print(f"Data: {data_dir}")
    t0 = time.perf_counter()
    ds = NewFormatDataset(data_dir)

    train_patients, test_patients = load_split(data_dir)
    train_idx = indices_for_patients(ds.meta, train_patients)
    test_idx = indices_for_patients(ds.meta, test_patients)
    y_train_full = labels_for_indices(ds.meta, train_idx)
    y_test = labels_for_indices(ds.meta, test_idx)

    if args.full_train:
        train_idx_fit = train_idx
        y_train = y_train_full
        print(f"Train: FULL {len(train_idx_fit):,}")
    else:
        max_samples = None if args.max_train_samples <= 0 else args.max_train_samples
        train_idx_fit = subsample_train_indices(
            train_idx,
            y_train_full,
            neg_per_pos=args.neg_per_pos,
            max_samples=max_samples,
            random_state=args.random_state,
        )
        y_train = labels_for_indices(ds.meta, train_idx_fit)
        print(
            f"Train: {len(train_idx_fit):,} | DEPD {y_train.sum():,} ({y_train.mean():.3f}) | "
            f"ratio 1:{args.neg_per_pos}"
        )

    print(f"Test: {len(test_idx):,} | DEPD train {y_train.sum():,} / test {y_test.sum():,}")

    reve_model, pos_bank, device = load_reve_models(args.reve_dir)

    print("\nREVE train embeddings...")
    reve_train = load_or_extract_reve(
        ds._raw,
        train_idx_fit,
        cache_dir=cache_dir,
        name="train",
        reve_model=reve_model,
        pos_bank=pos_bank,
        device=device,
        batch_size=args.reve_batch_size,
        reuse_cache=args.reuse_cache,
    )

    print("\nREVE test embeddings...")
    reve_test = load_or_extract_reve(
        ds._raw,
        test_idx,
        cache_dir=cache_dir,
        name="test",
        reve_model=reve_model,
        pos_bank=pos_bank,
        device=device,
        batch_size=args.reve_batch_size,
        reuse_cache=args.reuse_cache,
    )

    reve_train_flat = reve_train.reshape(len(reve_train), -1)
    reve_test_flat = reve_test.reshape(len(reve_test), -1)
    print(f"REVE dims: {reve_train_flat.shape[1]} (patches x 512)")

    X_train_raw = load_raw_batch(ds._raw, train_idx_fit)
    X_test_raw = load_raw_batch(ds._raw, test_idx)

    k = min(args.mi_k, reve_train_flat.shape[1])
    print(f"\nMI SelectKBest k={k} on REVE...")
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_reve_sel = selector.fit_transform(reve_train_flat, y_train)
    X_test_reve_sel = selector.transform(reve_test_flat)

    X_train = np.hstack([X_train_raw, X_train_reve_sel])
    X_test = np.hstack([X_test_raw, X_test_reve_sel])
    print(f"Combined features: raw {X_train_raw.shape[1]} + REVE {X_train_reve_sel.shape[1]} = {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    spw = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"\nTraining XGBoost (scale_pos_weight={spw:.2f})...")
    t_fit = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(spw),
        random_state=args.random_state,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train_s, y_train)
    print(f"  fit {time.perf_counter() - t_fit:.1f}s")

    y_pred = xgb.predict(X_test_s)
    metrics = evaluate_binary(y_test, y_pred)

    print("\n=== Test metrics (Raw + REVE MI + XGBoost) ===")
    for key in ("f1", "precision", "recall", "specificity", "accuracy"):
        print(f"  {key:14}: {metrics[key]:.4f}")

    suffix = "_full" if args.full_train else (args.output_suffix or "")
    out_path = data_dir / "splits" / f"reve_xgb_mi{suffix}.json"
    payload = {
        "data_dir": str(data_dir),
        "protocol": "raw_flat + SelectKBest(MI, REVE) + XGBoost",
        "mi_k": k,
        "neg_per_pos": args.neg_per_pos,
        "max_train_samples": None if args.max_train_samples <= 0 else args.max_train_samples,
        "n_raw_features": int(X_train_raw.shape[1]),
        "n_reve_features_selected": int(X_train_reve_sel.shape[1]),
        "n_reve_dims_total": int(reve_train_flat.shape[1]),
        "n_train_fit": int(len(train_idx_fit)),
        "n_test": int(len(test_idx)),
        "full_train": args.full_train,
        "random_state": args.random_state,
        "metrics": metrics,
        "train_patients": sorted(train_patients),
        "test_patients": sorted(test_patients),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"Total: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
