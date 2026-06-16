#!/usr/bin/env python3
"""
Riemannian pipeline on new_format_data.

Patient-level 80/20 split (no patient leakage).
Example:
  python -m riemann_new_format.run --data-dir ../new_format_data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .dataset import (
    N_CHANNELS,
    N_SAMPLES,
    NewFormatDataset,
    feature_dim_map,
    indices_for_patients,
    labels_for_indices,
    load_split,
    parse_feature_names,
    patient_train_test_split,
    save_split,
    subsample_train_indices,
)
from .pipeline import (
    HybridRiemannClassifier,
    build_riemann_pipeline,
    evaluate_binary,
    predict_in_batches,
)


def load_train_batch(ds: NewFormatDataset, indices: np.ndarray, batch_size: int) -> np.ndarray:
    """Load train windows into one array without extra copies per chunk."""
    n = len(indices)
    out = np.empty((n, N_CHANNELS, N_SAMPLES), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_idx = indices[start:end]
        out[start:end] = ds.get_batch(chunk_idx)
        if start == 0 or (start // batch_size) % 50 == 0:
            print(f"    EEG loaded {end:,}/{n:,}", flush=True)
    return out


def load_features_batch(ds: NewFormatDataset, indices: np.ndarray, batch_size: int) -> np.ndarray:
    """Load stacked memmap features for train indices."""
    n = len(indices)
    n_feat = ds.n_features
    out = np.empty((n, n_feat), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_idx = indices[start:end]
        out[start:end] = ds.get_features_batch(chunk_idx)
        if start == 0 or (start // batch_size) % 50 == 0:
            print(f"    features loaded {end:,}/{n:,}", flush=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Riemannian DEPD detection on new_format_data")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "new_format_data",
        help="Path to new_format_data (metadata.pkl + raw_signals.memmap)",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--classifier",
        choices=["lr", "mdm", "xgb", "lgb", "catboost"],
        default="lr",
        help="lr/mdm on raw EEG; xgb/lgb/catboost require --features (hybrid)",
    )
    p.add_argument("--cov-estimator", default="lwf", help="Covariances estimator (scm, lwf, oas, ...)")
    p.add_argument(
        "--neg-per-pos",
        type=float,
        default=5.0,
        help="Train undersampling: negatives per positive window",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=200_000,
        help="Cap train windows after balancing (ignored with --full-train)",
    )
    p.add_argument(
        "--full-train",
        action="store_true",
        help="Use all train windows (no undersampling); ~935k windows, needs ~14 GB RAM",
    )
    p.add_argument("--predict-batch-size", type=int, default=512)
    p.add_argument("--train-batch-size", type=int, default=4096, help="Chunk size when loading train memmap")
    p.add_argument("--reuse-split", action="store_true", help="Load saved patient split JSON")
    p.add_argument(
        "--features",
        default=None,
        help="Extra memmap features: 'default', 'all', or comma list (e.g. morphology,hjorth,spectral)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not (data_dir / "metadata.pkl").is_file():
        print(f"metadata.pkl not found in {data_dir}", file=sys.stderr)
        return 1

    feature_names = parse_feature_names(args.features)
    hybrid_backends = {"xgb", "lgb", "catboost"}
    if args.classifier in hybrid_backends and not feature_names:
        print(f"--classifier {args.classifier} requires --features", file=sys.stderr)
        return 1
    if feature_names and args.classifier == "mdm":
        print("MDM does not support --features; use --classifier lr/xgb/lgb/catboost", file=sys.stderr)
        return 1

    print(f"Data: {data_dir}")
    t0 = time.perf_counter()
    ds = NewFormatDataset(data_dir, feature_names=feature_names)
    if feature_names:
        dims = feature_dim_map(data_dir, feature_names, ds.n_windows)
        dim_parts = ", ".join(f"{name}={dims[name]}" for name in feature_names)
        print(f"Extra features ({ds.n_features} total): {dim_parts}")
    else:
        print("Extra features: none (raw EEG only)")
    print(f"Windows: {ds.n_windows:,} | load meta+memmap: {time.perf_counter() - t0:.1f}s")

    if args.reuse_split and (data_dir / "splits" / "patient_split_80_20.json").is_file():
        train_patients, test_patients = load_split(data_dir)
        print("Using saved patient split")
    else:
        train_patients, test_patients = patient_train_test_split(
            ds.meta,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        split_path = save_split(
            data_dir,
            train_patients,
            test_patients,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Saved split: {split_path}")

    train_set = set(train_patients)
    test_set = set(test_patients)
    assert not train_set & test_set, "Patient leak between train and test"

    print(f"Patients train/test: {len(train_patients)} / {len(test_patients)}")

    train_idx = indices_for_patients(ds.meta, train_patients)
    test_idx = indices_for_patients(ds.meta, test_patients)
    y_train_full = labels_for_indices(ds.meta, train_idx)
    y_test = labels_for_indices(ds.meta, test_idx)

    if args.full_train:
        train_idx_fit = train_idx
        y_train = y_train_full
        print("Mode: FULL train set (no undersampling)")
    else:
        train_idx_fit = subsample_train_indices(
            train_idx,
            y_train_full,
            neg_per_pos=args.neg_per_pos,
            max_samples=args.max_train_samples,
            random_state=args.random_state,
        )
        y_train = labels_for_indices(ds.meta, train_idx_fit)

    print(
        f"Train windows: {len(train_idx):,} -> fit {len(train_idx_fit):,} "
        f"(DEPD {y_train.sum():,} / фон {(y_train == 0).sum():,})"
    )
    print(f"Test windows:  {len(test_idx):,} (DEPD {y_test.sum():,} / фон {(y_test == 0).sum():,})")

    print(f"\nLoading train ({len(train_idx_fit):,})...")
    t1 = time.perf_counter()
    X_train = load_train_batch(ds, train_idx_fit, args.train_batch_size)
    X_train_feat = None
    if feature_names:
        X_train_feat = load_features_batch(ds, train_idx_fit, args.train_batch_size)
    print(
        f"  X_train {X_train.shape}"
        + (f" | X_feat {X_train_feat.shape}" if X_train_feat is not None else "")
        + f" | {time.perf_counter() - t1:.1f}s"
    )

    if feature_names or args.classifier in hybrid_backends:
        backend = args.classifier if args.classifier in hybrid_backends else "lr"
        model = HybridRiemannClassifier(
            cov_estimator=args.cov_estimator,
            backend=backend,
            random_state=args.random_state,
        )
        fit_label = f"Hybrid Riemann+features ({backend})"
    else:
        model = build_riemann_pipeline(
            classifier=args.classifier,
            cov_estimator=args.cov_estimator,
            random_state=args.random_state,
        )
        fit_label = args.classifier.upper()

    print(f"\nFitting {fit_label} (cov={args.cov_estimator})...")
    t2 = time.perf_counter()
    if feature_names or args.classifier in hybrid_backends:
        model.fit(X_train, y_train, X_feat=X_train_feat)
    else:
        model.fit(X_train, y_train)
    print(f"  fit done in {time.perf_counter() - t2:.1f}s")

    print("\nPredicting test (batched)...")
    t3 = time.perf_counter()
    y_pred = predict_in_batches(
        model,
        ds._raw,
        test_idx,
        batch_size=args.predict_batch_size,
        get_features_batch=ds.get_features_batch if (feature_names or args.classifier in hybrid_backends) else None,
    )
    print(f"  predict done in {time.perf_counter() - t3:.1f}s")

    metrics = evaluate_binary(y_test, y_pred)
    print("\n=== Test metrics (has_depd binary) ===")
    for k in ("f1", "precision", "recall", "specificity", "accuracy"):
        print(f"  {k:14}: {metrics[k]:.4f}")

    results = {
        "data_dir": str(data_dir),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "classifier": args.classifier,
        "cov_estimator": args.cov_estimator,
        "features": list(feature_names),
        "n_extra_features": int(ds.n_features),
        "full_train": args.full_train,
        "n_train_patients": len(train_patients),
        "n_test_patients": len(test_patients),
        "n_train_fit": int(len(train_idx_fit)),
        "n_test": int(len(test_idx)),
        "metrics": metrics,
        "train_patients": sorted(train_patients),
        "test_patients": sorted(test_patients),
    }
    suffix = "_full" if args.full_train else ""
    feat_suffix = "_feat" if feature_names else ""
    out_path = data_dir / "splits" / f"riemann_results_{args.classifier}{feat_suffix}{suffix}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
