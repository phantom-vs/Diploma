#!/usr/bin/env python3
"""Train and compare LR / XGBoost / LightGBM / CatBoost on hybrid Riemann+features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .dataset import (
    NewFormatDataset,
    feature_dim_map,
    indices_for_patients,
    labels_for_indices,
    load_split,
    parse_feature_names,
    subsample_train_indices,
)
from .pipeline import (
    HybridRiemannClassifier,
    build_tabular_classifier,
    evaluate_binary,
)
from .run import load_features_batch, load_train_batch


BOOSTERS = ("lr", "xgb", "lgb", "catboost")


def transform_test_features(
    prep: HybridRiemannClassifier,
    ds: NewFormatDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    get_feat = ds.get_features_batch if ds.feature_names else None
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        x_eeg = ds.get_batch(batch_idx)
        x_feat = get_feat(batch_idx) if get_feat is not None else None
        chunks.append(prep.transform_features(x_eeg, x_feat))
    return np.vstack(chunks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare boosters on hybrid Riemann+features")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "new_format_data")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cov-estimator", default="lwf")
    p.add_argument("--features", default="default")
    p.add_argument("--neg-per-pos", type=float, default=5.0)
    p.add_argument("--max-train-samples", type=int, default=200_000)
    p.add_argument("--full-train", action="store_true")
    p.add_argument("--train-batch-size", type=int, default=4096)
    p.add_argument("--predict-batch-size", type=int, default=512)
    p.add_argument(
        "--models",
        default=",".join(BOOSTERS),
        help=f"Comma-separated backends: {','.join(BOOSTERS)}",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not (data_dir / "metadata.pkl").is_file():
        print(f"metadata.pkl not found in {data_dir}", file=sys.stderr)
        return 1

    backends = tuple(b.strip() for b in args.models.split(",") if b.strip())
    bad = set(backends) - set(BOOSTERS)
    if bad:
        print(f"Unknown models: {sorted(bad)}", file=sys.stderr)
        return 1

    feature_names = parse_feature_names(args.features)
    if not feature_names:
        print("Boosters require --features (use 'default')", file=sys.stderr)
        return 1

    print(f"Data: {data_dir}")
    t0 = time.perf_counter()
    ds = NewFormatDataset(data_dir, feature_names=feature_names)
    dims = feature_dim_map(data_dir, feature_names, ds.n_windows)
    print(f"Features ({ds.n_features}): {', '.join(f'{k}={dims[k]}' for k in feature_names)}")

    train_patients, test_patients = load_split(data_dir)
    train_idx = indices_for_patients(ds.meta, train_patients)
    test_idx = indices_for_patients(ds.meta, test_patients)
    y_train_full = labels_for_indices(ds.meta, train_idx)
    y_test = labels_for_indices(ds.meta, test_idx)

    if args.full_train:
        train_idx_fit = train_idx
        y_train = y_train_full
        print("Mode: FULL train")
    else:
        train_idx_fit = subsample_train_indices(
            train_idx,
            y_train_full,
            neg_per_pos=args.neg_per_pos,
            max_samples=args.max_train_samples,
            random_state=args.random_state,
        )
        y_train = labels_for_indices(ds.meta, train_idx_fit)
        print(f"Mode: subsampled train {len(train_idx_fit):,}")

    print(f"Train fit: {len(train_idx_fit):,} | Test: {len(test_idx):,}")

    print("\nLoading train...")
    X_train = load_train_batch(ds, train_idx_fit, args.train_batch_size)
    X_train_feat = load_features_batch(ds, train_idx_fit, args.train_batch_size)

    print("\nFitting Riemann preprocessing (once)...")
    t_prep = time.perf_counter()
    prep = HybridRiemannClassifier(
        cov_estimator=args.cov_estimator,
        backend="lr",
        random_state=args.random_state,
    )
    prep.fit(X_train, y_train, X_feat=X_train_feat)
    Z_train = prep.transform_features(X_train, X_train_feat)
    print(f"  hybrid dim={Z_train.shape[1]} | prep {time.perf_counter() - t_prep:.1f}s")

    print("\nTransforming test...")
    t_te = time.perf_counter()
    Z_test = transform_test_features(
        prep,
        ds,
        test_idx,
        batch_size=args.predict_batch_size,
    )
    print(f"  Z_test {Z_test.shape} | {time.perf_counter() - t_te:.1f}s")

    all_results: dict[str, dict] = {}
    print("\n" + "=" * 72)
    print(f"{'model':<12} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8} {'Acc':>8} {'fit_s':>8}")
    print("-" * 72)

    for backend in backends:
        t_fit = time.perf_counter()
        clf = build_tabular_classifier(backend, y_train, random_state=args.random_state)
        clf.fit(Z_train, y_train)
        fit_s = time.perf_counter() - t_fit
        y_pred = clf.predict(Z_test)
        metrics = evaluate_binary(y_test, y_pred)
        all_results[backend] = {
            "metrics": metrics,
            "fit_seconds": fit_s,
            "hybrid_dim": int(Z_train.shape[1]),
        }
        print(
            f"{backend:<12} {metrics['f1']:8.4f} {metrics['precision']:8.4f} "
            f"{metrics['recall']:8.4f} {metrics['specificity']:8.4f} "
            f"{metrics['accuracy']:8.4f} {fit_s:8.1f}"
        )

    suffix = "_full" if args.full_train else ""
    out_path = data_dir / "splits" / f"booster_comparison_feat{suffix}.json"
    payload = {
        "data_dir": str(data_dir),
        "features": list(feature_names),
        "n_extra_features": ds.n_features,
        "hybrid_dim": int(Z_train.shape[1]),
        "n_train_fit": int(len(train_idx_fit)),
        "n_test": int(len(test_idx)),
        "full_train": args.full_train,
        "cov_estimator": args.cov_estimator,
        "random_state": args.random_state,
        "models": all_results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    best = max(all_results.items(), key=lambda kv: kv[1]["metrics"]["f1"])
    print("=" * 72)
    print(f"Best: {best[0]} (F1={best[1]['metrics']['f1']:.4f})")
    print(f"Saved: {out_path}")
    print(f"Total: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
