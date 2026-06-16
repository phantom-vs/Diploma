#!/usr/bin/env python3
"""Per-patient test metrics for the Riemannian DEPD pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from .dataset import (
    NewFormatDataset,
    indices_for_patients,
    labels_for_indices,
    load_split,
    parse_feature_names,
    patient_ids_for_indices,
)
from .pipeline import (
    HybridRiemannClassifier,
    build_riemann_pipeline,
    evaluate_binary,
    evaluate_per_patient,
    macro_average_per_patient,
    predict_in_batches,
)
from .run import load_features_batch, load_train_batch


def default_model_path(
    data_dir: Path,
    classifier: str,
    full_train: bool,
    *,
    with_features: bool,
) -> Path:
    suffix = "_full" if full_train else ""
    feat_suffix = "_feat" if with_features else ""
    return data_dir / "splits" / f"riemann_model_{classifier}{feat_suffix}{suffix}.joblib"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-patient metrics on test split")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "new_format_data")
    p.add_argument("--classifier", choices=["lr", "mdm"], default="lr")
    p.add_argument("--cov-estimator", default="lwf")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--full-train", action="store_true", help="Train on all train windows (matches full run)")
    p.add_argument("--neg-per-pos", type=float, default=5.0)
    p.add_argument("--max-train-samples", type=int, default=200_000)
    p.add_argument("--train-batch-size", type=int, default=4096)
    p.add_argument("--predict-batch-size", type=int, default=512)
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Load/save fitted pipeline (joblib). Default: splits/riemann_model_{clf}[_full].joblib",
    )
    p.add_argument("--force-retrain", action="store_true", help="Ignore saved model and retrain")
    p.add_argument("--features", default=None, help="Extra memmap features: default, all, or comma list")
    return p.parse_args()


def print_per_patient_table(per_patient: dict[str, dict[str, float]]) -> None:
    header = (
        f"{'patient':<22} {'n':>7} {'DEPD':>6} {'rate':>6} "
        f"{'F1':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'Acc':>6}"
    )
    print(header)
    print("-" * len(header))
    for pid, m in per_patient.items():
        print(
            f"{pid:<22} {int(m['n_windows']):>7,} {int(m['n_depd']):>6,} {m['depd_rate']:>6.1%} "
            f"{m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['specificity']:>6.3f} {m['accuracy']:>6.3f}"
        )


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not (data_dir / "metadata.pkl").is_file():
        print(f"metadata.pkl not found in {data_dir}", file=sys.stderr)
        return 1

    feature_names = parse_feature_names(args.features)
    model_path = args.model_path or default_model_path(
        data_dir,
        args.classifier,
        args.full_train,
        with_features=bool(feature_names),
    )

    print(f"Data: {data_dir}")
    t0 = time.perf_counter()
    ds = NewFormatDataset(data_dir, feature_names=feature_names)

    train_patients, test_patients = load_split(data_dir)
    train_idx = indices_for_patients(ds.meta, train_patients)
    test_idx = indices_for_patients(ds.meta, test_patients)
    y_test = labels_for_indices(ds.meta, test_idx)
    patient_ids = patient_ids_for_indices(ds.meta, test_idx)

    if model_path.is_file() and not args.force_retrain:
        print(f"Loading model: {model_path}")
        model = joblib.load(model_path)
    else:
        from .dataset import subsample_train_indices

        y_train_full = labels_for_indices(ds.meta, train_idx)
        if args.full_train:
            train_idx_fit = train_idx
            y_train = y_train_full
            print("Training: FULL train set")
        else:
            train_idx_fit = subsample_train_indices(
                train_idx,
                y_train_full,
                neg_per_pos=args.neg_per_pos,
                max_samples=args.max_train_samples,
                random_state=args.random_state,
            )
            y_train = labels_for_indices(ds.meta, train_idx_fit)
            print(f"Training: subsampled {len(train_idx_fit):,} windows")

        print(f"Loading train ({len(train_idx_fit):,})...")
        X_train = load_train_batch(ds, train_idx_fit, args.train_batch_size)
        X_train_feat = load_features_batch(ds, train_idx_fit, args.train_batch_size) if feature_names else None
        if feature_names:
            model = HybridRiemannClassifier(
                cov_estimator=args.cov_estimator,
                random_state=args.random_state,
            )
        else:
            model = build_riemann_pipeline(
                classifier=args.classifier,
                cov_estimator=args.cov_estimator,
                random_state=args.random_state,
            )
        print("Fitting...")
        if feature_names:
            model.fit(X_train, y_train, X_feat=X_train_feat)
        else:
            model.fit(X_train, y_train)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

    print(f"\nPredicting test ({len(test_idx):,} windows)...")
    y_pred = predict_in_batches(
        model,
        ds._raw,
        test_idx,
        batch_size=args.predict_batch_size,
        get_features_batch=ds.get_features_batch if feature_names else None,
    )

    global_metrics = evaluate_binary(y_test, y_pred)
    per_patient = evaluate_per_patient(y_test, y_pred, patient_ids)
    macro = macro_average_per_patient(per_patient)

    print("\n=== Global test metrics (micro, all windows) ===")
    for k in ("f1", "precision", "recall", "specificity", "accuracy"):
        print(f"  {k:14}: {global_metrics[k]:.4f}")

    print("\n=== Macro average (mean over patients, unweighted) ===")
    for k in ("f1", "precision", "recall", "specificity", "accuracy"):
        print(f"  {k:14}: {macro[k]:.4f}")

    print(f"\n=== Per-patient metrics ({len(per_patient)} patients) ===")
    print_per_patient_table(per_patient)

    suffix = "_full" if args.full_train else ""
    feat_suffix = "_feat" if feature_names else ""
    out_path = data_dir / "splits" / f"riemann_per_patient_{args.classifier}{feat_suffix}{suffix}.json"
    payload = {
        "data_dir": str(data_dir),
        "classifier": args.classifier,
        "cov_estimator": args.cov_estimator,
        "features": list(feature_names),
        "n_extra_features": int(ds.n_features),
        "full_train": args.full_train,
        "model_path": str(model_path),
        "n_test_patients": len(test_patients),
        "n_test_windows": int(len(test_idx)),
        "global_metrics": global_metrics,
        "macro_average": macro,
        "per_patient": per_patient,
        "test_patients": sorted(test_patients),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
