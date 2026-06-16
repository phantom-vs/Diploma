"""Riemannian geometry pipeline for new_format_data (memmap + metadata.pkl)."""

from .dataset import (
    DEFAULT_FEATURE_NAMES,
    NewFormatDataset,
    load_split,
    parse_feature_names,
    patient_train_test_split,
    save_split,
)
from .pipeline import HybridRiemannClassifier, build_riemann_pipeline, evaluate_binary

__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "NewFormatDataset",
    "patient_train_test_split",
    "save_split",
    "load_split",
    "parse_feature_names",
    "build_riemann_pipeline",
    "HybridRiemannClassifier",
    "evaluate_binary",
]
