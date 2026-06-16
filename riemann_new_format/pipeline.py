"""PyRiemann sklearn pipelines and metrics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ClassifierName = Literal["mdm", "lr"]
HybridBackend = Literal["lr", "xgb", "lgb", "catboost"]


def _pos_scale_weight(y: np.ndarray) -> float:
    n_pos = max(1, int(np.sum(y == 1)))
    n_neg = max(1, int(np.sum(y == 0)))
    return n_neg / n_pos


def build_tabular_classifier(
    backend: HybridBackend,
    y: np.ndarray,
    *,
    random_state: int = 42,
) -> Any:
    """Classifier on stacked Riemann tangent + memmap features."""
    spw = _pos_scale_weight(y)
    if backend == "lr":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
    if backend == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
    if backend == "lgb":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        )
    if backend == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_state=random_state,
            verbose=False,
        )
    raise ValueError(f"Unknown backend: {backend}")


class HybridRiemannClassifier(BaseEstimator, ClassifierMixin):
    """
    Covariances -> TangentSpace on raw EEG, concatenated with tabular memmap features.
    """

    def __init__(
        self,
        cov_estimator: str = "lwf",
        backend: HybridBackend = "lr",
        random_state: int = 42,
        select_k: int | None = None,
    ):
        self.cov_estimator = cov_estimator
        self.backend = backend
        self.random_state = random_state
        self.select_k = select_k

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_feat: np.ndarray | None = None,
    ) -> "HybridRiemannClassifier":
        self._riemann = Pipeline([
            ("cov", Covariances(estimator=self.cov_estimator)),
            ("ts", TangentSpace(metric="riemann")),
        ])
        z_riemann = self._riemann.fit_transform(X, y)
        self._n_riemann_ = z_riemann.shape[1]
        if X_feat is not None:
            self._feat_imputer = SimpleImputer(strategy="median")
            x_feat = self._feat_imputer.fit_transform(np.asarray(X_feat, dtype=np.float32))
            z = np.hstack([z_riemann, x_feat])
            self._n_extra_ = x_feat.shape[1]
        else:
            self._feat_imputer = None
            z = z_riemann
            self._n_extra_ = 0

        self._scaler = StandardScaler()
        z = self._scaler.fit_transform(z)
        self._selector = None
        if self.select_k is not None and self.select_k > 0:
            k = min(int(self.select_k), z.shape[1])
            self._selector = SelectKBest(mutual_info_classif, k=k)
            z = self._selector.fit_transform(z, y)
        self._clf = build_tabular_classifier(self.backend, y, random_state=self.random_state)
        self._clf.fit(z, y)
        return self

    def _to_classifier_input(self, X: np.ndarray, X_feat: np.ndarray | None) -> np.ndarray:
        z = self._scaler.transform(self._stack(X, X_feat))
        if self._selector is not None:
            z = self._selector.transform(z)
        return z

    def transform_features(self, X: np.ndarray, X_feat: np.ndarray | None = None) -> np.ndarray:
        """Return scaled hybrid feature matrix."""
        return self._to_classifier_input(X, X_feat)

    def _stack(self, X: np.ndarray, X_feat: np.ndarray | None) -> np.ndarray:
        z_riemann = self._riemann.transform(X)
        if self._n_extra_ == 0:
            return z_riemann
        if X_feat is None:
            raise ValueError("Model was trained with extra features; X_feat is required at predict time")
        x_feat = self._feat_imputer.transform(np.asarray(X_feat, dtype=np.float32))
        return np.hstack([z_riemann, x_feat])

    def predict(self, X: np.ndarray, X_feat: np.ndarray | None = None) -> np.ndarray:
        return self._clf.predict(self._to_classifier_input(X, X_feat))

    def predict_proba(self, X: np.ndarray, X_feat: np.ndarray | None = None) -> np.ndarray:
        return self._clf.predict_proba(self._to_classifier_input(X, X_feat))


def build_riemann_pipeline(
    classifier: ClassifierName = "lr",
    cov_estimator: str = "lwf",
    random_state: int = 42,
) -> Pipeline | MDM:
    """
  cov_estimator: 'scm', 'lwf', 'oas', ...
  classifier: 'mdm' (Riemannian MDM) or 'lr' (TangentSpace + LogisticRegression)
    """
    cov = Covariances(estimator=cov_estimator)

    if classifier == "mdm":
        return Pipeline([
            ("cov", cov),
            ("mdm", MDM(metric="riemann")),
        ])

    return Pipeline([
        ("cov", cov),
        ("ts", TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            ),
        ),
    ])


def evaluate_per_patient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Binary metrics per patient (external_id)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    patient_ids = np.asarray(patient_ids)

    out: dict[str, dict[str, float]] = {}
    for pid in sorted(np.unique(patient_ids), key=str):
        mask = patient_ids == pid
        yt = y_true[mask]
        yp = y_pred[mask]
        m = evaluate_binary(yt, yp)
        m["n_windows"] = float(len(yt))
        m["n_depd"] = float(yt.sum())
        m["n_background"] = float((yt == 0).sum())
        m["depd_rate"] = float(yt.mean()) if len(yt) else 0.0
        out[str(pid)] = m
    return out


def macro_average_per_patient(per_patient: dict[str, dict[str, float]]) -> dict[str, float]:
    """Unweighted mean of per-patient metrics (each patient counts equally)."""
    if not per_patient:
        return {}
    keys = ("f1", "precision", "recall", "specificity", "accuracy")
    return {
        k: float(np.mean([m[k] for m in per_patient.values()]))
        for k in keys
    }


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(spec),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def predict_in_batches(
    model: Any,
    X_memmap: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 512,
    reshape: tuple[int, int] = (18, 200),
    get_features_batch: Any | None = None,
) -> np.ndarray:
    """Predict on index slices without loading full array."""
    preds = []
    n_ch, n_t = reshape
    use_hybrid = isinstance(model, HybridRiemannClassifier)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        flat = np.asarray(X_memmap[batch_idx], dtype=np.float32)
        Xb = flat.reshape(len(batch_idx), n_ch, n_t)
        if use_hybrid:
            X_feat = get_features_batch(batch_idx) if get_features_batch is not None else None
            preds.append(model.predict(Xb, X_feat=X_feat))
        else:
            preds.append(model.predict(Xb))
    return np.concatenate(preds, axis=0)
