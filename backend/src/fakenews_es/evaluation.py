"""Métricas compartidas para baseline y futuros modelos."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .data import FAKE_LABEL, TRUE_LABEL


def evaluate_binary(y_true: Any, y_pred: Any) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[TRUE_LABEL, FAKE_LABEL])
    if matrix.shape != (2, 2):
        raise ValueError(f"Matriz de confusión inesperada: {matrix.shape}")
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=FAKE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=FAKE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=FAKE_LABEL, zero_division=0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "positive_class": "Fake",
        "positive_label": FAKE_LABEL,
        "label_order": ["True", "Fake"],
        "zero_division": 0,
    }

