"""Multi-label metrics. Mean ROC-AUC is the headline NIH / CheXNet number."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from cxr.constants import PATHOLOGIES


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.min() == y_true.max():
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    per_class = []
    aucs = []
    aps = []
    for i, name in enumerate(PATHOLOGIES):
        auc = _safe_auc(y_true[:, i], y_score[:, i])
        ap = _safe_ap(y_true[:, i], y_score[:, i])
        pred = (y_score[:, i] >= threshold).astype(np.float32)
        tp = float(((pred == 1) & (y_true[:, i] == 1)).sum())
        fp = float(((pred == 1) & (y_true[:, i] == 0)).sum())
        fn = float(((pred == 0) & (y_true[:, i] == 1)).sum())
        tn = float(((pred == 0) & (y_true[:, i] == 0)).sum())
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        row = {
            "pathology": name,
            "roc_auc": auc,
            "pr_auc": ap,
            "sensitivity": sens,
            "specificity": spec,
            "support": int(y_true[:, i].sum()),
        }
        per_class.append(row)
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(ap):
            aps.append(ap)

    return {
        "mean_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "mean_pr_auc": float(np.mean(aps)) if aps else float("nan"),
        "per_class": per_class,
        "n_samples": int(y_true.shape[0]),
    }


def youden_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> list[float]:
    """Per-class threshold that maximises sensitivity + specificity - 1."""
    thresholds = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].min() == y_true[:, i].max():
            thresholds.append(0.5)
            continue
        fpr, tpr, thr = roc_curve(y_true[:, i], y_score[:, i])
        j = tpr - fpr
        thresholds.append(float(thr[int(np.argmax(j))]))
    return thresholds


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def format_metrics_table(metrics: dict) -> str:
    lines = [
        f"n = {metrics['n_samples']}",
        f"mean ROC-AUC = {metrics['mean_roc_auc']:.4f}",
        f"mean PR-AUC  = {metrics['mean_pr_auc']:.4f}",
        "",
        f"{'pathology':22s} {'AUC':>7s} {'PR-AUC':>7s} {'sens':>7s} {'spec':>7s} {'n+':>6s}",
        "-" * 62,
    ]
    for row in metrics["per_class"]:
        auc = row["roc_auc"]
        ap = row["pr_auc"]
        auc_s = f"{auc:7.3f}" if not np.isnan(auc) else "    n/a"
        ap_s = f"{ap:7.3f}" if not np.isnan(ap) else "    n/a"
        lines.append(
            f"{row['pathology']:22s} {auc_s} {ap_s} {row['sensitivity']:7.3f} "
            f"{row['specificity']:7.3f} {row['support']:6d}"
        )
    return "\n".join(lines)
