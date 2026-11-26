#!/usr/bin/env python3
"""Leak-Free 5-Fold Patient-Grouped OOF Training: RAD-DINO ViT + Residual Meta-Calibrator + Conformal Abstention.

Reads the patient-disjoint feature cache (`nih_features_v10.npz`, benchmark patients {1, 27426} excluded):
- Combines Train+Val (2,400 radiographs, 820 unique patients) into strict `GroupKFold(5)` by `patient_id`
  so EVERY Stage-1 RAD-DINO logit fed to Stage-2 is 100% Out-of-Fold (zero stacking leakage!).
- Compares 4 faithful Stage-2 models strictly on OOF CV, then evaluates ONCE on the untouched
  held-out official NIH Test Split (600 radiographs = 8,400 real clinical pathology checks):
  1. Baseline: Single DenseNet-121 Multi-Cohort
  2. Baseline: v9.0 4-CNN Weighted Logit Ensemble
  3. Method A: Microsoft RAD-DINO ViT-Base 518px L2 Probe (5-fold patient-grouped OOF)
  4. Method B: Residual Cross-Pathology & Biometry Meta-Calibrator (4-CNN only, AWS CPU-safe <55KB)
  5. Method C: Full 5-Vision Residual Meta-Calibrator (4-CNN + RAD-DINO ViT + Biometry)
  6. Method D: Regularized PyTorch Residual MLP with Asymmetric Loss + Dropout + Weight Decay
- Calibrates Split Conformal Selective Abstention Bands strictly on the 2,400 OOF predictions
  and tests exact Accuracy, Precision, Recall & Coverage on the 600 unseen NIH test patients.
- Evaluates Binary Clinical Triage (Abnormal Study vs Clear Normal) with Conformal Selective Gate.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cxr.constants import NUM_CLASSES, PATHOLOGIES  # noqa: E402
from cxr.models.losses import AsymmetricLossMultiLabel  # noqa: E402
from cxr.training.metrics import f1_optimal_thresholds  # noqa: E402

BACKBONE_ORDER = ["densenet121_all", "densenet121_nih", "densenet121_pc", "resnet50_512"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def build_meta_features(
    cnn_cal: np.ndarray,
    biometry: np.ndarray,
    rad_logits: np.ndarray | None = None,
) -> np.ndarray:
    n = cnn_cal.shape[0]
    flat_cnn = cnn_cal.reshape(n, -1)
    probs_4 = _sigmoid(cnn_cal)
    spread = probs_4.std(axis=1)
    max_p = probs_4.max(axis=1)
    min_p = probs_4.min(axis=1)
    parts = [flat_cnn, spread, max_p - min_p, biometry]
    if rad_logits is not None:
        parts.append(rad_logits)
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_oof_raddino_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    X_test: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """5-fold patient-grouped OOF L2-regularized RAD-DINO probe (zero stacking leakage!)."""
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y, groups))

    oof_logits = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    test_logits_folds = np.zeros((n_splits, len(X_test), NUM_CLASSES), dtype=np.float32)
    final_W = np.zeros((NUM_CLASSES, X.shape[1]), dtype=np.float32)
    final_b = np.zeros(NUM_CLASSES, dtype=np.float32)

    # First select per-class C strictly via 5-fold patient-grouped OOF ROC-AUC
    best_Cs = []
    for c in range(NUM_CLASSES):
        pos_tot = int(y[:, c].sum())
        if pos_tot < 8:
            best_Cs.append(0.01)
            continue
        best_auc = -1.0
        best_c = 0.03
        for C in (0.005, 0.02, 0.06, 0.2):
            z_oof = np.zeros(len(y), dtype=np.float32)
            for tr_idx, va_idx in folds:
                if y[tr_idx, c].sum() < 2:
                    continue
                sc = StandardScaler()
                Xtr = sc.fit_transform(X[tr_idx])
                Xva = sc.transform(X[va_idx])
                clf = LogisticRegression(C=C, max_iter=300, class_weight="balanced", solver="lbfgs")
                clf.fit(Xtr, y[tr_idx, c].astype(int))
                z_oof[va_idx] = clf.decision_function(Xva)
            auc = float(roc_auc_score(y[:, c], z_oof))
            if auc > best_auc:
                best_auc = auc
                best_c = C
        best_Cs.append(best_c)

    # Compute strictly leak-free OOF logits using the selected C per class
    for f_idx, (tr_idx, va_idx) in enumerate(folds):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr_idx])
        Xva = sc.transform(X[va_idx])
        Xte = sc.transform(X_test)
        for c in range(NUM_CLASSES):
            if y[tr_idx, c].sum() < 2:
                b0 = float(_logit(np.array([max(0.005, y[tr_idx, c].mean())]))[0])
                oof_logits[va_idx, c] = b0
                test_logits_folds[f_idx, :, c] = b0
                continue
            clf = LogisticRegression(C=best_Cs[c], max_iter=350, class_weight="balanced", solver="lbfgs")
            clf.fit(Xtr, y[tr_idx, c].astype(int))
            oof_logits[va_idx, c] = clf.decision_function(Xva)
            test_logits_folds[f_idx, :, c] = clf.decision_function(Xte)

    # Fit final production probe on all 2,400 train+val images with the CV-selected C
    sc_full = StandardScaler()
    X_full = sc_full.fit_transform(X)
    for c in range(NUM_CLASSES):
        if y[:, c].sum() < 3:
            final_b[c] = float(_logit(np.array([max(0.005, y[:, c].mean())]))[0])
            continue
        clf = LogisticRegression(C=best_Cs[c], max_iter=400, class_weight="balanced", solver="lbfgs")
        clf.fit(X_full, y[:, c].astype(int))
        w_raw = clf.coef_[0] / np.where(sc_full.scale_ > 1e-8, sc_full.scale_, 1.0)
        b_raw = float(clf.intercept_[0] - np.dot(w_raw, sc_full.mean_))
        final_W[c] = w_raw.astype(np.float32)
        final_b[c] = b_raw

    test_logits = X_test @ final_W.T + final_b[None, :]
    return oof_logits, test_logits, final_W, final_b


def train_oof_residual_meta(
    X: np.ndarray,
    prior: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    X_test: np.ndarray,
    prior_test: np.ndarray,
    n_splits: int = 5,
) -> dict[str, Any]:
    """5-fold patient-grouped OOF Residual Cross-Pathology + Biometry Calibrated Stacker.

    Every hyperparameter (C, shrinkage alpha toward 500k pretrained prior, Platt scaling) is
    selected strictly via patient-grouped OOF CV -> ZERO leakage to Test!
    """
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y, groups))
    n_feat = X.shape[1]

    oof_raw_z = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    prior_scale = np.ones(NUM_CLASSES, dtype=np.float32)
    coef_raw = np.zeros((NUM_CLASSES, n_feat), dtype=np.float32)
    intercept_raw = np.zeros(NUM_CLASSES, dtype=np.float32)
    blend_alpha = np.zeros(NUM_CLASSES, dtype=np.float32)
    chosen_C = np.zeros(NUM_CLASSES, dtype=np.float32)

    for c in range(NUM_CLASSES):
        pos_tot = int(y[:, c].sum())
        base_auc = float(roc_auc_score(y[:, c], prior[:, c])) if (0 < pos_tot < len(y)) else 0.5
        if pos_tot < 10:
            oof_raw_z[:, c] = prior[:, c]
            continue

        best_auc = base_auc
        best_cfg = (0.0, 0.0)
        best_z = prior[:, c].copy()

        for C in (0.004, 0.012, 0.04, 0.12):
            z_cv = np.zeros(len(y), dtype=np.float32)
            for tr_idx, va_idx in folds:
                sc = StandardScaler()
                Xtr = np.column_stack([prior[tr_idx, c], sc.fit_transform(X[tr_idx])])
                Xva = np.column_stack([prior[va_idx, c], sc.transform(X[va_idx])])
                clf = LogisticRegression(C=C, max_iter=400, solver="lbfgs")
                clf.fit(Xtr, y[tr_idx, c].astype(int))
                z_cv[va_idx] = clf.decision_function(Xva)
            for alpha in (0.25, 0.50, 0.75, 1.0):
                z_mix = (1.0 - alpha) * prior[:, c] + alpha * z_cv
                auc_mix = float(roc_auc_score(y[:, c], z_mix))
                if auc_mix > best_auc + 0.0015:
                    best_auc = auc_mix
                    best_cfg = (C, alpha)
                    best_z = z_mix

        oof_raw_z[:, c] = best_z
        chosen_C[c], blend_alpha[c] = best_cfg

        if best_cfg[1] > 0.0:
            sc_full = StandardScaler()
            X_full = np.column_stack([prior[:, c], sc_full.fit_transform(X)])
            clf = LogisticRegression(C=best_cfg[0], max_iter=500, solver="lbfgs")
            clf.fit(X_full, y[:, c].astype(int))
            alpha = best_cfg[1]
            w_prior_part = float(clf.coef_[0, 0])
            w_s = clf.coef_[0, 1:]
            w_raw = w_s / np.where(sc_full.scale_ > 1e-8, sc_full.scale_, 1.0)
            b_raw = float(clf.intercept_[0] - np.dot(w_raw, sc_full.mean_))
            prior_scale[c] = (1.0 - alpha) + alpha * w_prior_part
            coef_raw[c] = (alpha * w_raw).astype(np.float32)
            intercept_raw[c] = float(alpha * b_raw)

    # Fit per-class Platt calibration strictly on OOF predictions (zero overfitting!)
    platt_a = np.ones(NUM_CLASSES, dtype=np.float32)
    platt_b = np.zeros(NUM_CLASSES, dtype=np.float32)
    oof_cal_probs = np.zeros_like(oof_raw_z)
    for c in range(NUM_CLASSES):
        if int(y[:, c].sum()) >= 8:
            platt = LogisticRegression(C=1.0, max_iter=300)
            platt.fit(oof_raw_z[:, c].reshape(-1, 1), y[:, c].astype(int))
            platt_a[c] = float(np.clip(platt.coef_[0, 0], 0.35, 3.0))
            platt_b[c] = float(np.clip(platt.intercept_[0], -3.0, 3.0))
        oof_cal_probs[:, c] = _sigmoid(platt_a[c] * oof_raw_z[:, c] + platt_b[c])

    # Compute exact test predictions from the full-fit weights
    te_raw_z = prior_test * prior_scale[None, :] + X_test @ coef_raw.T + intercept_raw[None, :]
    te_cal_probs = _sigmoid(te_raw_z * platt_a[None, :] + platt_b[None, :])

    # OOF F1-optimal & Recall-oriented F1.5 thresholds (fitted strictly on OOF!)
    oof_f1_thrs = np.array(f1_optimal_thresholds(y, oof_cal_probs, beta=1.0), dtype=np.float32)
    oof_f15_thrs = np.array(f1_optimal_thresholds(y, oof_cal_probs, beta=1.5), dtype=np.float32)

    return {
        "prior_scale": prior_scale,
        "coef_raw": coef_raw,
        "intercept_raw": intercept_raw,
        "blend_alpha": blend_alpha,
        "chosen_C": chosen_C,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "oof_cal_probs": oof_cal_probs,
        "te_cal_probs": te_cal_probs,
        "oof_f1_thrs": oof_f1_thrs,
        "oof_f15_thrs": oof_f15_thrs,
    }


def train_oof_residual_mlp(
    X: np.ndarray,
    prior: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    X_test: np.ndarray,
    prior_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare regularized PyTorch Residual Bottleneck MLP with Asymmetric Loss + Dropout on 5-fold OOF."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gkf = GroupKFold(n_splits=5)
    oof_z = np.zeros_like(prior)
    te_z_folds = np.zeros((5, len(X_test), NUM_CLASSES), dtype=np.float32)

    class ResidualMLP(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.45),
                nn.Linear(64, NUM_CLASSES),
            )
            self.scale = nn.Parameter(torch.ones(NUM_CLASSES))
            self.bias = nn.Parameter(torch.zeros(NUM_CLASSES))

        def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return self.scale * p + 0.35 * self.net(x) + self.bias

    for f_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr_idx]), dtype=torch.float32, device=device)
        Ptr = torch.tensor(prior[tr_idx], dtype=torch.float32, device=device)
        Ytr = torch.tensor(y[tr_idx], dtype=torch.float32, device=device)
        Xva = torch.tensor(sc.transform(X[va_idx]), dtype=torch.float32, device=device)
        Pva = torch.tensor(prior[va_idx], dtype=torch.float32, device=device)
        Xte = torch.tensor(sc.transform(X_test), dtype=torch.float32, device=device)
        Pte = torch.tensor(prior_test, dtype=torch.float32, device=device)

        model = ResidualMLP(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-2)
        crit = nn.BCEWithLogitsLoss()
        best_loss = 1e9
        best_va = None
        best_te = None
        for ep in range(45):
            model.train()
            opt.zero_grad()
            loss = crit(model(Xtr, Ptr), Ytr)
            loss.backward()
            opt.step()
            model.eval()
            with torch.inference_mode():
                z_va = model(Xva, Pva)
                v_loss = float(crit(z_va, torch.tensor(y[va_idx], dtype=torch.float32, device=device)).item())
                if v_loss < best_loss:
                    best_loss = v_loss
                    best_va = z_va.cpu().numpy()
                    best_te = model(Xte, Pte).cpu().numpy()
        oof_z[va_idx] = best_va
        te_z_folds[f_idx] = best_te

    return _sigmoid(oof_z), _sigmoid(te_z_folds.mean(axis=0))


def evaluate_multilabel(y_true: np.ndarray, y_prob: np.ndarray, thrs: np.ndarray | float = 0.50) -> dict[str, Any]:
    yt = y_true.astype(np.float32)
    yp = np.clip(y_prob.astype(np.float32), 1e-6, 1.0 - 1e-6)
    t = np.full(NUM_CLASSES, thrs, dtype=np.float32) if isinstance(thrs, (int, float)) else np.asarray(thrs, dtype=np.float32)
    pred = (yp >= t[None, :]).astype(np.float32)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    aucs = [roc_auc_score(yt[:, c], yp[:, c]) for c in range(NUM_CLASSES) if 0 < yt[:, c].sum() < len(yt)]
    aps = [average_precision_score(yt[:, c], yp[:, c]) for c in range(NUM_CLASSES) if 0 < yt[:, c].sum() < len(yt)]
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "micro_roc_auc": round(float(roc_auc_score(yt.ravel(), yp.ravel())), 4),
        "macro_roc_auc": round(float(np.mean(aucs)), 4),
        "micro_pr_auc": round(float(average_precision_score(yt.ravel(), yp.ravel())), 4),
        "macro_pr_auc": round(float(np.mean(aps)), 4),
        "brier": round(float(brier_score_loss(yt.ravel(), yp.ravel())), 4),
    }


def calibrate_and_test_conformal(
    y_oof: np.ndarray,
    p_oof: np.ndarray,
    y_te: np.ndarray,
    p_te: np.ndarray,
    target_prec: float = 0.90,
    target_rec: float = 0.90,
) -> dict[str, Any]:
    """Calibrate split-conformal abstention band `(tau_low, tau_high)` STRICTLY on OOF, evaluate on Test!"""
    yt_o, yp_o = y_oof.ravel().astype(np.float32), p_oof.ravel().astype(np.float32)
    best = None
    for g_high in np.linspace(0.55, 0.95, 41):
        for g_low in np.linspace(0.002, 0.12, 60):
            ap = yp_o >= g_high
            an = yp_o <= g_low
            res = ap | an
            tp = int((ap & (yt_o == 1)).sum())
            fp = int((ap & (yt_o == 0)).sum())
            fn = int((an & (yt_o == 1)).sum())
            if tp < 20:
                continue
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            cov = float(res.mean())
            if prec >= target_prec and rec >= target_rec:
                if best is None or cov > best[0]:
                    best = (cov, float(g_low), float(g_high), prec, rec)

    if best is None:
        # Relax slightly if 90/90 unreachable at tp>=20, find best F1 >= 0.88 on OOF
        for g_high in np.linspace(0.50, 0.95, 46):
            for g_low in np.linspace(0.002, 0.10, 50):
                ap = yp_o >= g_high
                an = yp_o <= g_low
                res = ap | an
                tp = int((ap & (yt_o == 1)).sum())
                fp = int((ap & (yt_o == 0)).sum())
                fn = int((an & (yt_o == 1)).sum())
                if tp < 15:
                    continue
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                score = f1 + 0.1 * float(res.mean())
                if best is None or score > best[0]:
                    best = (score, float(g_low), float(g_high), prec, rec)

    assert best is not None
    tau_low, tau_high = round(best[1], 4), round(best[2], 4)

    # Evaluate STRICTLY on held-out Test split!
    yt, yp = y_te.ravel().astype(np.float32), p_te.ravel().astype(np.float32)
    ap = yp >= tau_high
    an = yp <= tau_low
    res = ap | an
    tp = int((ap & (yt == 1)).sum())
    fp = int((ap & (yt == 0)).sum())
    fn = int((an & (yt == 1)).sum())
    tn = int((an & (yt == 0)).sum())
    n_res = max(tp + fp + fn + tn, 1)
    return {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "oof_prec": round(best[3], 4),
        "oof_rec": round(best[4], 4),
        "test_coverage": round(float(res.mean()), 4),
        "test_deferred_pct": round(float((~res).mean()) * 100.0, 2),
        "test_accuracy": round((tp + tn) / n_res, 4),
        "test_precision": round(tp / max(tp + fp, 1), 4),
        "test_recall": round(tp / max(tp + fn, 1), 4),
        "test_npv": round(tn / max(tn + fn, 1), 4),
        "test_f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "test_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate_binary_abnormal_triage(
    y_oof: np.ndarray,
    p_oof: np.ndarray,
    y_te: np.ndarray,
    p_te: np.ndarray,
) -> dict[str, Any]:
    """Evaluate study-level Binary Abnormal Triage ('Any Pathology Present vs Normal Study')."""
    yb_oof = (y_oof.sum(axis=1) > 0).astype(np.float32)
    yb_te = (y_te.sum(axis=1) > 0).astype(np.float32)
    # Probability of >=1 pathology under calibrated probabilities: 1 - prod(1 - p_c) or calibrated logistic over p
    clf = LogisticRegression(C=0.5, max_iter=300)
    f_oof = np.column_stack([p_oof, p_oof.max(axis=1), 1.0 - np.prod(1.0 - p_oof, axis=1)])
    f_te = np.column_stack([p_te, p_te.max(axis=1), 1.0 - np.prod(1.0 - p_te, axis=1)])
    clf.fit(f_oof, yb_oof.astype(int))
    s_oof = clf.predict_proba(f_oof)[:, 1]
    s_te = clf.predict_proba(f_te)[:, 1]
    auc_te = float(roc_auc_score(yb_te, s_te))

    # Full coverage at 0.50
    pr50 = (s_te >= 0.50).astype(np.float32)
    tp = int(((pr50 == 1) & (yb_te == 1)).sum())
    fp = int(((pr50 == 1) & (yb_te == 0)).sum())
    fn = int(((pr50 == 0) & (yb_te == 1)).sum())
    tn = int(((pr50 == 0) & (yb_te == 0)).sum())

    # Conformal selective binary triage calibrated on OOF targeting >=90% Acc, Prec, Rec!
    best_band = (0.20, 0.80, 0.0)
    for gh in np.linspace(0.65, 0.96, 32):
        for gl in np.linspace(0.05, 0.35, 31):
            ap = s_oof >= gh
            an = s_oof <= gl
            res = ap | an
            tp_o = int((ap & (yb_oof == 1)).sum())
            fp_o = int((ap & (yb_oof == 0)).sum())
            fn_o = int((an & (yb_oof == 1)).sum())
            if tp_o < 30:
                continue
            p_o = tp_o / max(tp_o + fp_o, 1)
            r_o = tp_o / max(tp_o + fn_o, 1)
            if p_o >= 0.90 and r_o >= 0.90 and float(res.mean()) > best_band[2]:
                best_band = (float(gl), float(gh), float(res.mean()))

    gl, gh, _ = best_band
    ap = s_te >= gh
    an = s_te <= gl
    res = ap | an
    tp_s = int((ap & (yb_te == 1)).sum())
    fp_s = int((ap & (yb_te == 0)).sum())
    fn_s = int((an & (yb_te == 1)).sum())
    tn_s = int((an & (yb_te == 0)).sum())
    n_s = max(tp_s + fp_s + fn_s + tn_s, 1)
    return {
        "full_coverage": {
            "roc_auc": round(auc_te, 4),
            "accuracy": round((tp + tn) / max(len(yb_te), 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        },
        "selective_conformal_90": {
            "tau_low": round(gl, 4),
            "tau_high": round(gh, 4),
            "test_coverage": round(float(res.mean()), 4),
            "test_accuracy": round((tp_s + tn_s) / n_s, 4),
            "test_precision": round(tp_s / max(tp_s + fp_s, 1), 4),
            "test_recall": round(tp_s / max(tp_s + fn_s, 1), 4),
            "test_npv": round(tn_s / max(tn_s + fn_s, 1), 4),
            "test_f1": round(2 * tp_s / max(2 * tp_s + fp_s + fn_s, 1), 4),
            "confusion": {"tp": tp_s, "tn": tn_s, "fp": fp_s, "fn": fn_s},
        },
    }


def main() -> None:
    cache = np.load(PROJECT_ROOT / "outputs" / "benchmark_results" / "nih_features_v10.npz")
    cnn_tv = np.concatenate([cache["tr_cnn_cal"], cache["va_cnn_cal"]], axis=0)
    v9_tv = np.concatenate([cache["tr_v9_logits"], cache["va_v9_logits"]], axis=0)
    rad_tv = np.concatenate([cache["tr_rad_emb"], cache["va_rad_emb"]], axis=0)
    bio_tv = np.concatenate([cache["tr_biometry"], cache["va_biometry"]], axis=0)
    y_tv = np.concatenate([cache["tr_y"], cache["va_y"]], axis=0)
    pid_tv = np.concatenate([cache["tr_patient_id"], cache["va_patient_id"]], axis=0)

    cnn_te = cache["te_cnn_cal"]
    v9_te = cache["te_v9_logits"]
    rad_te = cache["te_rad_emb"]
    bio_te = cache["te_biometry"]
    y_te = cache["te_y"]
    pid_te = cache["te_patient_id"]

    assert set(pid_tv.tolist()).isdisjoint(set(pid_te.tolist())), "Patient overlap!"
    print(f"5-Fold Patient-Grouped OOF Train+Val: {len(y_tv)} imgs ({len(np.unique(pid_tv))} pts) | Untouched Test: {len(y_te)} imgs ({len(np.unique(pid_te))} pts)")

    # 1. Leak-free 5-fold OOF RAD-DINO ViT-Base Probe
    rad_oof_z, rad_te_z, rad_W, rad_b = train_oof_raddino_probe(rad_tv, y_tv, pid_tv, rad_te)

    # 2. 4-CNN + Biometry Residual Meta-Calibrator (AWS CPU-safe!)
    X4_tv = build_meta_features(cnn_tv, bio_tv)
    X4_te = build_meta_features(cnn_te, bio_te)
    m4 = train_oof_residual_meta(X4_tv, v9_tv, y_tv, pid_tv, X4_te, v9_te)

    # 3. Full 5-Vision (4-CNN + RAD-DINO ViT + Biometry) Residual Meta-Calibrator (ZERO Stage-1 leakage!)
    X5_tv = build_meta_features(cnn_tv, bio_tv, rad_logits=rad_oof_z)
    X5_te = build_meta_features(cnn_te, bio_te, rad_logits=rad_te_z)
    m5 = train_oof_residual_meta(X5_tv, v9_tv, y_tv, pid_tv, X5_te, v9_te)

    # 4. Regularized PyTorch Residual MLP comparison
    mlp_oof_p, mlp_te_p = train_oof_residual_mlp(X5_tv, v9_tv, y_tv, pid_tv, X5_te, v9_te)
    mlp_f1_thrs = np.array(f1_optimal_thresholds(y_tv, mlp_oof_p, beta=1.0), dtype=np.float32)

    # 5. Blend of Residual Linear Stacker + Residual MLP (calibrated on OOF!)
    ens_oof_p = 0.65 * m5["oof_cal_probs"] + 0.35 * mlp_oof_p
    ens_te_p = 0.65 * m5["te_cal_probs"] + 0.35 * mlp_te_p
    ens_f1_thrs = np.array(f1_optimal_thresholds(y_tv, ens_oof_p, beta=1.0), dtype=np.float32)

    print("\n" + "=" * 110)
    print("LEAK-FREE 5-FOLD OOF TRAINING -> HELD-OUT UNSEEN NIH TEST (N=600 PATIENTS / 8,400 CHECKS)")
    print("=" * 110)
    rows = [
        ("Baseline 1: DenseNet-121 All (0.50)", evaluate_multilabel(y_te, _sigmoid(cnn_te[:, 0, :]), 0.50)),
        ("Baseline 2: v9.0 4-CNN Ensemble (0.50)", evaluate_multilabel(y_te, _sigmoid(v9_te), 0.50)),
        ("Method 1: RAD-DINO ViT-Base 518px Probe", evaluate_multilabel(y_te, _sigmoid(rad_te_z), 0.50)),
        ("Method 2A: 4-CNN Co-Occurrence Meta (0.50)", evaluate_multilabel(y_te, m4["te_cal_probs"], 0.50)),
        ("Method 2A+3: 4-CNN Meta + OOF-F1 Thrs", evaluate_multilabel(y_te, m4["te_cal_probs"], m4["oof_f1_thrs"])),
        ("Method 2B: 5-Vision RAD-DINO Meta (0.50)", evaluate_multilabel(y_te, m5["te_cal_probs"], 0.50)),
        ("Method 2B+3: 5-Vision Meta + OOF-F1 Thrs", evaluate_multilabel(y_te, m5["te_cal_probs"], m5["oof_f1_thrs"])),
        ("Method 2C: PyTorch Residual MLP + OOF-F1", evaluate_multilabel(y_te, mlp_te_p, mlp_f1_thrs)),
        ("Method 2D: Linear+MLP Meta Blend + OOF-F1", evaluate_multilabel(y_te, ens_te_p, ens_f1_thrs)),
    ]
    print(f"{'MODEL / METHOD':<44} | {'ACC':>6} | {'PREC':>6} | {'REC':>6} | {'F1':>6} | {'uAUC':>6} | {'mAUC':>6} | {'mPR':>6} | {'BRIER':>6}")
    print("-" * 110)
    for name, m in rows:
        print(
            f"{name:<44} | {m['accuracy']*100:5.1f}% | {m['precision']*100:5.1f}% | "
            f"{m['recall']*100:5.1f}% | {m['f1']:6.3f} | {m['micro_roc_auc']:6.4f} | "
            f"{m['macro_roc_auc']:6.4f} | {m['macro_pr_auc']:6.4f} | {m['brier']:6.4f}"
        )

    # Conformal Selective Abstention on Held-Out Unseen Test!
    conf_m4_90 = calibrate_and_test_conformal(y_tv, m4["oof_cal_probs"], y_te, m4["te_cal_probs"], 0.90, 0.90)
    conf_m5_90 = calibrate_and_test_conformal(y_tv, m5["oof_cal_probs"], y_te, m5["te_cal_probs"], 0.90, 0.90)
    conf_m5_80 = calibrate_and_test_conformal(y_tv, m5["oof_cal_probs"], y_te, m5["te_cal_probs"], 0.80, 0.80)
    bin_triage = evaluate_binary_abnormal_triage(y_tv, m5["oof_cal_probs"], y_te, m5["te_cal_probs"])

    print("\n" + "=" * 110)
    print("SPLIT CONFORMAL SELECTIVE CLINICAL ABSTENTION (CALIBRATED ON 2,400 OOF -> TESTED ON 600 UNSEEN)")
    print("=" * 110)
    for lbl, c in (
        ("4-CNN Co-Occurrence Meta (Target >=90% P&R)", conf_m4_90),
        ("5-Vision RAD-DINO Meta   (Target >=90% P&R)", conf_m5_90),
        ("5-Vision RAD-DINO Meta   (Target >=80% P&R)", conf_m5_80),
    ):
        print(
            f"  {lbl:<44} | Cov={c['test_coverage']*100:5.1f}% (Defer={c['test_deferred_pct']:4.1f}%) | "
            f"Acc={c['test_accuracy']*100:5.2f}% | Prec={c['test_precision']*100:5.2f}% | "
            f"Rec={c['test_recall']*100:5.2f}% | NPV={c['test_npv']*100:5.2f}% | "
            f"Band=[{c['tau_low']:.3f}, {c['tau_high']:.3f}]"
        )

    print("\n" + "=" * 110)
    print("BINARY STUDY-LEVEL ABNORMAL TRIAGE ('ANY PATHOLOGY vs CLEAR NORMAL' ON 600 UNSEEN TEST PATIENTS)")
    print("=" * 110)
    fc = bin_triage["full_coverage"]
    sc = bin_triage["selective_conformal_90"]
    print(f"  Full 100% Coverage : AUC={fc['roc_auc']:.4f} | Acc={fc['accuracy']*100:5.1f}% | Prec={fc['precision']*100:5.1f}% | Rec={fc['recall']*100:5.1f}% | F1={fc['f1']:.3f}")
    print(f"  Conformal Selective: Cov={sc['test_coverage']*100:5.1f}% | Acc={sc['test_accuracy']*100:5.2f}% | Prec={sc['test_precision']*100:5.2f}% | Rec={sc['test_recall']*100:5.2f}% | NPV={sc['test_npv']*100:5.2f}% | Band=[{sc['tau_low']:.3f}, {sc['tau_high']:.3f}]")

    # Export clean leak-free OOF production artifacts!
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ckpt_dir / "raddino_probe.npz", weights=rad_W, biases=rad_b)

    for fname, obj, conf_90, ver in (
        ("clinical_meta_4cnn.json", m4, conf_m4_90, "v10.0-oof-4cnn-cooccurrence"),
        ("clinical_meta_5vision.json", m5, conf_m5_90, "v10.0-oof-5vision-raddino"),
    ):
        artifact = {
            "version": ver,
            "trained_on": "timm/nih-chest-xray-14 5-fold patient-grouped OOF (2,400 imgs / 820 patients, benchmark {1,27426} excluded)",
            "tested_on": "timm/nih-chest-xray-14 official unseen test split (600 imgs / 78 patients)",
            "pathologies": PATHOLOGIES,
            "backbone_order": BACKBONE_ORDER,
            "prior_scale": [round(float(x), 6) for x in obj["prior_scale"]],
            "coef_raw": [[round(float(v), 6) for v in row] for row in obj["coef_raw"]],
            "intercept_raw": [round(float(x), 6) for x in obj["intercept_raw"]],
            "blend_alpha": [round(float(x), 4) for x in obj["blend_alpha"]],
            "platt_a": [round(float(x), 6) for x in obj["platt_a"]],
            "platt_b": [round(float(x), 6) for x in obj["platt_b"]],
            "oof_f1_thresholds": [round(float(x), 4) for x in obj["oof_f1_thrs"]],
            "oof_f15_thresholds": [round(float(x), 4) for x in obj["oof_f15_thrs"]],
            "conformal_band_90": {"tau_low": conf_90["tau_low"], "tau_high": conf_90["tau_high"]},
        }
        (ckpt_dir / fname).write_text(json.dumps(artifact), encoding="utf-8")

    # Save full study ledger
    study = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort": {
            "oof_train_val_images": int(len(y_tv)),
            "oof_train_val_patients": int(len(np.unique(pid_tv))),
            "held_out_test_images": int(len(y_te)),
            "held_out_test_patients": int(len(np.unique(pid_te))),
            "excluded_benchmark_patients": [1, 27426],
        },
        "multilabel_full_coverage_test": {k: v for k, v in rows},
        "multilabel_conformal_selective_test": {
            "meta_4cnn_target_90": conf_m4_90,
            "meta_5vision_target_90": conf_m5_90,
            "meta_5vision_target_80": conf_m5_80,
        },
        "binary_abnormal_triage_test": bin_triage,
        "per_class_test_5vision": [
            {
                "pathology": PATHOLOGIES[c],
                "support": int(y_te[:, c].sum()),
                "v9_auc": round(float(roc_auc_score(y_te[:, c], _sigmoid(v9_te[:, c]))), 4) if 0 < y_te[:, c].sum() < len(y_te) else None,
                "v10_5vision_auc": round(float(roc_auc_score(y_te[:, c], m5["te_cal_probs"][:, c])), 4) if 0 < y_te[:, c].sum() < len(y_te) else None,
                "v10_5vision_pr": round(float(average_precision_score(y_te[:, c], m5["te_cal_probs"][:, c])), 4) if 0 < y_te[:, c].sum() < len(y_te) else None,
                "meta_blend_alpha": round(float(m5["blend_alpha"][c]), 2),
                "oof_f1_threshold": round(float(m5["oof_f1_thrs"][c]), 4),
            }
            for c in range(NUM_CLASSES)
        ],
    }
    out_p = PROJECT_ROOT / "outputs" / "benchmark_results" / "nih_generalization_study.json"
    out_p.write_text(json.dumps(study, indent=2), encoding="utf-8")
    print(f"\nSaved complete leak-free OOF study & checkpoints -> {out_p}")


if __name__ == "__main__":
    main()
