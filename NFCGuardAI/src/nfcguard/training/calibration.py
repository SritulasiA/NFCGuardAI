"""Nested validation-only probability calibration."""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def fit_temperature(logits, y):
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    if z.numel() != y.numel() or z.numel() == 0:
        raise ValueError('Calibration logits/labels must have same nonzero size')
    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optim = torch.optim.LBFGS([log_t], lr=.1, max_iter=150, line_search_fn='strong_wolfe')
    def closure():
        optim.zero_grad()
        value = torch.nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        value.backward()
        return value
    optim.step(closure)
    return float(log_t.detach().exp().clamp(.05, 20.0).item())


def temperature_predict(logits, temperature=1.0):
    return expit(np.asarray(logits, dtype=np.float64) / float(temperature))


def fit_platt(scores, labels):
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError('Platt scaling requires both legitimate and relay samples')
    lr = LogisticRegression(C=1e4, solver='lbfgs', max_iter=500)
    lr.fit(np.asarray(scores).reshape(-1, 1), y)
    return lr


def reliability(y, p, bins=15):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    counts = np.zeros(bins, dtype=int)
    confidences = np.zeros(bins)
    accuracies = np.zeros(bins)
    ece = 0.0
    for i in range(bins):
        lower, upper = i / bins, (i + 1) / bins
        mask = (p >= lower) & ((p < upper) if i != bins - 1 else (p <= upper))
        counts[i] = int(mask.sum())
        if counts[i]:
            confidences[i] = float(p[mask].mean())
            accuracies[i] = float(y[mask].mean())
            ece += counts[i] / len(p) * abs(confidences[i] - accuracies[i])
    return {'ece': float(ece), 'brier': float(brier_score_loss(y, p)),
            'nll': float(log_loss(y, p, labels=[0, 1])),
            'counts': counts.tolist(), 'confidence': confidences.tolist(), 'observed': accuracies.tolist()}


def threshold_sensitivity(y, p):
    """Analysis of non-deployment thresholds on V_cal only."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(np.unique(y)) != 2:
        return {'youden': None, 'f1': None}
    fpr, tpr, thresholds = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    _, recall, pr_thresholds = precision_recall_curve(y, p)
    thresholds_scan = np.linspace(.05, .95, 181)
    f1_list = []
    for tau in thresholds_scan:
        pred = p >= tau
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        fn = (~pred & (y == 1)).sum()
        f1_list.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    return {'youden': float(np.clip(thresholds[j], 0, 1)),
            'f1': float(thresholds_scan[np.argmax(f1_list)]),
            'f1_at_fixed': float(f1_list[np.argmin(abs(thresholds_scan - .5))])}
