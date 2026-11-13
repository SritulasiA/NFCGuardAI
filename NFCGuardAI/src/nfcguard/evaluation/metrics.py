"""Primary metrics, subtype recall, classification reports and fold summaries."""
from __future__ import annotations
import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             balanced_accuracy_score)
from ..training.calibration import reliability


def binary_metrics(y, proba, subtype=None, threshold=.5):
    y = np.asarray(y, dtype=np.int32)
    p = np.clip(np.asarray(proba, dtype=float), 0, 1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError('Expected nonempty equal-length binary labels and probabilities')
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                  precision=float(precision_score(y, pred, zero_division=0)),
                  recall=float(recall_score(y, pred, zero_division=0)),
                  specificity=float(tn / max(tn + fp, 1)),
                  f1=float(f1_score(y, pred, zero_division=0)),
                  fpr=float(fp / max(tn + fp, 1)),
                  fnr=float(fn / max(tp + fn, 1)),
                  tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(threshold))
    result['roc_auc'] = float(roc_auc_score(y, p)) if len(set(y)) == 2 else None
    result['pr_auc'] = float(average_precision_score(y, p)) if len(set(y)) == 2 else None
    result['calibration'] = reliability(y, p)
    if subtype is not None:
        subtype = np.asarray(subtype)
        result['subtype_recall'] = {}
        for sid, name, expected in ((0, 'legitimate', 0), (1, 'wired', 1), (2, 'wireless', 1)):
            mask = subtype == sid
            result['subtype_recall'][name] = (float((pred[mask] == expected).mean()) if mask.sum() else None)
        result['subtype_counts'] = {name: int((subtype == sid).sum()) for sid, name in enumerate(('legitimate','wired','wireless'))}
    return result


def summarize_folds(rows, confidence=.95):
    rows = list(rows)
    if not rows:
        return {}
    keys = ('accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc', 'pr_auc')
    output = {}
    for key in keys:
        vals = np.asarray([x[key] for x in rows if x.get(key) is not None], dtype=float)
        n = len(vals)
        if not n:
            continue
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        half = float(student_t.ppf((1 + confidence) / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.
        output[key] = dict(n=n, mean=mean, sd=sd, ci_low=mean - half, ci_high=mean + half)
    return output
