"""Corrected fold-level tests, paired prediction tests, multiple comparison correction."""
from __future__ import annotations
import numpy as np
from scipy.stats import t as student_t, chi2, binomtest


def corrected_resampled_t(proposed, comparator, train_fraction=.8, test_fraction=.1):
    """Nadeau-Bengio: SE=sqrt((1/k + n_test/n_train) * sample_variance).
    Train fraction refers to effective fit-training size; for manuscript display use .9.
    """
    a, b = np.asarray(proposed, dtype=float), np.asarray(comparator, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError('Corrected t test requires paired fold vectors of length >=2')
    if train_fraction <= 0 or test_fraction <= 0:
        raise ValueError('Must specify positive train/test fractions')
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    inflation = 1 / n + test_fraction / train_fraction
    se = np.sqrt(inflation) * sd
    statistic = float(mean / se) if se > 0 else (0. if mean == 0 else np.sign(mean) * np.inf)
    p = float(2 * student_t.sf(abs(statistic), df=n - 1))
    half = float(student_t.ppf(.975, df=n - 1) * se)
    correction = 1 - 3 / (4 * (n - 1) - 1) if n > 2 else 1.
    hedges_gz = float(correction * mean / sd) if sd else None
    return dict(n=n, mean_difference=mean, corrected_t=statistic if np.isfinite(statistic) else None,
                p_value=p, ci_low=mean - half, ci_high=mean + half,
                effect_size_hedges_gz=hedges_gz, variance_inflation=inflation)


def holm_bonferroni(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    ordered = np.argsort(pvalues)
    adjusted = np.empty(n)
    prev = 0.
    for rank, idx in enumerate(ordered):
        prev = max(prev, min(1., (n - rank) * pvalues[idx]))
        adjusted[idx] = prev
    return adjusted.tolist()


def mcnemar_paired(correct_a, correct_b, exact_cutoff=25):
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('Matched predictions are required')
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    discord = n10 + n01
    if not discord:
        return dict(a_only=n10, b_only=n01, p_value=1., method='identical')
    if discord < exact_cutoff:
        p = min(1., float(2 * binomtest(min(n10, n01), discord, .5, alternative='less').pvalue))
        method = 'exact'
    else:
        statistic = (abs(n10 - n01) - 1)**2 / discord
        p = float(chi2.sf(statistic, 1))
        method = 'chi_square_continuity_corrected'
    return dict(a_only=n10, b_only=n01, p_value=p, method=method)
