"""Practical SVM and RF baselines with calibration isolated to V_cal."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from ..dataset.preprocess import normalize_trace, fit_global_stats
from ..training.calibration import fit_platt
from ..evaluation.metrics import binary_metrics
from ..utils import save_json, seed_everything


def extract_features(store, indices, mode='per_channel', stats=None, bins=60):
    """Training-only feature construction; 1D average/variance/differences (not proposed model)."""
    features = []
    for idx in indices:
        x = normalize_trace(store.X[int(idx)], mode, stats)
        if len(x) % bins:
            # binning via array_split also accommodates arbitrary lengths
            chunks = np.array_split(x, bins, axis=0)
            mu = np.stack([c.mean(axis=0) for c in chunks])
            std = np.stack([c.std(axis=0) for c in chunks])
            diff = np.stack([np.abs(np.diff(c, axis=0)).mean(axis=0) for c in chunks])
        else:
            xs = x.reshape(bins, len(x)//bins, x.shape[1])
            mu = xs.mean(axis=1)
            std = xs.std(axis=1)
            diff = np.abs(np.diff(xs, axis=1)).mean(axis=1)
        features.append(np.concatenate([mu.flatten(), std.flatten(), diff.flatten()]))
    return np.asarray(features, dtype=np.float32)


def run_classical(store, split, cfg, model_name, out_dir, seed=42, generate_plots=True):
    seed_everything(seed)
    model_name = model_name.lower()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = cfg['preprocessing']['mode']
    stats = fit_global_stats(store.X, split['train']) if mode == 'global' else None
    X = {name:extract_features(store, ids, mode, stats) for name,ids in split.items()}
    y = {name: store.labels()[ids] for name,ids in split.items()}
    sub = store.subtypes()[split['test']]
    if model_name == 'svm':
        estimator = Pipeline([('scale', StandardScaler()),
                              ('model', LinearSVC(C=1., class_weight='balanced', dual='auto', max_iter=3000,
                                                  random_state=seed))])
        # linear SVM is chosen to make >40k traces tractable without huge kernel matrices
        estimator.fit(X['train'], y['train'])
        cal_scores = estimator.decision_function(X['cal'])
        test_scores = estimator.decision_function(X['test'])
    elif model_name == 'random_forest':
        estimator = RandomForestClassifier(n_estimators=150, max_depth=18,
                                           min_samples_leaf=2, class_weight='balanced',
                                           random_state=seed, n_jobs=-1)
        estimator.fit(X['train'], y['train'])
        cal_scores = estimator.predict_proba(X['cal'])[:,1]
        test_scores = estimator.predict_proba(X['test'])[:,1]
    else:
        raise ValueError('Classical models: svm, random_forest')
    platt = fit_platt(cal_scores, y['cal'])
    proba = platt.predict_proba(test_scores.reshape(-1,1))[:,1]
    tau = float(cfg['calibration'].get('threshold', .5))
    metrics = binary_metrics(y['test'], proba, sub, tau)
    pd.DataFrame({'index':split['test'], 'label':y['test'], 'subtype_id':sub,
                  'probability':proba, 'prediction':(proba >= tau).astype(int)}).to_csv(out_dir / 'predictions.csv', index=False)
    output = dict(model_name=model_name, seed=seed, test=metrics,
                  baseline_detail=('LinearSVC with binned mean/SD/difference features' if model_name=='svm'
                                   else 'RandomForest with binned mean/SD/difference features'),
                  calibration='Platt scaling on V_cal (no test fitted)',
                  caveat='Classical baselines use a fixed learned-feature replacement; not manuscript-specific tuned hyperparameters')
    save_json(out_dir / 'metrics.json', output)
    if generate_plots:
        from ..visualization.plots import evaluation_figures
        evaluation_figures(y['test'], proba, out_dir / 'figures', cfg['experiments'].get('figure_dpi',600))
    return output
