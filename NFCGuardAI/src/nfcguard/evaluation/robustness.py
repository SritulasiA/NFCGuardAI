"""Controlled, reproducible test-time channel impairments."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from ..dataset.preprocess import perturb_trace, normalize_trace
from ..evaluation.metrics import binary_metrics
from ..training.trainer import load_checkpoint
from ..utils import save_json

DEFAULT_CONDITIONS = {'awgn': [5, 10, 15, 20, 30],
                      'fading': [0, 1, 5, 10],
                      'cfo': [50, 100, 500, 1000],
                      'iq_imbalance': [.02, .05, .1, .2],
                      'interference': [-20, -10, -5, 0]}


def run_robustness(store, split, checkpoint, out_dir, max_samples=0, seed=42,
                   conditions=None, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, saved = load_checkpoint(checkpoint, device)
    if saved['channels'] != store.channels or saved['length'] != store.length:
        raise ValueError('Model checkpoint and evaluation store disagree on waveform shape')
    cfg = saved['config']
    indices = np.asarray(split['test'], dtype=int)
    if max_samples and len(indices) > max_samples:
        indices = np.sort(np.random.default_rng(seed).choice(indices, size=max_samples, replace=False))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = conditions or DEFAULT_CONDITIONS
    labels = store.labels()[indices]
    subtypes = store.subtypes()[indices]
    all_rows = []
    for kind, levels in {'clean': [0], **conditions}.items():
        if store.channels == 1 and kind in ('fading','cfo','iq_imbalance'):
            print(f'Skipping {kind}: these perturbations require two genuine channels')
            continue
        for level in levels:
            logits = []
            with torch.inference_mode():
                for start in range(0, len(indices), batch_size):
                    batch_ids = indices[start:start+batch_size]
                    waveforms = []
                    for idx in batch_ids:
                        wave = np.asarray(store.X[idx], dtype=np.float32)
                        if kind != 'clean':
                            rng = np.random.default_rng(seed * 1000003 + idx * 97 +
                                                        int(abs(float(level))*1000) + sum(ord(c) for c in kind))
                            wave = perturb_trace(wave, kind, level, rng, cfg['dataset'].get('sample_rate',10000000))
                        wave = normalize_trace(wave, cfg['preprocessing']['mode'], saved['global_stats'])
                        waveforms.append(wave.T)
                    tensor = torch.tensor(np.stack(waveforms).copy(), dtype=torch.float32, device=device)
                    logits.append(model(tensor).cpu().numpy())
            p = expit(np.concatenate(logits) / float(saved['temperature']))
            metrics = binary_metrics(labels, p, subtypes, threshold=cfg['calibration'].get('threshold', .5))
            row = {'impairment':kind, 'level':float(level), 'n':len(indices),
                   **{k:metrics[k] for k in ('accuracy','f1','recall','roc_auc','pr_auc')}}
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(out_dir / 'robustness.csv', index=False)
    save_json(out_dir / 'robustness.json', all_rows)
    from ..visualization.plots import robustness_figures
    robustness_figures(pd.DataFrame(all_rows), out_dir, cfg['experiments'].get('figure_dpi', 600))
    return all_rows
