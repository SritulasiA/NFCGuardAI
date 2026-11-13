"""Paired predictions / cross-validation statistical reports."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .statistics import corrected_resampled_t, holm_bonferroni, mcnemar_paired
from ..utils import save_json


def compare_heldout_predictions(proposed_file, comparator_files, out_file):
    proposed = pd.read_csv(proposed_file).set_index('index')
    results = []
    for path in comparator_files:
        b = pd.read_csv(path).set_index('index')
        if set(b.index) != set(proposed.index):
            raise ValueError(f'Predictions have different test IDs: {path}')
        b = b.loc[proposed.index]
        if not np.array_equal(proposed.label.values,b.label.values):
            raise ValueError('Inconsistent truth labels across models')
        truth = proposed.label.to_numpy()
        a_correct = (proposed.prediction.to_numpy() == truth)
        b_correct = (b.prediction.to_numpy() == truth)
        results.append({'comparator':str(path), **mcnemar_paired(a_correct,b_correct)})
    adjusted=holm_bonferroni([r['p_value'] for r in results])
    for record, value in zip(results,adjusted):
        record['holm_adjusted_p']=value
    save_json(out_file,results)
    return results


def compare_cv_metric(cv_csvs, proposed_name='iqsecurenet', metric='accuracy',
                      out_file=None, reproduction_ratio=.9, actual_train_ratio=.7):
    """Requires a CSV with columns model, fold, accuracy etc in every file."""
    frames=[pd.read_csv(f) for f in cv_csvs]
    df=pd.concat(frames,ignore_index=True)
    if df.duplicated(['model','fold']).any():
        raise ValueError('Duplicate model/fold rows')
    pivot=df.pivot(index='fold',columns='model',values=metric)
    if proposed_name not in pivot.columns:
        raise ValueError(f'{proposed_name} missing from fold table')
    records=[]
    for model in pivot.columns:
        if model==proposed_name:
            continue
        pair=pivot[[proposed_name,model]].dropna()
        if len(pair)<2:
            continue
        paper=corrected_resampled_t(pair[proposed_name],pair[model],reproduction_ratio,.1)
        actual=corrected_resampled_t(pair[proposed_name],pair[model],actual_train_ratio,.1)
        records.append({'comparator':model,'metric':metric, 'manuscript_correction':paper,
                        'effective_fit_correction':actual})
    pvals=[rec['manuscript_correction']['p_value'] for rec in records]
    adjusted=holm_bonferroni(pvals)
    for rec,adj in zip(records,adjusted):
        rec['holm_adjusted_p']=adj
    if out_file:
        save_json(out_file,records)
    return records
