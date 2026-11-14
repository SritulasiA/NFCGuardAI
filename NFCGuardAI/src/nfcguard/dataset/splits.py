from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from ..utils import save_json


def _split(indices, labels, test_frac, seed):
    try:
        return train_test_split(np.asarray(indices), test_size=test_frac, random_state=int(seed), stratify=labels)
    except ValueError as e:
        raise ValueError(f'Not enough examples in each subtype for stratified split: {e}') from e


def heldout(subtypes, seed=42):
    all_ids = np.arange(len(subtypes))
    train_val, test = _split(all_ids, np.asarray(subtypes), .15, seed)
    # 15% of original is validation; half reserved for checkpoints and calibration.
    train, validation = _split(train_val, np.asarray(subtypes)[train_val], 15 / 85, seed + 1)
    ckpt, cal = _split(validation, np.asarray(subtypes)[validation], .5, seed + 2)
    return dict(train=train.tolist(), ckpt=ckpt.tolist(), cal=cal.tolist(), test=test.tolist())


def cv_splits(subtypes, folds=10, seed=42):
    subtype = np.asarray(subtypes)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_val, test) in enumerate(splitter.split(np.zeros(len(subtype)), subtype)):
        # Outer split 90/10; internal 70/10/10 globally
        train, validation = _split(train_val, subtype[train_val], 2 / 9, seed + fold + 101)
        ckpt, cal = _split(validation, subtype[validation], .5, seed + fold + 201)
        yield fold, dict(train=train.tolist(), ckpt=ckpt.tolist(), cal=cal.tolist(), test=test.tolist())


def group_splits(subtypes, groups, seed=42, test_fraction=.2):
    subtype = np.asarray(subtypes)
    groups = np.asarray(groups).astype(str)
    if len(subtype) != len(groups) or any(g in ('', 'nan', 'None') for g in groups):
        raise ValueError('Cannot use unknown group labels for group-disjoint evaluation')
    groups_unique = np.unique(groups)
    if len(groups_unique) < 3:
        raise ValueError('At least three groups are required for train/checkpoint/calibration/test')
    split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_pos, test_pos = next(split.split(np.zeros(len(groups)), subtype, groups))
    g_train = groups[train_val_pos]
    tv_split = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1)
    train_pos, validation_pos = next(tv_split.split(np.zeros(len(train_val_pos)), subtype[train_val_pos], g_train))
    train = train_val_pos[train_pos]
    val = train_val_pos[validation_pos]
    val_groups = sorted(set(groups[val]))
    if len(val_groups) >= 2:
        midpoint = max(1, len(val_groups) // 2)
        ckpt = val[np.isin(groups[val], val_groups[:midpoint])]
        cal = val[np.isin(groups[val], val_groups[midpoint:])]
    else:
        # Explicit: group disjointness is only required between train and test;
        # ckpt/cal separation has zero records in common, though same group possible.
        ckpt, cal = _split(val, subtype[val], .5, seed + 2)
    return dict(train=train.tolist(), ckpt=ckpt.tolist(), cal=cal.tolist(), test=test_pos.tolist())


def leave_one_tag_out(meta, seed=42):
    """Hold out normal/wired traces for one tag; wireless traces remain excluded
    from tag-specific test because they cannot be linked to the tag. The manuscript's
    claimed disjoint wireless block requires trustworthy metadata and separate study.
    """
    sub = meta['subtype'].astype(str).to_numpy()
    tags = meta['tag_id'].astype(str).to_numpy()
    known = (sub != 'wireless') & (tags != '')
    eligible = np.where(known)[0]
    if len(eligible) == 0 or len(set(tags[eligible])) < 2:
        raise ValueError('No verifiable tag identities for leave-one-tag-out evaluation')
    for j, tag in enumerate(sorted(set(tags[eligible]))):
        test = eligible[tags[eligible] == tag]
        train_pool = eligible[tags[eligible] != tag]
        # Stratify remaining by subtype for checkpoint/calibration
        train, val = _split(train_pool, sub[train_pool], .2, seed + j)
        ckpt, cal = _split(val, sub[val], .5, seed + 100 + j)
        yield tag, dict(train=train.tolist(), ckpt=ckpt.tolist(), cal=cal.tolist(), test=test.tolist())


def validate_splits(split, n_total=None, groups=None):
    splits = {k: np.asarray(v, dtype=int) for k, v in split.items()}
    if set(splits) != {'train', 'ckpt', 'cal', 'test'}:
        raise ValueError('Split must contain train, ckpt, cal, test')
    for a in splits:
        if len(splits[a]) == 0:
            raise ValueError(f'Empty partition: {a}')
        if len(np.unique(splits[a])) != len(splits[a]):
            raise ValueError(f'Duplicate indices inside {a}')
        for b in splits:
            if a != b and np.intersect1d(splits[a], splits[b]).size:
                raise ValueError(f'Sample-level leakage between {a} and {b}')
    if n_total is not None and any(np.any((idx < 0) | (idx >= n_total)) for idx in splits.values()):
        raise ValueError('Partition indices outside dataset length')
    if groups is not None:
        g = np.asarray(groups).astype(str)
        if set(g[splits['train']]) & set(g[splits['test']]):
            raise ValueError('Group leakage between training and evaluation')
    return True


def save_splits(path, split):
    save_json(path, split)


def load_splits(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
