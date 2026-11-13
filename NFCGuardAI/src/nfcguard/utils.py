"""General configuration, seeding and stable file operations."""
from __future__ import annotations
import copy
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a mapping')
    return cfg


def save_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False)


def seed_everything(seed=42, deterministic=True):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def get_device(name='auto'):
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def set_nested(config, dotted_key, value):
    cfg = copy.deepcopy(config)
    target = cfg
    parts = dotted_key.split('.')
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    return cfg


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
