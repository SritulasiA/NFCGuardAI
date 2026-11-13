"""Reproducible training and freeze-then-calibrate experiment runner."""
from __future__ import annotations
import copy
import logging
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from ..utils import save_json, seed_everything, get_device, worker_seed, utc_now
from ..dataset.preprocess import fit_global_stats
from ..dataset.torch_data import NFCWaveDataset, SubtypeBatchSampler
from ..dataset.splits import validate_splits, save_splits
from ..models.factory import create_model, complexity
from ..training.calibration import fit_temperature, temperature_predict, threshold_sensitivity
from ..evaluation.metrics import binary_metrics

LOG = logging.getLogger(__name__)


def make_loader(store, indices, cfg, global_stats=None, train=False, seed=42):
    pcfg = cfg['preprocessing']
    tcfg = cfg['training']
    dset = NFCWaveDataset(store, indices, normalization=pcfg['mode'], global_stats=global_stats,
                           augment=train and pcfg.get('augment_wireless', True),
                           probability=pcfg.get('augment_probability', .5), seed=seed,
                           eps=pcfg.get('epsilon', 1e-6))
    kw = dict(num_workers=int(tcfg.get('num_workers', 0)), pin_memory=(tcfg.get('device', 'auto') != 'cpu'),
              worker_init_fn=worker_seed)
    if train and tcfg.get('stratified_batches', True):
        sampler = SubtypeBatchSampler(dset.subtypes, tcfg['batch_size'], seed)
        loader = DataLoader(dset, batch_sampler=sampler, **kw)
    else:
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dset, batch_size=int(tcfg['batch_size']), shuffle=train, generator=generator, **kw)
    return loader


def class_weights(y):
    label = np.asarray(y, dtype=int)
    counts = np.bincount(label, minlength=2)
    if (counts == 0).any():
        raise ValueError('Both binary classes must appear in training')
    return [len(label) / (2 * int(c)) for c in counts]


def _epoch(model, loader, device, optimizer=None, weights=(1., 1.)):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.
    labels, predictions = [], []
    weight_vec = torch.tensor(weights, dtype=torch.float32, device=device)
    with torch.set_grad_enabled(train):
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_unweighted = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
            loss = (loss_unweighted * weight_vec[y.long()]).mean() if train else loss_unweighted.mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            labels.extend(y.detach().cpu().numpy().tolist())
            predictions.extend((logits.detach().cpu().numpy() >= 0).astype(int).tolist())
    return float(total_loss / len(loader.dataset)), float(f1_score(labels, predictions, zero_division=0))


def predict_logits(model, loader, device):
    model.eval()
    outputs, labels, subtypes, indices = [], [], [], []
    with torch.inference_mode():
        for x, y, s, idx in loader:
            logits = model(x.to(device)).cpu().numpy()
            outputs.append(logits)
            labels.append(y.numpy().astype(int))
            subtypes.append(s.numpy().astype(int))
            indices.append(idx.numpy().astype(int))
    return (np.concatenate(outputs), np.concatenate(labels),
            np.concatenate(subtypes), np.concatenate(indices))


def save_checkpoint(path, model, cfg, model_name, channels, length, global_stats, temperature, best_epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=copy.deepcopy(cfg),
                    model_name=model_name, channels=int(channels), length=int(length),
                    global_stats=global_stats, temperature=float(temperature),
                    best_epoch=int(best_epoch), created=utc_now()), path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    kwargs = {k: v for k, v in cfg['model'].items() if k != 'name'} if checkpoint['model_name'] == 'iqsecurenet' else {}
    model = create_model(checkpoint['model_name'], kwargs, checkpoint['channels'], checkpoint['length'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    return model, checkpoint


def train_experiment(store, splits, cfg, out_dir, model_name=None, seed=None, generate_plots=True):
    """Train exactly once; choose checkpoint on ckpt, T on cal, evaluate on test once."""
    cfg = copy.deepcopy(cfg)
    model_name = model_name or cfg['model'].get('name', 'iqsecurenet')
    seed = int(cfg.get('seed', 42) if seed is None else seed)
    seed_everything(seed, cfg['training'].get('deterministic', True))
    validate_splits(splits, len(store.X))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / 'config.json', cfg)
    save_splits(out_dir / 'splits.json', splits)
    device = get_device(cfg['training'].get('device', 'auto'))
    global_stats = fit_global_stats(store.X, splits['train']) if cfg['preprocessing']['mode'] == 'global' else None
    loaders = {name: make_loader(store, ids, cfg, global_stats, train=name == 'train', seed=seed)
               for name, ids in splits.items()}
    kwargs = cfg['model'].copy() if model_name == 'iqsecurenet' else {}
    kwargs.pop('name', None)
    model = create_model(model_name, kwargs, store.channels, store.length).to(device)
    model_complexity = complexity(model, (1, store.channels, store.length))
    model.train()
    tcfg = cfg['training']
    w = class_weights(store.labels()[splits['train']]) if tcfg.get('class_weighted', True) else [1., 1.]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg['lr']),
                                   weight_decay=float(tcfg.get('weight_decay', 1e-4)),
                                   betas=(.9, .999))
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(tcfg.get('step_size', 10)),
                                               gamma=float(tcfg.get('gamma', .5)))
    best_loss, best_epoch, stagnant = math.inf, -1, 0
    ckpt_path = out_dir / 'best.pt'
    history = []
    for epoch in range(int(tcfg['epochs'])):
        loaders['train'].dataset.set_epoch(epoch)
        if hasattr(loaders['train'].batch_sampler, 'set_epoch'):
            loaders['train'].batch_sampler.set_epoch(epoch)
        start = time.perf_counter()
        tr_loss, tr_f1 = _epoch(model, loaders['train'], device, optimizer=optimizer, weights=w)
        val_loss, val_f1 = _epoch(model, loaders['ckpt'], device)
        history.append(dict(epoch=epoch + 1, train_loss=tr_loss, train_f1=tr_f1,
                            ckpt_loss=val_loss, ckpt_f1=val_f1,
                            learning_rate=float(optimizer.param_groups[0]['lr']),
                            epoch_seconds=time.perf_counter() - start))
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stagnant = val_loss, epoch + 1, 0
            save_checkpoint(ckpt_path, model, cfg, model_name, store.channels, store.length, global_stats, 1., best_epoch)
        else:
            stagnant += 1
        LOG.info('epoch %d train=%.5f val=%.5f val_f1=%.4f', epoch + 1, tr_loss, val_loss, val_f1)
        schedule.step()
        if stagnant >= int(tcfg['patience']):
            break
    pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)
    model, _ = load_checkpoint(ckpt_path, device)
    cal_logits, cal_y, _, _ = predict_logits(model, loaders['cal'], device)
    temperature = fit_temperature(cal_logits, cal_y)
    tau = float(cfg['calibration'].get('threshold', .5))
    cal_thresholds = threshold_sensitivity(cal_y, temperature_predict(cal_logits, temperature))
    # Freeze architecture, checkpoint, temperature and threshold before evaluating test.
    test_logits, test_y, test_sub, test_ids = predict_logits(model, loaders['test'], device)
    test_prob = temperature_predict(test_logits, temperature)
    test_metrics = binary_metrics(test_y, test_prob, test_sub, threshold=tau)
    pd.DataFrame({'index':test_ids, 'label':test_y, 'subtype_id':test_sub,
                  'logit':test_logits, 'probability':test_prob,
                  'prediction':(test_prob >= tau).astype(int)}).to_csv(out_dir / 'predictions.csv', index=False)
    save_checkpoint(ckpt_path, model, cfg, model_name, store.channels, store.length, global_stats, temperature, best_epoch)
    output = dict(model_name=model_name, seed=seed, device=str(device),
                  best_epoch=best_epoch, class_weights=w, temperature=temperature,
                  calibration_threshold_analysis=cal_thresholds, complexity=model_complexity,
                  test=test_metrics, train_count=len(splits['train']), test_count=len(splits['test']),
                  data_caveat=store.manifest['caveat'],
                  protocols='No group-disjoint claim from random split; see generalization evaluations')
    save_json(out_dir / 'metrics.json', output)
    if generate_plots:
        from ..visualization.plots import evaluation_figures, training_figure
        evaluation_figures(test_y, test_prob, out_dir / 'figures', cfg['experiments'].get('figure_dpi', 600))
        training_figure(history, out_dir / 'figures', cfg['experiments'].get('figure_dpi', 600))
    return output
