"""1D Grad-CAM and controlled attribution fidelity analyses."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ..dataset.preprocess import normalize_trace
from ..training.trainer import load_checkpoint
from ..utils import save_json


def gradcam_1d(model, batch):
    """Grad-CAM on final Conv1D block, interpolate at input's time resolution."""
    if not hasattr(model, 'blocks'):
        raise ValueError('Grad-CAM requires an IQSecureNet model with convolutional blocks')
    activations = {}
    gradients = {}

    def get_activations(module, ins, outs):
        activations['tensor'] = outs
        outs.register_hook(lambda grad: gradients.__setitem__('tensor', grad))

    handle = model.blocks[-1].register_forward_hook(get_activations)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logit = model(batch)
            logit.sum().backward()
            weights = gradients['tensor'].mean(dim=2, keepdim=True)
            heat = torch.relu((weights * activations['tensor']).sum(dim=1, keepdim=True))
            heat = F.interpolate(heat, size=batch.shape[-1], mode='linear', align_corners=False).squeeze(1)
            heat_min = heat.amin(dim=1, keepdim=True)
            heat_max = heat.amax(dim=1, keepdim=True)
            return ((heat - heat_min) / (heat_max - heat_min + 1e-8)).detach().cpu().numpy()
    finally:
        handle.remove()
        model.train(was_training)


def attribution_fidelity(model, signal, saliency, steps=20, device='cpu'):
    """Deletion/insertion, 0-baseline: measures changes to predicted relay probability."""
    n = signal.shape[-1]
    order = np.argsort(saliency)[::-1]
    y_del, y_ins, fractions = [], [], []
    with torch.inference_mode():
        for fraction in np.linspace(0, 1, steps + 1):
            count = min(n, int(round(fraction * n)))
            mask = np.zeros(n, dtype=bool)
            mask[order[:count]] = True
            deleting = signal.copy()
            inserting = np.zeros_like(signal)
            deleting[:, mask] = 0
            inserting[:, mask] = signal[:, mask]
            pair = torch.tensor(np.stack([deleting, inserting]), dtype=torch.float32, device=device)
            scores = torch.sigmoid(model(pair)).cpu().numpy()
            y_del.append(float(scores[0]))
            y_ins.append(float(scores[1]))
            fractions.append(float(fraction))
    return {'fractions':fractions, 'deletion':y_del, 'insertion':y_ins,
            'deletion_auc':float(np.trapezoid(y_del, fractions)) if hasattr(np,'trapezoid') else float(np.trapz(y_del, fractions)),
            'insertion_auc':float(np.trapezoid(y_ins, fractions)) if hasattr(np,'trapezoid') else float(np.trapz(y_ins, fractions))}


def run_explanation(store, sample_id, checkpoint, output_dir, trials=5, seed=42, dpi=600):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_checkpoint(checkpoint, device=device)
    if ckpt['model_name'] != 'iqsecurenet':
        raise ValueError('The standard Grad-CAM adapter expects IQSecureNet')
    idx = int(sample_id)
    if idx < 0 or idx >= len(store.X):
        raise IndexError('Sample ID outside store')
    raw = np.asarray(store.X[idx], dtype=np.float32)
    x = normalize_trace(raw, ckpt['config']['preprocessing']['mode'], ckpt['global_stats'])
    arr = x.T.copy()
    tensor = torch.from_numpy(arr[None]).to(device)
    saliency = gradcam_1d(model, tensor)[0]
    fidelity = attribution_fidelity(model, arr, saliency, device=device)
    rng = np.random.default_rng(seed)
    similarities = []
    for trial in range(trials):
        perturbed = arr + rng.normal(0, .005, arr.shape).astype(np.float32)
        heat_new = gradcam_1d(model, torch.from_numpy(perturbed[None]).to(device))[0]
        if saliency.std() > 0 and heat_new.std() > 0:
            similarities.append(float(np.corrcoef(saliency, heat_new)[0,1]))
    # Controlled temporal/phase perturbation and attribution shift indicators.
    for shift in (-8, 8):
        moved = np.roll(arr, shift, axis=1)
        heat_new = gradcam_1d(model, torch.from_numpy(moved[None].copy()).to(device))[0]
        moved_cam = np.roll(heat_new, -shift)
        if saliency.std() > 0 and moved_cam.std() > 0:
            similarities.append(float(np.corrcoef(saliency, moved_cam)[0,1]))
    result = {'sample_index':idx, 'label':int(store.labels()[idx]),
              'subtype':int(store.subtypes()[idx]), 'fidelity':fidelity,
              'attribution_stability_pearson':float(np.mean(similarities)) if similarities else None,
              'trials':trials, 'temperature':ckpt['temperature']}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f'gradcam_{idx}.json', result)
    time_us = np.arange(len(saliency)) / ckpt['config']['dataset'].get('sample_rate',10000000) * 1e6
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout='constrained')
    ax1.plot(time_us, arr[0], label='I / channel 1', lw=.7)
    if len(arr) == 2:
        ax1.plot(time_us, arr[1], label='Q / channel 2', lw=.7, alpha=.8)
    ax1.legend(fontsize=8)
    ax1.set_ylabel('Normalized amplitude')
    ax2.plot(time_us, saliency, label='Grad-CAM', lw=1.1)
    ax2.fill_between(time_us, 0, saliency, alpha=.24)
    ax2.set_ylabel('Attribution (0–1)')
    ax2.set_xlabel('Time (µs)')
    fig.savefig(output_dir / f'gradcam_{idx}.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.plot(fidelity['fractions'], fidelity['deletion'], marker='o', label='Deletion')
    ax.plot(fidelity['fractions'], fidelity['insertion'], marker='s', label='Insertion')
    ax.set(xlabel='Fraction of highest-attribution samples modified', ylabel='Predicted relay probability')
    ax.legend()
    fig.savefig(output_dir / f'gradcam_fidelity_{idx}.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return result
