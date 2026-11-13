"""Model factories, operation estimates and checkpoint-compatible names."""
from __future__ import annotations
import copy
import torch
from torch import nn
from .iqsecurenet import IQSecureNet
from .baselines import CNNBaseline, ResNet1D18, BiLSTMBaseline, OriginalRelayCNN

NEURAL_MODELS = ('iqsecurenet', 'cnn', 'resnet18', 'bilstm', 'cnn_se', 'original_cnn')
CLASSICAL_MODELS = ('svm', 'random_forest')


def create_model(name, config=None, in_channels=2, sequence_length=1800):
    cfg = copy.deepcopy(config or {})
    name = str(name).lower()
    if name == 'iqsecurenet':
        return IQSecureNet(in_channels=in_channels, sequence_length=sequence_length, **cfg)
    if name == 'cnn':
        return CNNBaseline(in_channels)
    if name == 'resnet18':
        return ResNet1D18(in_channels)
    if name == 'bilstm':
        return BiLSTMBaseline(in_channels)
    if name == 'cnn_se':
        return CNNBaseline(in_channels, se=True)
    if name == 'original_cnn':
        return OriginalRelayCNN(in_channels)
    raise ValueError(f'Unknown neural architecture {name}. Valid: {NEURAL_MODELS}')


def complexity(model, input_shape):
    """Estimate Conv1d and Linear MACs by forward hooks; 2 FLOPs per MAC.
    Normalization, activation, pooling are not counted in FLOPs estimate.
    """
    result = {'parameters': sum(p.numel() for p in model.parameters()), 'macs': 0}
    hooks = []

    def measure(mod, inputs, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(mod, nn.Conv1d):
            per_output = (mod.in_channels // mod.groups) * mod.kernel_size[0]
            result['macs'] += int(out.numel() * per_output)
        elif isinstance(mod, nn.Linear):
            result['macs'] += int(out.numel() * mod.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_hook(measure))
    device = next(model.parameters()).device
    was_train = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(*input_shape, device=device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_train)
    result['flops'] = 2 * result['macs']
    result['fp32_weights_mb'] = result['parameters'] * 4 / 1e6
    if hasattr(model, 'receptive_field'):
        result['receptive_field'] = model.receptive_field()
    return result
