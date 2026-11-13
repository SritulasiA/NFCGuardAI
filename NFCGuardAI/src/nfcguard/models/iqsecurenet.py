"""IQSecureNet: exact six-block manuscript topology for two-channel waveforms."""
from __future__ import annotations
import torch
from torch import nn


class SE1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        s = self.net(self.gap(x).squeeze(-1)).unsqueeze(-1)
        return x * s


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, use_se=False):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError('Primary IQSecureNet uses odd kernels for exact same padding')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.se = SE1D(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.act(self.bn(self.conv(x))))


class IQSecureNet(nn.Module):
    def __init__(self, in_channels=2, widths=(32, 64, 128, 256, 304, 384),
                 kernels=(7, 5, 5, 5, 3, 3), strides=(1, 2, 2, 2, 1, 1),
                 dropout=.3, hidden=128, pooling='gap', squeeze_excitation=False,
                 sequence_length=1800):
        super().__init__()
        if len(widths) != len(kernels) or len(widths) != len(strides):
            raise ValueError('Each convolution must have width, kernel and stride')
        if not widths:
            raise ValueError('Need at least one convolution block')
        self.in_channels = int(in_channels)
        self.widths = list(widths)
        self.kernels = list(kernels)
        self.strides = list(strides)
        self.pooling = pooling
        self.blocks = nn.ModuleList()
        old = in_channels
        for w, k, s in zip(widths, kernels, strides):
            self.blocks.append(ConvBlock(old, int(w), int(k), int(s), use_se=squeeze_excitation))
            old = int(w)
        if pooling == 'gap':
            self.pool = nn.AdaptiveAvgPool1d(1)
            head_features = old
        elif pooling == 'mean_max':
            self.pool = None
            head_features = 2 * old
        elif pooling == 'flatten':
            self.pool = nn.AdaptiveAvgPool1d(8)
            head_features = 8 * old
        else:
            raise ValueError('Unknown pooling method')
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(head_features, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def features(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self.features(x)
        if self.pooling == 'mean_max':
            x = torch.cat([x.mean(dim=-1), x.max(dim=-1).values], dim=-1)
        else:
            x = self.pool(x).flatten(1)
        return self.fc2(self.dropout2(self.relu(self.fc1(self.dropout1(x))))).squeeze(1)

    def receptive_field(self):
        field, jump = 1, 1
        for k, s in zip(self.kernels, self.strides):
            field += (k - 1) * jump
            jump *= s
        return field
