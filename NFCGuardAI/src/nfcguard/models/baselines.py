"""Comparative neural models. Original published CNN is an approximate adaptation.

The manuscript does not fully specify every baseline layer. These are documented,
repeatable implementations, not claims of byte-for-byte reproduction.
"""
from __future__ import annotations
import torch
from torch import nn
from .iqsecurenet import SE1D


class CNNBaseline(nn.Module):
    def __init__(self, in_channels=2, se=False, dropout=.3):
        super().__init__()
        stages = []
        old = in_channels
        for width, kernel in ((64, 7), (128, 5), (256, 5)):
            stages.extend([nn.Conv1d(old, width, kernel, padding=kernel // 2), nn.BatchNorm1d(width),
                           nn.ReLU(inplace=True), SE1D(width) if se else nn.Identity(), nn.MaxPool1d(4)])
            old = width
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(old * 16, 384), nn.ReLU(), nn.Dropout(dropout), nn.Linear(384, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, input_c, output_c, stride=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(input_c, output_c, 3, stride=stride, padding=1, bias=False),
                                 nn.BatchNorm1d(output_c), nn.ReLU(inplace=True),
                                 nn.Conv1d(output_c, output_c, 3, padding=1, bias=False), nn.BatchNorm1d(output_c))
        self.shortcut = (nn.Identity() if input_c == output_c and stride == 1 else
                         nn.Sequential(nn.Conv1d(input_c, output_c, 1, stride=stride, bias=False), nn.BatchNorm1d(output_c)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))


class ResNet1D18(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1))
        stages = []
        last = 64
        for width, step in ((64, 1), (128, 2), (256, 2), (512, 2)):
            stages += [ResidualBlock(last, width, step), ResidualBlock(width, width)]
            last = width
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, 1))

    def forward(self, x):
        return self.head(self.features(self.stem(x))).squeeze(1)


class BiLSTMBaseline(nn.Module):
    def __init__(self, in_channels=2, hidden=64):
        super().__init__()
        self.downsample = nn.AvgPool1d(8, stride=8)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden, num_layers=2,
                            dropout=.3, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.downsample(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1)).squeeze(1)


class OriginalRelayCNN(nn.Module):
    """Binary comparison using published filter counts/kernels (72/8, 64/6, 32/6).
    Pooling and dense-head details are adapted; not an exact original author checkpoint.
    """
    def __init__(self, in_channels=2):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(in_channels, 72, 8, padding=3), nn.ReLU(), nn.MaxPool1d(3),
                                 nn.Conv1d(72, 64, 6, padding=2), nn.ReLU(), nn.MaxPool1d(3),
                                 nn.Conv1d(64, 32, 6, padding=2), nn.ReLU(), nn.MaxPool1d(3),
                                 nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(32 * 16, 256),
                                 nn.ReLU(), nn.Dropout(.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)
