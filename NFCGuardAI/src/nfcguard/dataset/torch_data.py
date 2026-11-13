"""Memory-mapped dataset and non-oversampling subtype-stratified mini-batches."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .preprocess import augment_wireless, normalize_trace


class NFCWaveDataset(Dataset):
    def __init__(self, store, indices, normalization='per_channel', global_stats=None,
                 augment=False, probability=.5, seed=42, eps=1e-6):
        self.store = store
        self.indices = np.asarray(indices, dtype=int)
        self.normalization = normalization
        self.global_stats = global_stats
        self.augment = augment
        self.probability = probability
        self.seed = seed
        self.epoch = 0
        self.eps = eps
        self.subtypes = store.subtypes()[self.indices]
        self.labels = store.labels()[self.indices]

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, local_index):
        idx = int(self.indices[local_index])
        wave = np.asarray(self.store.X[idx], dtype=np.float32)
        if self.augment and int(self.subtypes[local_index]) == 2:
            # independent of data-loader worker ordering
            rng = np.random.default_rng(self.seed + self.epoch * 1000003 + idx * 19)
            if rng.random() < self.probability:
                wave = augment_wireless(wave, rng)
        wave = normalize_trace(wave, mode=self.normalization, stats=self.global_stats, eps=self.eps)
        return (torch.from_numpy(np.ascontiguousarray(wave.T)),
                torch.tensor(float(self.labels[local_index]), dtype=torch.float32),
                torch.tensor(int(self.subtypes[local_index]), dtype=torch.long),
                torch.tensor(idx, dtype=torch.long))


class SubtypeBatchSampler(Sampler):
    """Approximately subtype-stratified batches, sampling every record once per epoch."""
    def __init__(self, subtypes, batch_size, seed=42):
        self.subtypes = np.asarray(subtypes)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError('Batch size must be positive')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(len(self.subtypes) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        unique, counts = np.unique(self.subtypes, return_counts=True)
        queues = {int(cls): rng.permutation(np.where(self.subtypes == cls)[0]).tolist() for cls in unique}
        proportion = counts / counts.sum()
        while any(queues.values()):
            batch = []
            cap = min(self.batch_size, sum(len(v) for v in queues.values()))
            # Prefer records from underrepresented subtypes within the current batch,
            # relative to the full training subtype frequencies.
            for _ in range(cap):
                available = [j for j, cls in enumerate(unique) if queues[int(cls)]]
                deficits = [float(proportion[j] * cap - sum(self.subtypes[k] == cls for k in batch)) for j, cls in enumerate(unique)]
                pick_j = max(available, key=lambda j: (deficits[j], rng.random() * 1e-5))
                batch.append(queues[int(unique[pick_j])].pop())
            rng.shuffle(batch)
            yield batch
