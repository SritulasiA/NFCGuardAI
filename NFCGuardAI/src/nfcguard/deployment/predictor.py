"""Checkpoint-backed offline and file-stream inference. No SDR hardware is bundled."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from ..dataset.readers import inspect_layout, read_trace, _read_array, EXTENSIONS
from ..dataset.preprocess import normalize_trace
from ..training.trainer import load_checkpoint
from ..utils import utc_now


class NFCPredictor:
    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self.config = self.meta['config']
        self.temperature = float(self.meta['temperature'])
        self.threshold = float(self.config['calibration'].get('threshold',.5))
        self.channels = int(self.meta['channels'])
        self.length = int(self.meta['length'])

    def preprocess(self, trace):
        x = np.asarray(trace)
        if np.iscomplexobj(x):
            x = np.column_stack((x.real, x.imag))
        if x.ndim == 1:
            x = x[:, None]
        if x.shape == (self.channels, self.length):
            x = x.T
        if x.shape != (self.length, self.channels):
            raise ValueError(f'Expected [{self.length},{self.channels}] trace. Received {x.shape}')
        if not np.isfinite(x).all():
            raise ValueError('Trace contains NaN or infinite samples')
        return normalize_trace(x, self.config['preprocessing']['mode'], self.meta['global_stats'],
                               self.config['preprocessing'].get('epsilon',1e-6))

    def predict(self, trace, signal_id=None):
        started = time.perf_counter()
        x = self.preprocess(trace)
        tensor = torch.from_numpy(np.ascontiguousarray(x.T[None])).float().to(self.device)
        with torch.inference_mode():
            logit = float(self.model(tensor).item())
        probability = float(torch.sigmoid(torch.tensor(logit / self.temperature)).item())
        return dict(signal_id=str(signal_id) if signal_id is not None else None,
                    timestamp=utc_now(), relay_probability=probability,
                    decision='relay_attack' if probability >= self.threshold else 'legitimate',
                    label=int(probability >= self.threshold), threshold=self.threshold,
                    processing_ms=(time.perf_counter() - started)*1000,
                    waveform_shape=[self.length,self.channels],
                    action='alert' if probability >= self.threshold else 'pass',
                    note='Software decision only; host reader integration needed to enforce a block')

    def predict_file(self, input_path, log_path=None):
        path = Path(input_path)
        count, channel, _, layout, _ = inspect_layout(path, self.length, strict_lengths=True)
        if channel != self.channels:
            raise ValueError(f'Input {path} has {channel} channels but checkpoint requires {self.channels}')
        out = []
        opened = None if layout == 'audio' else _read_array(path)
        for offset in range(count):
            trace = read_trace(path, offset, self.length, layout, opened_array=opened)
            record = self.predict(trace, signal_id=f'{path.name}:{offset}')
            out.append(record)
            if log_path:
                logfile = Path(log_path)
                logfile.parent.mkdir(parents=True,exist_ok=True)
                with logfile.open('a', encoding='utf-8') as fout:
                    fout.write(json.dumps(record)+'\n')
        return out

    def predict_directory(self, path, log_path=None):
        """Use pre-extracted files as an ordered stream; does not pretend to tap a live reader."""
        files = sorted(p for p in Path(path).rglob('*') if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError('No supported signal files in the supplied directory')
        for f in files:
            for pred in self.predict_file(f, log_path):
                yield pred
