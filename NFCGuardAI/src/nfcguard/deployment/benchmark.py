"""Explicit CPU inference profiling, independent of RF acquisition and reader actions."""
from __future__ import annotations
from pathlib import Path
import time
import platform
import numpy as np
import torch
from ..training.trainer import load_checkpoint
from ..dataset.preprocess import normalize_trace
from ..utils import save_json


def benchmark(checkpoint, out_dir, warmup=50, runs=200, threads=1, waveform=None):
    threads = int(threads)
    if threads > 0:
        torch.set_num_threads(threads)
    device = 'cpu'
    model, ckpt = load_checkpoint(checkpoint, device)
    length, channels = ckpt['length'], ckpt['channels']
    if waveform is None:
        # Timing input only, NOT a real-world detection result.
        rng = np.random.default_rng(42)
        waveform = rng.standard_normal((length,channels)).astype(np.float32)
    raw = np.asarray(waveform, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(normalize_trace(raw, ckpt['config']['preprocessing']['mode'],
                                      ckpt['global_stats']).T[None]))
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        times=[]
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            model(tensor)
            times.append((time.perf_counter_ns() - t0) / 1e6)
        full_times=[]
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            x=normalize_trace(raw, ckpt['config']['preprocessing']['mode'], ckpt['global_stats'])
            logit=model(torch.from_numpy(np.ascontiguousarray(x.T[None])))
            probability = torch.sigmoid(logit / ckpt['temperature'])
            classification=bool(probability[0].item() >= ckpt['config']['calibration'].get('threshold', .5))
            full_times.append((time.perf_counter_ns() - t0)/1e6)
    result=dict(cpu=platform.processor() or platform.machine(), platform=platform.platform(),
                torch_version=torch.__version__, num_threads=torch.get_num_threads(),
                warmup=warmup, runs=runs, batch_size=1, model_only_mean_ms=float(np.mean(times)),
                model_only_sd_ms=float(np.std(times,ddof=1)) if runs>1 else 0.,
                model_only_p95_ms=float(np.percentile(times,95)),
                preprocess_inference_decision_mean_ms=float(np.mean(full_times)),
                preprocess_inference_decision_p95_ms=float(np.percentile(full_times,95)),
                excluded=['SDR sampling/segmentation', 'reader stack enforcement', 'network alert delivery'],
                timer='perf_counter_ns wall clock, full 1800-sample trace',
                note='No fabricated embedded-reader measurement; model-only and software pipeline distinguished.')
    save_json(Path(out_dir)/'cpu_benchmark.json',result)
    return result
