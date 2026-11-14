from __future__ import annotations
import numpy as np


def fit_global_stats(X, indices, chunk_size=256, epsilon=1e-6):
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError('Cannot fit normalizer on empty training set')
    count = 0
    sums = None
    sums2 = None
    for start in range(0, len(indices), chunk_size):
        batch = np.asarray(X[indices[start:start + chunk_size]], dtype=np.float64)
        if sums is None:
            sums = np.zeros(batch.shape[-1], dtype=np.float64)
            sums2 = np.zeros_like(sums)
        sums += batch.sum(axis=(0, 1))
        sums2 += (batch * batch).sum(axis=(0, 1))
        count += batch.shape[0] * batch.shape[1]
    mean = sums / count
    std = np.sqrt(np.maximum(sums2 / count - mean**2, 0.0))
    return {'mean': mean.tolist(), 'std': np.maximum(std, epsilon).tolist()}


def normalize_trace(trace, mode='per_channel', stats=None, eps=1e-6):
    x = np.asarray(trace, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] not in (1, 2):
        raise ValueError('Trace must have shape [time,1] or [time,2]')
    if mode == 'none':
        return x.copy()
    if mode == 'per_channel':
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        return (x - mu) / np.maximum(sd, eps)
    if mode == 'joint_complex':
        mu = x.mean(axis=0, keepdims=True)
        centered = x - mu
        rms = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
        return centered / max(rms, eps)
    if mode == 'global':
        if stats is None:
            raise ValueError('Global normalization requires stats fitted only on the training set')
        mu = np.asarray(stats['mean'], dtype=np.float32)[None, :]
        sd = np.asarray(stats['std'], dtype=np.float32)[None, :]
        return (x - mu) / np.maximum(sd, eps)
    raise ValueError(f'Unknown normalization: {mode}')


def augment_wireless(x, rng, sample_rate=10000000):
    """Apply plausible small waveform perturbations before normalization; no dataset expansion."""
    y = np.asarray(x, dtype=np.float32).copy()
    n, channels = y.shape
    # Time stretch/resampling around center: +/- 1--2%, same output length.
    scale = 1 + float(rng.choice((-1, 1))) * rng.uniform(.01, .02)
    points = np.arange(n, dtype=np.float64)
    source = (points - (n - 1) / 2) / scale + (n - 1) / 2
    for c in range(channels):
        y[:, c] = np.interp(source, points, y[:, c], left=y[0, c], right=y[-1, c])
    y *= 1 + float(rng.choice((-1, 1))) * rng.uniform(.03, .05)
    if channels == 2:
        theta = rng.normal(0, np.deg2rad(rng.uniform(1, 3)), size=n).astype(np.float32)
        old_i, old_q = y[:, 0].copy(), y[:, 1].copy()
        y[:, 0] = old_i * np.cos(theta) - old_q * np.sin(theta)
        y[:, 1] = old_i * np.sin(theta) + old_q * np.cos(theta)
    snr = rng.uniform(20, 40)
    power = float(np.mean(y**2))
    if power > 0:
        y += rng.normal(0, np.sqrt(power / 10**(snr / 10)), size=y.shape).astype(np.float32)
    return y


def perturb_trace(x, kind, level, rng, sample_rate=10000000):
    """Reproducible test-only impairments, applied to a raw trace BEFORE normalization.

    level conventions: awgn=snr dB, fading=Rician K-factor, cfo=Hz,
    iq_imbalance=relative gain skew, interference=interference-to-signal dB.
    """
    y = np.array(x, dtype=np.float32, copy=True)
    n, ch = y.shape
    power = max(float(np.mean(y**2)), 1e-12)
    t = np.arange(n, dtype=np.float32) / sample_rate
    if kind == 'awgn':
        y += rng.normal(0, np.sqrt(power / 10**(float(level) / 10)), y.shape).astype(np.float32)
    elif kind == 'fading':
        if ch != 2:
            raise ValueError('Complex fading requires measured I and Q channels')
        k = max(0.0, float(level))
        gain = np.sqrt(k / (k + 1.)) + (rng.normal() + 1j * rng.normal()) / np.sqrt(2 * (k + 1.))
        z = (y[:, 0] + 1j * y[:, 1]) * gain
        y = np.column_stack((z.real, z.imag)).astype(np.float32)
    elif kind == 'cfo':
        if ch != 2:
            raise ValueError('Carrier frequency offset requires measured I and Q channels')
        z = (y[:, 0] + 1j * y[:, 1]) * np.exp(2j * np.pi * float(level) * t)
        y = np.column_stack((z.real, z.imag)).astype(np.float32)
    elif kind == 'iq_imbalance':
        if ch != 2:
            raise ValueError('I/Q imbalance requires measured I and Q channels')
        mismatch = float(level)
        angle = np.deg2rad(5.0)
        y[:, 0] *= 1 + mismatch
        y[:, 1] = (1 - mismatch) * (y[:, 1] * np.cos(angle) + y[:, 0] * np.sin(angle))
    elif kind == 'interference':
        ir = 10**(float(level) / 20)
        freq = rng.uniform(5000, 750000)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = np.sqrt(2 * power) * ir
        y[:, 0] += amplitude * np.cos(2 * np.pi * freq * t + phase)
        if ch == 2:
            y[:, 1] += amplitude * np.sin(2 * np.pi * freq * t + phase)
    else:
        raise ValueError(f'Unknown impairment kind: {kind}')
    return y
