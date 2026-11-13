"""Safely inspect dataset files, including long WAV files segmented at ATQA frame boundaries.

Aligned segmentation of recordings is an assumption; NOT a universal ATQA event detector.
For continuous recordings with unknown boundaries, supply pre-extracted traces.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import re
import zipfile
import numpy as np
import soundfile as sf
import scipy.io
import yaml

EXTENSIONS = {'.wav', '.flac', '.npy', '.npz', '.csv', '.mat'}


def extract_archive(source: str | Path, target: str | Path, max_expanded_gb=12):
    src, dst = Path(source), Path(target)
    if src.is_dir():
        return src
    if not zipfile.is_zipfile(src):
        raise ValueError(f'Not a readable ZIP or directory: {src}')
    dst.mkdir(parents=True, exist_ok=True)
    base = dst.resolve()
    with zipfile.ZipFile(src) as z:
        total = sum(i.file_size for i in z.infolist() if not i.is_dir())
        if total > max_expanded_gb * 1024**3:
            raise ValueError('Unpacked archive exceeds configured maximum size')
        for info in z.infolist():
            dest = (dst / info.filename).resolve()
            if not dest.is_relative_to(base):
                raise ValueError('Unsafe ZIP path detected')
            # block symbolic links and similar entry types
            filetype = (info.external_attr >> 16) & 0o170000
            if filetype == 0o120000:
                raise ValueError('Archive contains a symlink; refusing extraction')
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.stat().st_size == info.file_size:
                    continue
                with z.open(info) as fin, dest.open('wb') as fout:
                    import shutil
                    shutil.copyfileobj(fin, fout)
    return dst


def iter_files(root):
    return sorted((p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: str(p).lower())


def _read_array(path):
    suffix = path.suffix.lower()
    if suffix == '.npy':
        return np.load(path, mmap_mode='r', allow_pickle=False)
    if suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            keys = [k for k in ('iq', 'signals', 'x', 'X', 'waveform', 'arr_0') if k in data]
            if not keys:
                keys = [k for k in data.keys() if np.asarray(data[k]).ndim >= 1]
            if not keys:
                raise ValueError(f'No waveform arrays in {path}')
            return np.array(data[keys[0]])
    if suffix == '.mat':
        try:
            mat = scipy.io.loadmat(path)
        except NotImplementedError as e:
            raise ValueError('MATLAB v7.3 requires conversion to NPY first') from e
        candidates = [(k, v) for k, v in mat.items() if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim >= 1]
        if not candidates:
            raise ValueError(f'No arrays found in {path}')
        return max(candidates, key=lambda kv: kv[1].size)[1]
    if suffix == '.csv':
        try:
            data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        except ValueError:
            data = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
        return data
    raise ValueError(f'Unsupported numeric file: {path}')


def inspect_layout(path, length=1800, strict_lengths=True):
    """Return (num_frames, channels, sample_rate, layout, waveform_length).

    layout='timeseries' for [T] or [T,C]; 'batch' for [B,T] or [B,T,C].
    """
    path = Path(path)
    if path.suffix.lower() in ('.wav', '.flac'):
        info = sf.info(str(path))
        total, channels, rate = int(info.frames), int(info.channels), int(info.samplerate)
        if total < length or (strict_lengths and total % length):
            raise ValueError(f'{path}: {total} samples cannot be split into aligned {length}-sample traces')
        if channels not in (1, 2):
            raise ValueError(f'{path}: expected 1 or 2 signal channels, got {channels}')
        return total // length, channels, rate, 'audio', total
    arr = _read_array(path)
    shape = arr.shape
    if np.iscomplexobj(arr):
        if arr.ndim == 1:
            channels, total, layout = 2, len(arr), 'complex_series'
            if total < length or (strict_lengths and total % length):
                raise ValueError(f'{path}: complex length {total} is not divisible by {length}')
            return total // length, channels, None, layout, total
        if arr.ndim == 2 and shape[1] == length:
            return shape[0], 2, None, 'complex_batch', shape[1]
        if arr.ndim == 2 and shape[0] == length:
            return shape[1], 2, None, 'complex_batch_transposed', length
    if arr.ndim == 1:
        total = shape[0]
        if total < length or (strict_lengths and total % length):
            raise ValueError(f'{path}: expected aligned length divisible by {length}; got {total}')
        return total // length, 1, None, 'mono_series', total
    if arr.ndim == 2:
        if shape[1] in (1, 2) and shape[0] >= length:
            if strict_lengths and shape[0] % length:
                raise ValueError(f'{path}: first axis not divisible by {length}')
            return shape[0] // length, shape[1], None, 'channel_series', shape[0]
        if shape[0] in (1, 2) and shape[1] >= length:
            if strict_lengths and shape[1] % length:
                raise ValueError(f'{path}: second axis not divisible by {length}')
            return shape[1] // length, shape[0], None, 'channel_first_series', shape[1]
        if shape[1] == length:
            return shape[0], 1, None, 'mono_batch', shape[1]
        if shape[0] == length:
            return shape[1], 1, None, 'mono_batch_transposed', shape[0]
    if arr.ndim == 3:
        if shape[1] == length and shape[2] in (1, 2):
            return shape[0], shape[2], None, 'batch_channel_last', shape[1]
        if shape[2] == length and shape[1] in (1, 2):
            return shape[0], shape[1], None, 'batch_channel_first', shape[2]
    raise ValueError(f'Unsupported waveform shape {shape} in {path}')


def read_trace(path, offset, length, layout, opened_array=None):
    """Offset is a zero-based trace index (not the sample offset). Return [T,C] float32."""
    path = Path(path)
    a, b = int(offset) * length, (int(offset) + 1) * length
    if layout == 'audio':
        with sf.SoundFile(path) as wave:
            wave.seek(a)
            out = wave.read(length, dtype='float32', always_2d=True)
    else:
        arr = opened_array if opened_array is not None else _read_array(path)
        if layout in ('complex_series', 'complex_batch', 'complex_batch_transposed'):
            if layout == 'complex_series':
                c = arr[a:b]
            elif layout == 'complex_batch':
                c = arr[offset]
            else:
                c = arr[:, offset]
            out = np.column_stack([np.real(c), np.imag(c)])
        elif layout == 'mono_series':
            out = arr[a:b, None]
        elif layout == 'channel_series':
            out = arr[a:b, :]
        elif layout == 'channel_first_series':
            out = arr[:, a:b].T
        elif layout == 'mono_batch':
            out = arr[offset, :, None]
        elif layout == 'mono_batch_transposed':
            out = arr[:, offset, None]
        elif layout == 'batch_channel_last':
            out = arr[offset]
        elif layout == 'batch_channel_first':
            out = arr[offset].T
        else:
            raise ValueError(f'Unknown layout: {layout}')
    out = np.asarray(out, dtype=np.float32)
    if out.shape[0] != length or out.shape[1] not in (1, 2):
        raise ValueError(f'Malformed trace read {path}: {out.shape}')
    if not np.isfinite(out).all():
        raise ValueError(f'Non-finite signal values in {path} at offset {offset}')
    return out


def _extract_id(path_text, kind, override=None):
    if override:
        m = re.search(override, path_text)
        return m.group(1) if m else None
    if kind == 'tag':
        m = re.search(r'(?:tag|card|mifare|normal|legit|wired?)[_\s-]*0*(\d+)(?:\D|$)', path_text, re.I)
        return m.group(1) if m else None
    m = re.search(r'(?:session|phase|week|day|recording)[_\s-]*0*(\d+)(?:\D|$)', path_text, re.I)
    return m.group(1) if m else None


def infer_label(rel_path, label_map=None):
    raw = str(rel_path).replace('\\', '/')
    parts = raw.split('/')
    # Dataset root can be named NFC_Relay. Disambiguate class from filename and child folders.
    name = parts[-1]
    local = '/'.join(parts[-3:])
    tag_regex, session_regex = None, None
    if label_map:
        tag_regex = label_map.get('tag_regex')
        session_regex = label_map.get('session_regex')
        for rule in label_map.get('rules', []):
            if re.search(rule['pattern'], raw):
                subtype = rule['subtype'].lower()
                if subtype not in ('legitimate', 'wired', 'wireless'):
                    raise ValueError(f'Unrecognized mapped subtype {subtype}')
                tag = str(rule['tag_id']) if rule.get('tag_id') is not None else None
                if tag is None and rule.get('tag_from_path', False):
                    tag = _extract_id(local, 'tag', tag_regex)
                session = str(rule['session_id']) if rule.get('session_id') is not None else _extract_id(local, 'session', session_regex)
                return subtype, tag, session
    if re.search(r'normal|legitimate|genuine|original|direct', name, re.I) or any(re.search(r'^(normal|legitimate|genuine)$', p, re.I) for p in parts[-3:-1]):
        subtype = 'legitimate'
    elif re.search(r'wireless|wi[-_ ]?fi|wifi|bluetooth', local, re.I):
        subtype = 'wireless'
    elif re.search(r'wired|wire[-_ ]?relay|cable', local, re.I):
        subtype = 'wired'
    else:
        raise ValueError(f'Cannot infer legitimate/wired/wireless label from {raw}; provide --label-map')
    return subtype, _extract_id(local, 'tag', tag_regex), _extract_id(local, 'session', session_regex)


def load_label_map(path):
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if 'rules' not in cfg:
        raise ValueError('label map must include rules list')
    return cfg


def source_report(source, length=1800, label_map=None, strict_lengths=True):
    """For ZIP sources, reports names and uncompressed sizes without extracting.
    Format/channel inspection requires extraction; use prepare for validation.
    """
    source = Path(source)
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            entries = [i for i in z.infolist() if Path(i.filename).suffix.lower() in EXTENSIONS]
            return {'source': str(source), 'archive': True, 'waveform_files': len(entries),
                    'expanded_bytes': sum(i.file_size for i in entries),
                    'examples': [i.filename for i in entries[:18]],
                    'warning': 'ZIP names only. Use prepare to verify wave shapes, channels and labels.'}
    label_cfg = load_label_map(label_map)
    files = iter_files(source)
    summary = Counter()
    examples = []
    errors = []
    for p in files:
        try:
            n, ch, rate, layout, total = inspect_layout(p, length, strict_lengths)
            subtype, tag, session = infer_label(p.relative_to(source), label_cfg)
            summary[subtype] += n
            if len(examples) < 16:
                examples.append(dict(path=str(p.relative_to(source)), frames=n, channels=ch, sample_rate=rate, subtype=subtype, tag_id=tag, session_id=session))
        except Exception as e:
            errors.append(f'{p}: {e}')
    return {'source': str(source), 'archive': False, 'files': len(files),
            'class_counts': dict(summary), 'total_frames': sum(summary.values()),
            'examples': examples, 'errors': errors[:50], 'error_count': len(errors)}
