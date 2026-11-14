
from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from .readers import extract_archive, iter_files, inspect_layout, infer_label, load_label_map, read_trace, _read_array
from ..utils import save_json

LOGGER = logging.getLogger(__name__)
SUBTYPE_ID = {'legitimate': 0, 'wired': 1, 'wireless': 2}


def prepare_store(source, target, length=1800, channels='auto', label_map=None,
                  strict_lengths=True, cache_dtype='float16', force=False, max_expanded_gb=12):
    target = Path(target)
    if (target / 'signals.npy').exists() and not force:
        raise FileExistsError(f'{target}/signals.npy exists. Use --force to rebuild.')
    target.mkdir(parents=True, exist_ok=True)
    root = extract_archive(source, target / 'raw', max_expanded_gb=max_expanded_gb)
    paths = iter_files(root)
    if not paths:
        raise ValueError(f'No supported signal files under {root}')
    mapping = load_label_map(label_map)
    infos = []
    missing = []
    for path in tqdm(paths, desc='Inspecting signal files'):
        try:
            count, ch, rate, layout, total = inspect_layout(path, int(length), strict_lengths)
            subtype, tag, session = infer_label(path.relative_to(root), mapping)
            infos.append((path, count, ch, rate, layout, subtype, tag, session))
        except Exception as e:
            missing.append(f'{path.relative_to(root)}: {e}')
    if missing:
        raise ValueError(f'{len(missing)} files could not be verified. No training cache created.\n' + '\n'.join(missing[:25]))
    ch_set = sorted(set(row[2] for row in infos))
    if len(ch_set) != 1:
        raise ValueError(f'Mixed channel counts in dataset {ch_set}; convert to a homogeneous set before training')
    n_channels = ch_set[0]
    if channels != 'auto' and int(channels) != n_channels:
        raise ValueError(f'Specified {channels} channels, actual files have {n_channels}; refusing to fabricate or discard channels')
    n = sum(row[1] for row in infos)
    dtype = np.dtype(cache_dtype)
    if dtype not in (np.dtype('float16'), np.dtype('float32')):
        raise ValueError('Cache dtype must be float16 or float32')
    tmp = target / 'signals.partial.npy'
    if tmp.exists():
        tmp.unlink()
    arr = open_memmap(tmp, mode='w+', dtype=dtype, shape=(n, length, n_channels))
    metadata = []
    cursor = 0
    try:
        for path, count, ch, rate, layout, subtype, tag, session in tqdm(infos, desc='Preparing signals'):
            opened = None if layout == 'audio' else _read_array(path)
            for offset in range(count):
                wave = read_trace(path, offset, length, layout, opened_array=opened)
                arr[cursor] = wave.astype(dtype)
                metadata.append(dict(index=cursor, source=str(path.relative_to(root)), offset=offset,
                                     subtype=subtype, subtype_id=SUBTYPE_ID[subtype],
                                     label=int(subtype != 'legitimate'),
                                     tag_id=tag or '', session_id=session or '',
                                     source_channels=ch, sample_rate=rate or '',
                                     layout=layout))
                cursor += 1
        arr.flush()
        del arr
        tmp.replace(target / 'signals.npy')
    except Exception:
        del arr
        tmp.unlink(missing_ok=True)
        raise
    df = pd.DataFrame(metadata)
    df.to_csv(target / 'metadata.csv', index=False)
    counts = {str(k): int(v) for k, v in df['subtype'].value_counts().to_dict().items()}
    stats = {'total': n, 'length': length, 'channels': n_channels, 'counts': counts,
             'source': str(source), 'dtype': str(dtype), 'waveform_files': len(paths),
             'known_tag_records': int((df['tag_id'] != '').sum()),
             'known_session_records': int((df['session_id'] != '').sum()),
             'sample_rate_expected': 10000000,
             'caveat': ('One-channel waveform, NOT measured I/Q; IQSecureNet runs in mono-compatibility mode.' if n_channels == 1 else 'Two channels present. Verify independently that channels are measured I and Q, not generic stereo.')}
    save_json(target / 'manifest.json', stats)
    return stats


class SignalStore:
    def __init__(self, root):
        root = Path(root)
        self.root = root
        self.X = np.load(root / 'signals.npy', mmap_mode='r', allow_pickle=False)
        self.meta = pd.read_csv(root / 'metadata.csv', keep_default_na=False)
        with (root / 'manifest.json').open('r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        if self.X.ndim != 3 or len(self.X) != len(self.meta):
            raise ValueError('Corrupt signal store: input tensor and metadata disagree')
        if not np.array_equal(self.meta['index'].to_numpy(dtype=int), np.arange(len(self.meta))):
            raise ValueError('Metadata index does not match tensor offsets')

    @property
    def length(self):
        return int(self.X.shape[1])

    @property
    def channels(self):
        return int(self.X.shape[2])

    def labels(self):
        return self.meta['label'].to_numpy(dtype=np.int64)

    def subtypes(self):
        return self.meta['subtype_id'].to_numpy(dtype=np.int64)

    def group_ids(self, name):
        if name not in ('tag_id', 'session_id'):
            raise ValueError('Only observed tag_id and session_id metadata can define disjoint splits')
        groups = self.meta[name].astype(str).to_numpy()
        if any(x == '' or x.lower() in ('none', 'nan') for x in groups):
            raise ValueError(f'{name} incomplete: genuine group-disjoint evaluation is impossible without verified metadata')
        if len(set(groups)) < 2:
            raise ValueError(f'{name} must contain at least two distinct groups')
        return groups
