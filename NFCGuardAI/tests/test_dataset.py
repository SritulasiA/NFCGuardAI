from pathlib import Path
import numpy as np
import pytest
from nfcguard.dataset.readers import inspect_layout, infer_label, read_trace, source_report
from nfcguard.dataset.preprocess import normalize_trace, perturb_trace, fit_global_stats
from nfcguard.dataset.store import prepare_store, SignalStore
from nfcguard.dataset.torch_data import NFCWaveDataset, SubtypeBatchSampler


def test_labels_and_shapes(tmp_path):
    source=tmp_path/'raw'
    source.mkdir()
    np.save(source/'normal_tag1.npy',np.random.default_rng(7).normal(size=(12,1800,2)).astype('float32'))
    np.save(source/'wired_tag1.npy',np.random.default_rng(8).normal(size=(12,1800,2)).astype('float32'))
    np.save(source/'wifi.npy',np.random.default_rng(9).normal(size=(12,1800,2)).astype('float32'))
    assert inspect_layout(source/'normal_tag1.npy')[0:2] == (12,2)
    assert infer_label('wired_tag1.npy') == ('wired','1',None)
    stats=prepare_store(source,tmp_path/'store',length=1800,channels=2)
    assert stats['total']==36
    assert stats['channels']==2
    store=SignalStore(tmp_path/'store')
    assert store.X.shape==(36,1800,2)
    assert sorted(store.meta.subtype.unique())==['legitimate','wired','wireless']
    d=NFCWaveDataset(store,[0,1],augment=True)
    assert d[0][0].shape==(2,1800)


def test_mono_is_not_fabricated(tmp_path):
    source=tmp_path/'raw'
    source.mkdir()
    np.save(source/'normal_tag1.npy',np.zeros((4,1800),dtype=np.float32))
    result=prepare_store(source,tmp_path/'out')
    assert result['channels']==1
    assert 'NOT measured I/Q' in result['caveat']
    with pytest.raises(ValueError):
        prepare_store(source,tmp_path/'out2',channels=2)


def test_normalization_and_fading():
    rng=np.random.default_rng(42)
    wave=rng.normal(1,4,(1800,2)).astype(np.float32)
    out=normalize_trace(wave)
    assert np.allclose(out.mean(0),0,atol=1e-6)
    assert np.allclose(out.std(0),1,atol=1e-6)
    joint=normalize_trace(wave,'joint_complex')
    assert np.allclose(np.mean(np.sum(joint**2,axis=1)),1,atol=1e-5)
    altered=perturb_trace(wave,'cfo',100,rng)
    assert altered.shape==wave.shape
    with pytest.raises(ValueError):
        perturb_trace(wave[:,:1],'cfo',100,rng)


def test_sampler_no_replacements():
    arr=np.array([0]*30+[1]*20+[2]*10)
    sampler=SubtypeBatchSampler(arr,batch_size=12,seed=42)
    seq=[i for batch in sampler for i in batch]
    assert len(seq)==len(arr)
    assert len(set(seq))==len(arr)
