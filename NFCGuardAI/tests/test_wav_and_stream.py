import numpy as np
import soundfile as sf
from nfcguard.dataset.readers import inspect_layout, read_trace
from nfcguard.deployment.streaming import SignalEvent, NFCGuardStream


def test_wav_segmenting_and_mono(tmp_path):
    rng=np.random.default_rng(42)
    wave=rng.uniform(-.5,.5,size=(3600,1)).astype('float32')
    sf.write(tmp_path/'normal_tag1.wav',wave,10000000,subtype='PCM_16')
    count,channels,rate,layout,total=inspect_layout(tmp_path/'normal_tag1.wav')
    assert (count,channels,rate,layout,total)==(2,1,10000000,'audio',3600)
    frame=read_trace(tmp_path/'normal_tag1.wav',1,1800,layout)
    assert frame.shape==(1800,1)
    assert np.max(np.abs(frame-wave[1800:]))<1e-4
