from pathlib import Path
import numpy as np
import torch
from nfcguard.models.iqsecurenet import IQSecureNet
from nfcguard.training.trainer import save_checkpoint
from nfcguard.deployment.predictor import NFCPredictor
from nfcguard.deployment.export import export_model
from nfcguard.utils import load_config


def test_checkpoint_inference_export(tmp_path):
    cfg=load_config(Path(__file__).parents[1]/'configs'/'quickstart.yaml')
    cfg['model']['widths']=[8,12,16]
    cfg['model']['kernels']=[7,5,3]
    cfg['model']['strides']=[1,2,2]
    model=IQSecureNet(widths=[8,12,16],kernels=[7,5,3],strides=[1,2,2],hidden=32)
    path=tmp_path/'checkpoint.pt'
    save_checkpoint(path,model,cfg,'iqsecurenet',2,1800,None,1.25,3)
    pred=NFCPredictor(path)
    result=pred.predict(np.random.default_rng(1).normal(size=(1800,2)).astype('float32'))
    assert 0<=result['relay_probability']<=1
    assert result['label'] in (0,1)
    artifact=export_model(path,tmp_path/'model.pt','torchscript')
    assert Path(artifact['file']).exists()
    assert artifact['max_logit_abs_difference']<1e-5
