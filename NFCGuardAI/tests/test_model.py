import torch
from nfcguard.models.factory import create_model, complexity
from nfcguard.models.iqsecurenet import IQSecureNet


def test_paper_exact_architecture():
    model = IQSecureNet()
    c = complexity(model, (1, 2, 1800))
    assert c['parameters'] == 852081
    assert c['macs'] == 196695680
    assert c['receptive_field'] == 67
    with torch.inference_mode():
        assert model(torch.ones(1, 2, 1800)).shape == (1,)


def test_mono_and_variable_width():
    model = IQSecureNet(in_channels=1, widths=[8,12,16], kernels=[7,5,3],strides=[1,2,2], hidden=12)
    assert model(torch.zeros(3,1,1800)).shape == (3,)
    assert model.receptive_field() == 15


def test_neural_baselines_shapes():
    torch.set_num_threads(2)
    for name in ('cnn','resnet18','bilstm','cnn_se','original_cnn'):
        net=create_model(name,in_channels=2)
        net.eval()
        with torch.inference_mode():
            assert net(torch.randn(1,2,1800)).shape == (1,), name
