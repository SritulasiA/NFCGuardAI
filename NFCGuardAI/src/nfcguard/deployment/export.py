"""Export TorchScript (available by default) and optional ONNX."""
from __future__ import annotations
from pathlib import Path
import json
import torch
from ..training.trainer import load_checkpoint
from ..utils import save_json


def export_model(checkpoint, output_file, fmt='torchscript'):
    model, cfg = load_checkpoint(checkpoint, 'cpu')
    model.eval()
    x = torch.zeros(1, cfg['channels'], cfg['length'], dtype=torch.float32)
    dest = Path(output_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt in ('torchscript','jit'):
        traced = torch.jit.trace(model, x, strict=True)
        traced.save(str(dest))
        recovered = torch.jit.load(str(dest))
        with torch.inference_mode():
            difference = float(torch.max(torch.abs(model(x) - recovered(x))))
    elif fmt == 'onnx':
        try:
            import onnx
        except ImportError as e:
            raise RuntimeError('ONNX optional dependency missing. Install pip install -e ".[onnx]"') from e
        try:
            torch.onnx.export(model, x, str(dest), opset_version=17, input_names=['waveform'],
                              output_names=['relay_logit'], dynamic_axes={'waveform':{0:'batch'},'relay_logit':{0:'batch'}},
                              dynamo=False)
            onnx.checker.check_model(onnx.load(str(dest)))
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f'ONNX export or validation failed: {e}') from e
        difference = None
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(dest), providers=['CPUExecutionProvider'])
            result = session.run(None, {'waveform':x.numpy()})[0]
            difference=float(abs(result.squeeze()-model(x).detach().numpy().squeeze()))
        except ImportError:
            pass
    else:
        raise ValueError('Format must be torchscript or onnx')
    result=dict(format=fmt, source_checkpoint=str(checkpoint), file=str(dest),
                max_logit_abs_difference=difference, output_is_logit=True,
                temperature=cfg['temperature'], threshold=cfg['config']['calibration'].get('threshold', .5),
                normalization=cfg['config']['preprocessing']['mode'], global_stats=cfg['global_stats'],
                warning='Exported graph accepts preprocessed [B,C,T] float32. Apply metadata normalization, temperature and threshold externally.')
    save_json(dest.with_suffix('.metadata.json'),result)
    return result
