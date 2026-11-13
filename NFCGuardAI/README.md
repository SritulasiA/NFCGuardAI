# NFCGuardAI — IQSecureNet

**Reproducible physical-layer NFC relay attack detection from ATQA response waveforms.**

This repository implements the architecture and experiment workflow described in
*Deep Learning-Based Real-Time Physical-Layer Relay Attack Detection in Near-Field
Communication Using Raw I/Q Signal Analysis*. It includes a six-block Conv1D
IQSecureNet classifier, data loaders, leakage-aware partitions, checkpoint selection,
post-hoc calibration, seven baselines, robustness tests, statistical tests, Grad-CAM,
publication-quality plots, and offline/batch inference with model export.

> **Important scientific boundary**: The ZIP contains source code, not the external
> NFC_Relay dataset, experimental measurements, or pretrained model weights. The
> provided synthetic smoke test verifies *software function*, not research claims.
> The original publication describes WAV recordings; WAV files may contain only
> mono amplitude samples rather than verified complex I/Q. When the actual archive
> contains only one channel, this software **trains a clearly labeled mono-mode
> model**; it does not invent a quadrature channel or claim exact reproduction of
> a two-channel model without measured I/Q. Claims of 94.6% accuracy, 95.1% F1,
> and 8.3 ms CPU time must be independently verified on the proper source data.

## 1. Requirements

- Python 3.10+; PyTorch 2.0+; Windows/Linux/macOS.
- For full reproduction, sufficient space for the ~200 MB external archive and
  ~0.25–1 GB extracted/memory-mapped signals. Memory depends on source channels.
- CPU-only execution supported. A GPU is strongly recommended for all ten folds,
  seven baseline comparisons and systematic ablation experiments.
- For GPU-enabled PyTorch, install the wheel matching your NVIDIA driver and CUDA
  version from <https://pytorch.org/get-started/locally/> before package installation.

```bash
unzip NFCGuardAI.zip
cd NFCGuardAI
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -e ".[test]"
pytest -q
```

For an immediate smoke test **without downloading any dataset**:

```bash
nfcguard demo --config configs/quickstart.yaml --out outputs/demo
# Or: python -m nfcguard.cli demo
```

The demo constructs *synthetic, artificial* I/Q-like waveforms solely to test code
paths and produces `outputs/demo/training/best.pt`, metrics, and predictions.
**Do not cite synthetic demonstration metrics in a paper.**

## 2. Original dataset and provenance

Dataset: **NFC_Relay**, released with Y. Wang, J. Zou and K. Zhang,
"Deep-Learning-Aided RF Fingerprinting for NFC Relay Attack Detection,"
*Electronics* 12(3), 559 (2023), DOI
<https://doi.org/10.3390/electronics12030559>.
Public repository: <https://github.com/Yifeng-Wang/nfc-relayattack>.

```bash
# Option 1: Try built-in Git LFS media download
nfcguard download --destination data/NFC_Relay.zip

# Option 2: Git LFS, when the media endpoint is unavailable
# git lfs install
# git clone https://github.com/Yifeng-Wang/nfc-relayattack.git data/nfc-source
# git -C data/nfc-source lfs pull

nfcguard inspect --source data/NFC_Relay.zip
nfcguard prepare --source data/NFC_Relay.zip --config configs/reproduction.yaml
# Or for extracted data:
# nfcguard inspect --source data/nfc-source
# nfcguard prepare --source data/nfc-source --config configs/reproduction.yaml
```

`prepare` safely extracts ZIP files, scans WAV/FLAC/NPY/NPZ/CSV/MAT recordings,
verifies frame lengths, normalizes file layout to `[N,1800,C]`, and produces
`data/processed/{signals.npy,metadata.csv,manifest.json}`. WAV decoding uses
`soundfile` and retains the source channel count. Long files whose sample count
is divisible by 1800 are treated as **pre-aligned** consecutive frames. If the
source is a continuous RF recording, external ATQA localization is required
first: simply cutting arbitrary 1800-sample windows is not event detection.

**The manuscript and original publication disagree on dataset counts.** The
published NFC_Relay study reports **66,366** records, while the manuscript
under implementation specifies **67,259**. Neither count is assumed by the
loader; actual verified records are logged in `manifest.json` and dataset
class distributions are calculated from that manifest.

### Class mapping

If the source filename or parent folder contains `normal`, `legitimate`,
`wired`, `wireless` or `wifi`, the program can infer its three-way subtype.
Otherwise inspect the extracted filenames and supply a customized
`configs/label_map.example.yaml`:

```bash
nfcguard prepare --source data/NFC_Relay.zip \
  --label-map configs/label_map.example.yaml
```

Do **not** assign labels by guessed numeric ordering. Actual tag/session
metadata must be taken from the files or verified collection mapping before
claiming tag/session-independent generalization. `tag_id` and `session_id`
remain empty if not recoverable. `generalize --protocol session` will raise
an informative error if sessions cannot be established.

### Channel handling

| Actual channels | Behavior |
|---|---|
| Two verified I/Q channels | IQSecureNet uses input `[B,2,1800]`, exact parameter count **852,081**. |
| One measured mono channel | Honest `[B,1,1800]` compatibility mode, *not* the same model as two-channel paper architecture. |
| Mixed one- and two-channel data | Fails, requiring source reconciliation; no silent channel duplication. |

A two-channel WAV header alone does not prove that the two channels are physically
measured I and Q: check recording provenance. For independently acquired genuine
complex samples, `.npy` with a complex vector or a two-channel matrix is supported.

## 3. Exact proposed architecture

| Stage | Operation | Output, `[C,T]` |
|---|---|---|
| Input | raw channels, per-trace Z-score | `[2,1800]` |
| 1 | Conv1D 2→32, kernel 7, stride 1; BN; ReLU | `[32,1800]` |
| 2 | Conv1D 32→64, kernel 5, stride 2; BN; ReLU | `[64,900]` |
| 3 | Conv1D 64→128, kernel 5, stride 2; BN; ReLU | `[128,450]` |
| 4 | Conv1D 128→256, kernel 5, stride 2; BN; ReLU | `[256,225]` |
| 5 | Conv1D 256→304, kernel 3, stride 1; BN; ReLU | `[304,225]` |
| 6 | Conv1D 304→384, kernel 3, stride 1; BN; ReLU | `[384,225]` |
| Pool | Global Average Pooling | `[384]` |
| Head | Dropout .3; Dense 384→128; ReLU | `[128]` |
| Output | Dropout .3; Dense 128→1 | relay logit |

**Automated exact checks**: 852,081 parameters; 196,695,680 MACs;
393,391,360 convolution/dense FLOPs (2 per MAC); receptive field 67 samples.
Model outputs a **logit** for stable weighted BCE; the sigmoid is applied during
calibration/inference. At 32-bit, weight-only storage is approximately 3.41 MB.

## 4. Training and evaluation

```bash
# One held-out 70/7.5/7.5/15 experiment
nfcguard train --config configs/reproduction.yaml --out outputs/holdout

# CV: 70% fit/10% checkpoint/10% calibration/10% outer test
nfcguard cross-validate --folds 10 --out outputs/cv

# All seven comparisons + IQSecureNet on identical partitions
nfcguard compare --models all --out outputs/comparison

# All CV models (large and time-consuming)
nfcguard cross-validate \
  --models iqsecurenet,cnn,resnet18,bilstm,cnn_se,original_cnn,svm,random_forest \
  --out outputs/cv_all

# LOTO test if tag identifiers are observed (normal/wired test traces only)
nfcguard generalize --protocol loto --out outputs/tag_independent

# Verified session IDs required for cross-session evaluation
nfcguard generalize --protocol session --out outputs/session_independent
```

Configuration: `configs/reproduction.yaml` (fixed seed, deterministic torch
settings, 50 max epochs, AdamW lr 0.001, weight decay 0.0001, batch size 64,
StepLR gamma .5 every 10 epochs, checkpoint-loss early stopping patience 10,
class-weighted BCE, training-only wireless augmentation). The trained model
uses the training split only; V_ckpt chooses its epoch; V_cal fits temperature
and analyzes threshold sensitivity; the test set is never used to tune the
model, calibration or fixed threshold 0.50.

Primary random splits are **stratified by legitimate/wired/wireless subtype**.
10-fold partitions are identical across models. Splits are persisted as
indices for exact reuse (`--splits` can load a held-out split). A 10-fold
corrected resampled t-test with Holm adjustment and a held-out McNemar
comparison are available. The variance-correction result is shown both for
the manuscript's 0.1/0.9 ratio and the effective training fraction 0.1/0.7.

### Comparison model transparency

- `cnn`: three-layer 1D CNN with flattening-based dense classifier.
- `resnet18`: standard four-stage, two-block-per-stage 1D ResNet-18.
- `bilstm`: bidirectional two-layer LSTM, preceded by pooling for feasibility.
- `cnn_se`: channel-reweighted CNN with SE blocks.
- `original_cnn`: **architectural approximation** using published 72/64/32
  filters and kernels 8/6/6. The exact original checkpoint is not provided.
- `svm`: scalable **linear SVM**, trained on downsampled signal statistics.
- `random_forest`: balanced RF on the same fixed signal statistics.

These choices are functional, documented adaptations of baselines with
incomplete hyperparameters; they are **not claims of exactly reproducing the
original authors' experiments**. The primary IQSecureNet topology is exact.

## 5. Ablations, robustness, explainability

```bash
nfcguard ablate --variants all --out outputs/ablation
nfcguard robustness --checkpoint outputs/holdout/best.pt --out outputs/robustness
nfcguard explain --checkpoint outputs/holdout/best.pt --sample-id 0 --out outputs/gradcam
nfcguard figures --out outputs/eda
```

Ablations: per-channel / joint-complex / training-global / none, no
augmentation, no class weighting, 4-/5-block architectures, mean/max pooling
and optional squeeze-excitation. Robustness: AWGN (SNR dB), Rician flat fading
(K), carrier frequency offset (Hz), I/Q imbalance (gain mismatch), and CW
interference (interference-to-signal dB). Two-channel-specific perturbations
fail rather than masquerade as valid mono experiments. Test perturbations are
seeded and never used for model selection. Grad-CAM operates on the last
Conv1D block, upsampled to 1,800 points. Fidelity plots show insertion and
deletion by attribution order, with perturbation stability diagnostics.

Figures are exported as 600 dpi PNGs, with axes/legends and no titles inside
figures. **All experimental figures are generated from actual predictions.**

## 6. Inference, benchmarks and export

```bash
# CPU inference, including separate model and software-pipeline timing
nfcguard benchmark --checkpoint outputs/holdout/best.pt \
  --warmup 50 --runs 1000 --threads 1 --out outputs/benchmark

# TorchScript export: always available with supported PyTorch
nfcguard export --checkpoint outputs/holdout/best.pt \
  --format torchscript --output-file outputs/IQSecureNet.pt

# Optional ONNX export
pip install -e ".[onnx]"
nfcguard export --checkpoint outputs/holdout/best.pt \
  --format onnx --output-file outputs/IQSecureNet.onnx

# Single file or batch of pre-extracted waveforms
nfcguard infer --checkpoint outputs/holdout/best.pt \
  --input my_recording.npy --output outputs/infer.json --log outputs/audit.jsonl
nfcguard infer --checkpoint outputs/holdout/best.pt \
  --input path/to/extracted_traces --log outputs/audit.jsonl
```

The prediction result includes signal ID, timestamp, calibrated relay
probability, binary label, threshold, elapsed processing time and a suggested
alert/pass action. **No actual NFC transaction is blocked by the software
alone.** A compatible SDR I/Q capture/segmentation adapter, manufacturer reader
firmware and enforceable host security policy are necessary before deployment.
The benchmark excludes RF acquisition, reader actions and alert delivery;
actual embedded-device timing must be measured on the target device.

## 7. Output interpretation

| Path | Content |
|---|---|
| `outputs/holdout/best.pt` | Frozen model, configuration, normalization stats, temperature |
| `outputs/holdout/history.csv` | Training/validation learning curves |
| `outputs/holdout/metrics.json` | Test scores, calibration and complexity |
| `outputs/holdout/splits.json` | Record indices for each partition |
| `outputs/holdout/predictions.csv` | One calibrated prediction for every held-out test trace |
| `outputs/holdout/figures/` | Confusion matrix, ROC/PR, distributions, learning curves |
| `outputs/cv/cv_folds.csv` | Fold-level metrics; mean ± SD and 95% CI |
| `outputs/cv/cv_corrected_tests.json` | Corrected paired tests and Holm correction |
| `outputs/comparison/mcnemar_tests.json` | Matched prediction-level tests |
| `outputs/robustness/` | Perturbed scores, CSV, PNG |
| `outputs/gradcam/` | Explanations and fidelity analysis |
| `outputs/benchmark/` | CPU timings, platform description, exclusions |

## 8. Testing and consistency checks

```bash
pytest -q
python -m compileall -q src
python -m nfcguard.cli --help
```

The tests cover exact model count/MAC/receptive field, mono-handling, WAV and
array shapes, label mapping, leak-free partitions, stratified sampling,
preprocessing, calibration, paired statistical tests, saved checkpoints,
TorchScript export, and functional model forward passes.

## 9. Study validity notes

1. The public dataset may contain 1-channel WAV waveforms; do not label these
   as measured I/Q without verifying source acquisition.
2. The original publication mentions phases one week apart but does not
   necessarily release per-record session identifiers. Cross-session tests
   require real session metadata, not random pseudo-groups.
3. A wireless relay class may have no observed tag identity; LOTO reports only
   true held-out-tag **legitimate/wired** traces unless a verified wireless
   grouping can be supplied. This is not equivalent to every claim in the
   manuscript's later tables.
4. Per-trace channelwise Z-scoring erases absolute gain; augmenting with a
   uniform amplitude scale before it may have negligible effect.
5. Classical and deep baseline details that are not completely specified in
   the manuscript must be reported as adaptations in any resulting article.
6. Security software requires hardware/reader integration and threat-model
   validation before real-world use. The code is an experimental research
   pipeline, not a certified payment security module.

## 10. Software license and attribution

Source code: MIT (`LICENSE`). The external dataset remains the property of
its original authors and is not included in the distribution. Please cite the
original dataset paper above, as well as the NFCGuardAI manuscript when
appropriate.

### Integration with an external live ATQA capture source

The callable `nfcguard.deployment.streaming.NFCGuardStream` processes an
iterable of `SignalEvent(trace, signal_id)` objects, performs identical model
preprocessing/inference, appends optional JSONL logs, and invokes an optional
host callback. It does not include or assert unverified SDR device drivers:

```python
from nfcguard.deployment.streaming import NFCGuardStream, SignalEvent

stream = NFCGuardStream('outputs/holdout/best.pt',
                        callback=lambda result: print(result['decision']),
                        log_path='outputs/live_alerts.jsonl')

def captured_atqa_events():  # supply real hardware adapter here
    # yield SignalEvent(trace=measured_array_1800_by_2, signal_id='txn-001')
    return
    yield

for prediction in stream.process(captured_atqa_events()):
    print(prediction)
```

The host callback can be connected to a reader's supported control interface
by its system integrator; a mere `action='alert'` suggestion does not enforce
an NFC denial.
