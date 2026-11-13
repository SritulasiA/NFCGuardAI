# Local Software Validation Report

The following tests were conducted on synthetic waveforms solely for integration
and unit-test verification. **No experiment has been run on the actual NFC_Relay
archive in this distribution.** The manuscript's published performance is not
reproduced or claimed by the software smoke test.

- Verified exact IQSecureNet network shape, 852,081 parameters, 196,695,680 MACs,
  and 67-sample receptive field for `[1,2,1800]` input.
- Executed synthetic signal generation, dataset preparation and manifest,
  subtype stratification, 2-epoch lightweight training, best-checkpoint load,
  temperature scaling and held-out test prediction.
- Executed synthetic-checkpoint CPU latency benchmark, TorchScript model
  export, file-based inference, Grad-CAM with deletion/insertion, and the
  controllable channel-impairment suite.
- Run `pytest -q` to reproduce the up-to-date test outcomes on your machine.

A fully trained research checkpoint and real NFC signal performance outputs
require the public original dataset, sufficient compute and confirmed signal
provenance.

## Validation results in the build environment

- Unit/integration tests: **16 passed** (`pytest -q`).
- `compileall`: passed.
- `nfcguard demo`: completed preprocessing, small-model optimization,
  calibration, evaluation and checkpoint writing.
- `nfcguard compare`: SVM, Random Forest, three-layer CNN, original-CNN
  approximation executed on synthetic examples.
- `nfcguard cross-validate`: 3-fold IQSecureNet/SVM functionality tested
  on synthetic examples, including corrected paired statistics.
- `nfcguard ablate`: `no_aug` and `mean_max` variants executed.
- `nfcguard explain`, `robustness`, `benchmark`, `infer`, `export`: executed
  successfully with a *synthetic* checkpoint.
- Tested PyTorch version: **2.10.0+cpu**; Python version: **3.13** in the
  build environment. The dependency range also permits PyTorch >=2.0 and
  Python >=3.10; no all-platform guarantee is implied by local tests.

These checks establish operational code paths but do not validate accuracy,
calibration quality, model robustness or true reader-side latency on NFC_Relay.
