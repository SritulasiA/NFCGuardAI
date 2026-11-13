"""Run all manuscript-aligned experiments on a prepared dataset.

This is computationally expensive (10 folds x many models plus ablations).
Invoke from the repository root AFTER pip install -e . and preparing data.
"""
from __future__ import annotations
from nfcguard.cli import main

def main_all():
    main(['figures','--out','outputs/paper/eda'])
    main(['train','--out','outputs/paper/holdout'])
    main(['compare','--out','outputs/paper/baselines'])
    main(['cross-validate','--models','iqsecurenet,cnn,resnet18,bilstm,cnn_se,original_cnn,svm,random_forest','--out','outputs/paper/cross_validation'])
    main(['ablate','--out','outputs/paper/ablation'])
    main(['generalize','--protocol','loto','--out','outputs/paper/tag_disjoint'])
    # Cross-session is skipped unless the public records contain verified session IDs.
    ckpt='outputs/paper/holdout/best.pt'
    main(['robustness','--checkpoint',ckpt,'--out','outputs/paper/robustness'])
    main(['benchmark','--checkpoint',ckpt,'--out','outputs/paper/benchmark'])
    main(['explain','--checkpoint',ckpt,'--sample-id','0','--out','outputs/paper/gradcam'])
    main(['export','--checkpoint',ckpt,'--output-file','outputs/paper/IQSecureNet.pt'])

if __name__=='__main__':
    main_all()
