"""Publication-oriented, standalone figures; no invented numerical measurements."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


def evaluation_figures(y, p, root, dpi=600):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    pred = (p >= .5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5,4.5), layout='constrained')
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1], ['Legitimate','Relay'])
    ax.set_yticks([0,1], ['Legitimate','Relay'])
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Actual class')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j] > cm.max() * .55 else 'black')
    fig.savefig(root / 'confusion_matrix.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
        ax.plot(fpr,tpr,label='Detector')
        ax.plot([0,1],[0,1],ls='--',label='Chance',alpha=.6)
        ax.set(xlabel='False positive rate',ylabel='True positive rate')
        ax.legend()
        fig.savefig(root / 'roc_curve.png',dpi=dpi,bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
        ax.plot(recall,precision,label='Detector')
        ax.set(xlabel='Recall',ylabel='Precision')
        ax.legend()
        fig.savefig(root / 'precision_recall_curve.png',dpi=dpi,bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5,4),layout='constrained')
    ax.hist(p[y==0],bins=25,alpha=.6,label='Legitimate')
    ax.hist(p[y==1],bins=25,alpha=.6,label='Relay')
    ax.axvline(.5, ls='--',c='gray',label='Operating threshold')
    ax.set(xlabel='Calibrated relay probability',ylabel='Number of traces')
    ax.legend()
    fig.savefig(root / 'score_distributions.png',dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def training_figure(history, root, dpi=600):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,4),layout='constrained')
    ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Training')
    ax.plot([r['epoch'] for r in history],[r['ckpt_loss'] for r in history],label='Checkpoint validation')
    ax.set(xlabel='Epoch',ylabel='Loss')
    ax.legend()
    fig.savefig(root/'training_curves.png',dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def robustness_figures(df, root, dpi=600):
    root=Path(root)
    for name,group in df.groupby('impairment'):
        if name=='clean':
            continue
        fig, ax = plt.subplots(figsize=(5,3.5),layout='constrained')
        group=group.sort_values('level')
        ax.plot(group['level'], group['f1'], marker='o',label='F1')
        ax.plot(group['level'], group['accuracy'],marker='s',label='Accuracy')
        ax.set(xlabel=f'{name} severity (see documentation for units)',ylabel='Score',ylim=(0,1.03))
        ax.legend()
        fig.savefig(root/f'robustness_{name}.png',dpi=dpi,bbox_inches='tight')
        plt.close(fig)


def eda_figures(store, output_dir, dpi=600):
    root=Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    counts=store.meta['subtype'].value_counts().reindex(['legitimate','wired','wireless'],fill_value=0)
    fig, ax = plt.subplots(figsize=(6,4),layout='constrained')
    ax.bar(counts.index, counts.values)
    ax.set(xlabel='Signal subtype',ylabel='Number of traces')
    fig.savefig(root/'dataset_distribution.png',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
    for sid,name in enumerate(['legitimate','wired','wireless']):
        positions=np.where(store.subtypes()==sid)[0]
        if not len(positions):
            continue
        wave=np.asarray(store.X[positions[0]],dtype=float)
        t=np.arange(store.length)/1e7*1e6
        fig, ax = plt.subplots(figsize=(8,3),layout='constrained')
        ax.plot(t,wave[:,0],label='Channel 1',lw=.6)
        if store.channels == 2:
            ax.plot(t,wave[:,1],label='Channel 2',lw=.6,alpha=.75)
        ax.set(xlabel='Time (µs)',ylabel='Recorded amplitude')
        ax.legend()
        fig.savefig(root/f'{name}_trace.png',dpi=dpi,bbox_inches='tight')
        plt.close(fig)
        if store.channels == 2:
            fig, ax=plt.subplots(figsize=(4.5,4.5),layout='constrained')
            ax.scatter(wave[:,0],wave[:,1],s=1,alpha=.3)
            ax.set(xlabel='Channel 1 (I if measured)',ylabel='Channel 2 (Q if measured)')
            fig.savefig(root/f'{name}_iq_scatter.png',dpi=dpi,bbox_inches='tight')
            plt.close(fig)
