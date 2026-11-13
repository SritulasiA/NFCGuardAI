"""NFCGuardAI command-line interface. `python -m nfcguard.cli --help`."""
from __future__ import annotations
import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from .utils import load_config, save_json

LOG = logging.getLogger(__name__)


def _shared(parser, include_store=True, include_config=True, include_out=True):
    if include_config:
        parser.add_argument('--config', default='configs/reproduction.yaml', help='YAML configuration')
    if include_store:
        parser.add_argument('--store', default=None, help='Prepared data store (default from YAML)')
    if include_out:
        parser.add_argument('--out', default='outputs/experiment', help='Output directory')
    parser.add_argument('--seed', type=int, default=None)


def _cfg(args):
    cfg = load_config(args.config)
    if getattr(args, 'seed', None) is not None:
        cfg['seed'] = args.seed
    return cfg


def _store(args, cfg):
    from .dataset.store import SignalStore
    return SignalStore(args.store or cfg['dataset']['store'])


def _split(args, store, cfg):
    from .dataset.splits import heldout, load_splits
    if getattr(args, 'splits', None):
        return load_splits(args.splits)
    return heldout(store.subtypes(), cfg.get('seed',42))


def _emit(value):
    print(json.dumps(value, indent=2, default=str))


def generate_synthetic(directory, length=1800, samples_per_class=36, seed=123):
    """Only for testing software plumbing; not actual NFC data or reported paper evidence."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(length)/1e7
    for cls in ('normal_tag1', 'wired_tag1', 'wireless'):
        waves = []
        for i in range(samples_per_class):
            phase = rng.uniform(-.2,.2)
            carrier = 2 * np.pi * 847500 * t + phase
            envelope = np.zeros(length)
            for center in (160,650,1160,1600):
                envelope += np.exp(-.5*((np.arange(length)-center)/85)**4)
            if cls == 'wired_tag1':
                envelope = np.convolve(envelope, np.array([.08,.2,.44,.2,.08]), mode='same')
                carrier += .11*np.sin(2*np.pi*26000*t)
            elif cls == 'wireless':
                envelope = np.roll(envelope, int(rng.integers(-8,9)))
                carrier += .17*np.sin(2*np.pi*48000*t)
            amplitude = (1. + rng.normal(0,.03)) * envelope
            signal = np.column_stack([amplitude*np.cos(carrier),amplitude*np.sin(carrier)])
            signal += rng.normal(0,.035,signal.shape)
            waves.append(signal.astype(np.float32))
        np.save(root/f'{cls}.npy',np.stack(waves))
    return root


def cmd_download(args):
    from .dataset.download import download_dataset
    result=download_dataset(args.destination, overwrite=args.overwrite)
    _emit({'downloaded':str(result),'size_bytes':result.stat().st_size})


def cmd_inspect(args):
    from .dataset.readers import source_report
    report=source_report(args.source,args.length,args.label_map)
    _emit(report)


def cmd_prepare(args):
    from .dataset.store import prepare_store
    cfg=_cfg(args)
    d=cfg['dataset']
    stats=prepare_store(args.source,args.target or d['store'], length=d['length'],
                        channels=d.get('channels','auto'),label_map=args.label_map or d.get('label_map'),
                        strict_lengths=d.get('strict_lengths',True),cache_dtype=d.get('cache_dtype','float16'),
                        force=args.force)
    _emit(stats)


def cmd_demo(args):
    from .dataset.store import prepare_store,SignalStore
    from .dataset.splits import heldout
    from .training.trainer import train_experiment
    cfg=load_config(args.config)
    import torch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    root=generate_synthetic(Path(args.out)/'synthetic_raw',length=cfg['dataset']['length'],
                            samples_per_class=args.samples_per_class,seed=args.seed or 42)
    target=Path(args.out)/'synthetic_processed'
    prepare_store(root,target,length=cfg['dataset']['length'],channels=2,force=True)
    store=SignalStore(target)
    split=heldout(store.subtypes(),args.seed or 42)
    result=train_experiment(store,split,cfg,Path(args.out)/'training',seed=args.seed or 42,
                            generate_plots=not args.skip_plots)
    result['IMPORTANT']='Synthetic waveforms are a software smoke test, NOT NFC_Relay experiment results.'
    _emit(result)


def cmd_train(args):
    from .training.trainer import train_experiment
    from .dataset.splits import save_splits
    cfg=_cfg(args)
    store=_store(args,cfg)
    split=_split(args,store,cfg)
    output=train_experiment(store,split,cfg,args.out,model_name=args.model,generate_plots=not args.skip_plots)
    _emit(output)


def cmd_compare(args):
    from .training.trainer import train_experiment
    from .evaluation.classical import run_classical
    from .models.factory import NEURAL_MODELS,CLASSICAL_MODELS
    from .evaluation.reports import compare_heldout_predictions
    cfg=_cfg(args)
    store=_store(args,cfg)
    split=_split(args,store,cfg)
    names = (list(NEURAL_MODELS)+list(CLASSICAL_MODELS)) if args.models=='all' else [x.strip() for x in args.models.split(',')]
    result=[]
    for name in names:
        subdir=Path(args.out)/name
        if name in CLASSICAL_MODELS:
            r=run_classical(store,split,cfg,name,subdir,cfg['seed'],generate_plots=not args.skip_plots)
        else:
            r=train_experiment(store,split,cfg,subdir,model_name=name,generate_plots=not args.skip_plots)
        result.append({'model':name, **{k:r['test'].get(k) for k in ('accuracy','f1','roc_auc','pr_auc')}})
    pd.DataFrame(result).to_csv(Path(args.out)/'model_comparison.csv',index=False)
    if 'iqsecurenet' in names and len(names)>1:
        compare_heldout_predictions(Path(args.out)/'iqsecurenet'/'predictions.csv',
                  [Path(args.out)/n/'predictions.csv' for n in names if n!='iqsecurenet'],
                  Path(args.out)/'mcnemar_tests.json')
    _emit(result)


def cmd_cv(args):
    from .training.trainer import train_experiment
    from .evaluation.classical import run_classical
    from .dataset.splits import cv_splits
    from .evaluation.metrics import summarize_folds
    from .models.factory import CLASSICAL_MODELS
    cfg=_cfg(args)
    store=_store(args,cfg)
    nfolds=int(args.folds or cfg['experiments']['cv_folds'])
    seeds=cfg['experiments'].get('cv_seeds',list(range(nfolds)))
    rows=[]
    for fold,split in cv_splits(store.subtypes(),nfolds,cfg['seed']):
        seed=int(seeds[fold] if fold<len(seeds) else cfg['seed']+fold)
        for name in [n.strip() for n in args.models.split(',')]:
            dest=Path(args.out)/name/f'fold_{fold:02d}'
            if name in CLASSICAL_MODELS:
                result=run_classical(store,split,cfg,name,dest,seed,generate_plots=args.plot_folds)
            else:
                result=train_experiment(store,split,cfg,dest,name,seed,generate_plots=args.plot_folds)
            rows.append(dict(model=name,fold=fold,seed=seed,
                             **{key:result['test'].get(key) for key in ('accuracy','precision','recall','specificity','f1','roc_auc','pr_auc')}))
            Path(args.out).mkdir(parents=True,exist_ok=True)
            pd.DataFrame(rows).to_csv(Path(args.out)/'cv_folds.csv',index=False)
    summary={name:summarize_folds([r for r in rows if r['model']==name]) for name in set(r['model'] for r in rows)}
    save_json(Path(args.out)/'cv_summary.json',summary)
    if 'iqsecurenet' in summary and len(summary)>1:
        from .evaluation.reports import compare_cv_metric
        compare_cv_metric([Path(args.out)/'cv_folds.csv'],out_file=Path(args.out)/'cv_corrected_tests.json')
    _emit(summary)


def cmd_generalize(args):
    from .dataset.splits import leave_one_tag_out,group_splits,validate_splits
    from .training.trainer import train_experiment
    cfg=_cfg(args)
    store=_store(args,cfg)
    outcomes=[]
    if args.protocol=='loto':
        protocols=list(leave_one_tag_out(store.meta,cfg['seed']))
    elif args.protocol=='session':
        groups=store.group_ids('session_id')
        protocols=[('cross_session',group_splits(store.subtypes(),groups,cfg['seed']))]
    else:
        raise ValueError('Protocol must be loto or session')
    for name,split in protocols:
        validate_splits(split,len(store.X),groups=(store.group_ids('session_id') if args.protocol=='session' else None))
        try:
            metrics=train_experiment(store,split,cfg,Path(args.out)/str(name),generate_plots=not args.skip_plots)
            outcomes.append({'group':str(name),'test':metrics['test']})
        except ValueError as error:
            outcomes.append({'group':str(name),'error':str(error)})
    save_json(Path(args.out)/'generalization.json',outcomes)
    _emit(outcomes)


def cmd_robustness(args):
    from .evaluation.robustness import run_robustness
    cfg=_cfg(args)
    store=_store(args,cfg)
    split=_split(args,store,cfg)
    results=run_robustness(store,split,args.checkpoint,args.out,max_samples=args.max_samples,
                           seed=cfg['seed'])
    _emit({'conditions':len(results),'output':str(args.out)})


def cmd_ablate(args):
    from .training.trainer import train_experiment
    cfg=_cfg(args)
    store=_store(args,cfg)
    split=_split(args,store,cfg)
    variants={
        'primary':{},
        'joint_complex':{'preprocessing.mode':'joint_complex'},
        'global':{'preprocessing.mode':'global'},
        'none':{'preprocessing.mode':'none'},
        'no_aug':{'preprocessing.augment_wireless':False},
        'no_loss_weights':{'training.class_weighted':False},
        'four_blocks':{'model.widths':[32,64,128,256],'model.kernels':[7,5,5,5],'model.strides':[1,2,2,2]},
        'five_blocks':{'model.widths':[32,64,128,256,304],'model.kernels':[7,5,5,5,3],'model.strides':[1,2,2,2,1]},
        'mean_max':{'model.pooling':'mean_max'},
        'with_se':{'model.squeeze_excitation':True},
    }
    from .utils import set_nested
    names = list(variants) if args.variants=='all' else [a.strip() for a in args.variants.split(',')]
    rows=[]
    for name in names:
        if name not in variants:
            raise ValueError(f'Unknown ablation {name}. Valid: {list(variants)}')
        modified=copy.deepcopy(cfg)
        for key,value in variants[name].items():
            modified=set_nested(modified,key,value)
        result=train_experiment(store,split,modified,Path(args.out)/name,
                                generate_plots=not args.skip_plots)
        rows.append({'variant':name,'accuracy':result['test']['accuracy'],'f1':result['test']['f1'],
                     'params':result['complexity']['parameters']})
        Path(args.out).mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(Path(args.out)/'ablation.csv',index=False)
    _emit(rows)


def cmd_explain(args):
    from .explainability.gradcam import run_explanation
    cfg=_cfg(args)
    store=_store(args,cfg)
    result=run_explanation(store,args.sample_id,args.checkpoint,args.out,seed=cfg['seed'],
                           dpi=cfg['experiments'].get('figure_dpi',600))
    _emit(result)


def cmd_figures(args):
    from .visualization.plots import eda_figures
    cfg=_cfg(args)
    store=_store(args,cfg)
    eda_figures(store,args.out,cfg['experiments'].get('figure_dpi',600))
    _emit({'figures':str(args.out)})


def cmd_benchmark(args):
    from .deployment.benchmark import benchmark
    result=benchmark(args.checkpoint,args.out,args.warmup,args.runs,args.threads)
    _emit(result)


def cmd_export(args):
    from .deployment.export import export_model
    _emit(export_model(args.checkpoint,args.output_file,args.format))


def cmd_predict(args):
    from .deployment.predictor import NFCPredictor
    pred=NFCPredictor(args.checkpoint,args.device)
    if Path(args.input).is_dir():
        results=list(pred.predict_directory(args.input,args.log))
    else:
        results=pred.predict_file(args.input,args.log)
    if args.output:
        save_json(args.output,results)
    _emit({'predictions':len(results),'results':results[:args.print_first],
           'log':args.log,'saved':args.output})


def cmd_stats(args):
    from .evaluation.reports import compare_cv_metric,compare_heldout_predictions
    if args.cv:
        report=compare_cv_metric(args.cv,proposed_name=args.proposed,metric=args.metric,out_file=args.output)
    else:
        if not args.reference or not args.comparators:
            raise ValueError('--reference and --comparators required for held-out McNemar tests')
        report=compare_heldout_predictions(args.reference,args.comparators,args.output)
    _emit(report)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='nfcguard',description='NFCGuardAI reproducible RF relay detection research')
    parser.add_argument('--verbose',action='store_true')
    sub=parser.add_subparsers(dest='cmd',required=True)
    dl=sub.add_parser('download',help='Download official Git LFS dataset archive')
    dl.add_argument('--destination',default='data/NFC_Relay.zip'); dl.add_argument('--overwrite',action='store_true')
    dl.set_defaults(func=cmd_download)
    ins=sub.add_parser('inspect',help='Inspect archive names or inspect extracted waveform formats')
    ins.add_argument('--source',required=True); ins.add_argument('--length',type=int,default=1800)
    ins.add_argument('--label-map'); ins.set_defaults(func=cmd_inspect)
    prep=sub.add_parser('prepare',help='Prepare verified memory-mapped signals and subtype metadata')
    _shared(prep,False,True,False)
    prep.add_argument('--source',required=True); prep.add_argument('--target'); prep.add_argument('--label-map')
    prep.add_argument('--force',action='store_true'); prep.set_defaults(func=cmd_prepare)
    demo=sub.add_parser('demo',help='End-to-end synthetic smoke test (not research evidence)')
    _shared(demo,False,True,True)
    demo.set_defaults(config='configs/quickstart.yaml',out='outputs/demo')
    demo.add_argument('--samples-per-class',type=int,default=36); demo.add_argument('--skip-plots',action='store_true')
    demo.set_defaults(func=cmd_demo)
    train=sub.add_parser('train',help='Train IQSecureNet or a neural baseline on holdout')
    _shared(train)
    train.add_argument('--model',default='iqsecurenet'); train.add_argument('--splits')
    train.add_argument('--skip-plots',action='store_true'); train.set_defaults(func=cmd_train)
    compare=sub.add_parser('compare',help='All seven baselines + proposed model on same split')
    _shared(compare)
    compare.add_argument('--models',default='all'); compare.add_argument('--splits'); compare.add_argument('--skip-plots',action='store_true')
    compare.set_defaults(func=cmd_compare)
    cv=sub.add_parser('cross-validate',help='Fold-by-fold 10-fold cross-validation')
    _shared(cv)
    cv.add_argument('--folds',type=int); cv.add_argument('--models',default='iqsecurenet'); cv.add_argument('--plot-folds',action='store_true')
    cv.set_defaults(func=cmd_cv)
    gen=sub.add_parser('generalize',help='Tag-disjoint or session-disjoint evaluation using observed groups')
    _shared(gen)
    gen.add_argument('--protocol',choices=['loto','session'],required=True); gen.add_argument('--skip-plots',action='store_true')
    gen.set_defaults(func=cmd_generalize)
    rob=sub.add_parser('robustness',help='Test AWGN, fading, CFO, IQ imbalance and interference')
    _shared(rob)
    rob.add_argument('--checkpoint',required=True); rob.add_argument('--splits')
    rob.add_argument('--max-samples',type=int,default=0,help='0 means entire test partition')
    rob.set_defaults(func=cmd_robustness)
    ab=sub.add_parser('ablate',help='Normalization/architecture/loss ablation suite')
    _shared(ab)
    ab.add_argument('--variants',default='all'); ab.add_argument('--skip-plots',action='store_true')
    ab.set_defaults(func=cmd_ablate)
    explain=sub.add_parser('explain',help='1D Grad-CAM and deletion/insertion fidelity')
    _shared(explain)
    explain.add_argument('--checkpoint',required=True); explain.add_argument('--sample-id',type=int,required=True)
    explain.set_defaults(func=cmd_explain)
    figs=sub.add_parser('figures',help='Dataset charts from actual prepared waveforms')
    _shared(figs); figs.set_defaults(func=cmd_figures)
    bench=sub.add_parser('benchmark',help='Batch 1 CPU latency with warmup')
    bench.add_argument('--checkpoint',required=True); bench.add_argument('--out',default='outputs/benchmark')
    bench.add_argument('--warmup',type=int,default=50); bench.add_argument('--runs',type=int,default=200)
    bench.add_argument('--threads',type=int,default=1); bench.set_defaults(func=cmd_benchmark)
    export=sub.add_parser('export',help='TorchScript or ONNX export')
    export.add_argument('--checkpoint',required=True); export.add_argument('--format',choices=['torchscript','onnx'],default='torchscript')
    export.add_argument('--output-file',required=True); export.set_defaults(func=cmd_export)
    pred=sub.add_parser('infer',help='Predict a waveform file or directory; optionally append JSONL audit records')
    pred.add_argument('--checkpoint',required=True); pred.add_argument('--input',required=True)
    pred.add_argument('--device',default='cpu'); pred.add_argument('--output'); pred.add_argument('--log')
    pred.add_argument('--print-first',type=int,default=5); pred.set_defaults(func=cmd_predict)
    stats=sub.add_parser('statistics',help='Paired held-out McNemar or corrected fold tests')
    stats.add_argument('--cv',nargs='+'); stats.add_argument('--reference'); stats.add_argument('--comparators',nargs='+')
    stats.add_argument('--proposed',default='iqsecurenet'); stats.add_argument('--metric',default='accuracy')
    stats.add_argument('--output',required=True); stats.set_defaults(func=cmd_stats)
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,format='%(levelname)s %(message)s')
    return args.func(args)


if __name__=='__main__':
    main()
