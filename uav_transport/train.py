"""Fit and freeze both local-model sizes before composed mission evaluation.

Run with the existing Torch environment. The resulting NumPy checkpoints and
validation margins are consumed by the lightweight MuJoCo runtime.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from .local_model import FrozenPredictor, fit_model, load_dataset, prediction_metrics, save_model, dataset_family


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(dataset_path,dataset_hash,output,name,width,seed,steps):
    if sha(dataset_path)!=dataset_hash:raise AssertionError("Registered local data changed")
    data=load_dataset(dataset_path)
    pack=fit_model(data['X_train'],data['Y_train'],data['X_val'],data['Y_val'],
        width=width,seed=seed,steps=steps)
    pack['metadata'].update(dataset_path=str(dataset_path),dataset_sha256=dataset_hash,
                            plant_family=dataset_family(data))
    predictor=FrozenPredictor(pack)
    for split in ('test','test_action'):
        if 'X_'+split in data:
            pack['metadata'][split+'_metrics']=prediction_metrics(predictor.predict(data['X_'+split]),data['Y_'+split])
    prefix=Path(f"{output}_{name}_seed{seed}")
    saved=save_model(pack,prefix)
    margins=np.quantile(np.maximum(data['Y_val'][:,:2]-predictor.predict(data['X_val'])[:,:2],0.),.95,axis=0)
    return f"{name}/seed{seed}",dict(prefix=str(prefix),weights_sha256=sha(saved['weights']),
        metadata_sha256=sha(saved['metadata']),parameters=pack['metadata']['parameters'],
        margins=margins.tolist(),margin_method="Validation-only marginal q95 positive underprediction residual: peak excursion and peak tension")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/uav_transport/local_v1.npz'))
    parser.add_argument('--out',type=Path,default=Path('data/uav_transport/models_v1'))
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=6 or args.steps<1 or len(args.seeds)!=len(set(args.seeds)):
        raise ValueError('Use1..6workers, positive training steps and unique seeds')
    plant_family=dataset_family(load_dataset(args.data))
    registration=args.out.with_name(args.out.name+'_registration.json')
    manifest=args.out.with_name(args.out.name+'_manifest.json')
    schedule=[(name,width,seed) for name,width in (('small',32),('large',128)) for seed in args.seeds]
    paths=[registration,manifest]+[Path(f'{args.out}_{name}_seed{seed}{suffix}')
        for name,_,seed in schedule for suffix in ('.npz','.json')]
    if any(p.exists() for p in paths):raise FileExistsError('Use a fresh model prefix to preserve all frozen checkpoints')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sources={str(Path(__file__)):sha(__file__),'uav_transport/local_model.py':sha('uav_transport/local_model.py')}
    data_hash=sha(args.data)
    protocol=dict(plant_family=plant_family,dataset_path=str(args.data),dataset_sha256=data_hash,source_sha256=sources,
        schedule=[dict(name=n,width=w,seed=s,steps=args.steps) for n,w,s in schedule],
        checkpoint_selection='Validation only; all fit seeds retained',
        constraint_margins='Validation-only q95 positive underprediction residual, first two physical targets',
        training_scope='Independent local recovery operations; all complete missions remain outside model fitting')
    registration.write_text(json.dumps(protocol,indent=2)+'\n')
    models={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,args.data,data_hash,args.out,name,width,seed,args.steps)
            for name,width,seed in schedule]
        for future in as_completed(futures):
            key,entry=future.result();models[key]=entry
            print(json.dumps(dict(model=key,**entry)),flush=True)
    for path,digest in {str(args.data):data_hash,**sources}.items():
        if sha(path)!=digest:raise AssertionError('Frozen training input changed: '+path)
    report=dict(plant_family=plant_family,models={key:models[key] for key in sorted(models)},dataset_path=str(args.data),
        dataset_sha256=data_hash,source_sha256=sources,training_registration_path=str(registration),
        training_registration_sha256=sha(registration),model_status='Allmodelsfrozenbeforecomposedmissionevaluation')
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    print(manifest,flush=True)


if __name__=='__main__':main()
