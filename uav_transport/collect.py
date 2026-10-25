"""Collect goal-free, matched local recovery outcomes from physical histories."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np

from .plant import TransportConfig, TransportEnv
from .control import TransportController, RecoveryConfig, smooth_transition
from .evaluation import fork, run_option
from .local_model import features, FEATURE_NAMES, TARGET_NAMES
from .snapshot import capture, physical_state


MAX_OPTION_SECONDS = 18.
REARRANGE = (1., 2., 3.)
RETENSION = (1.5, 2., 3.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    return value


def candidate_bank(depth, *, expanded=True, holdout=False):
    rearrange,retension=((1.5,2.5),(1.75,2.5)) if holdout else (REARRANGE,RETENSION)
    return [(bool(mode),RecoveryConfig(rearrange_seconds=a,retension_seconds=b,slack_distance=depth))
        for mode,a,b in itertools.product((False,True) if expanded else (False,),rearrange,retension)]


def contexts(seed=20261025, counts=(24,8,12)):
    rng=np.random.default_rng(seed);out=[]
    for split,count,offset in zip(("train","val","test"),counts,(0,100,200)):
        for index in range(count):
            cfg=TransportConfig(payload_mass=float(rng.uniform(.024,.036)),
                formation_radius=float(rng.uniform(.22,.27)),formation_yaw=float(rng.uniform(-np.pi,np.pi)),
                payload_height=float(rng.uniform(1.05,1.35)),initial_velocity=tuple(rng.uniform(-.03,.03,3)))
            out.append(dict(context_id=offset+index,split=split,plant_config=asdict(cfg),donor=index%3,
                slack_distance=float(rng.uniform(.05,.10)),
                prelude_displacement=(rng.uniform(-1,1,3)*np.array([.15,.15,.04])).tolist(),
                prelude_force=(rng.uniform(-1,1,3)*.03).tolist()))
    return out


def prepare_context(spec):
    env=TransportEnv(TransportConfig(**spec["plant_config"]));controller=TransportController(env.config)
    obs=env.observe();origin=obs["payload_pos"].copy();states=[physical_state(env)];actions=[];forces=[]
    steps=round(2./env.config.control_dt)
    for step in range(steps):
        reference,velocity=smooth_transition(origin,origin+spec["prelude_displacement"],
            step*env.config.control_dt,.9)
        external=np.zeros((4,3))
        if steps-10 <= step < steps-5:
            external[1+spec["donor"]]=spec["prelude_force"]
        command=controller.command(obs,reference,payload_velocity=velocity)
        obs=env.step(command,external)
        states.append(physical_state(env));actions.append(command);forces.append(external)
    return env,controller,dict(states=np.asarray(states),actions=np.asarray(actions),external_forces=np.asarray(forces))


def collect_context(spec,output):
    env,controller,prelude=prepare_context(spec)
    start=capture(env);memory=controller.capture_memory();obs=env.observe();X=[];Y=[];rows=[]
    banks=[(spec["split"],candidate_bank(spec["slack_distance"]))]
    if spec["split"]=="test":banks.append(("test_action",candidate_bank(spec["slack_distance"],holdout=True)))
    for split,bank in banks:
        for mode,config in bank:
            X.append(features(obs,env.config,spec["donor"],config,mode,memory))
            other,control=fork(env,controller)
            result=run_option(other,control,config,spec["donor"],mode,
                max_seconds=MAX_OPTION_SECONDS,record=False)
            Y.append(result["label"])
            rows.append(dict(context_id=spec["context_id"],split=split,reconfigure=mode,
                recovery_config=asdict(config),summary=result["summary"]))
    if not np.array_equal(start.state,capture(env).state):raise AssertionError("Counterfactual parent state changed")
    path=Path(f"{output}_context{spec['context_id']:03d}.pkl")
    if path.exists():raise FileExistsError(path)
    with path.open("wb") as stream:
        pickle.dump(dict(spec=spec,initial_snapshot=start,controller_memory=memory,prelude=prelude,
            X=np.asarray(X),Y=np.asarray(Y),rows=rows),stream,protocol=5)
    return dict(path=str(path),sha256=sha(path),context_id=spec["context_id"],examples=len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("data/uav_transport/local_v1"))
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--counts",type=int,nargs=3,default=[24,8,12])
    parser.add_argument("--seed",type=int,default=20261025)
    args=parser.parse_args()
    if not 1<=args.workers<=8 or min(args.counts)<1:raise ValueError("Use1..8workers andpositive split counts")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.with_suffix(".npz").exists() or args.out.with_suffix(".json").exists():raise FileExistsError(args.out)
    specs=contexts(args.seed,tuple(args.counts))
    paths=[Path(__file__)]+[Path("uav_transport")/(name+".py") for name in ("plant","snapshot","control","evaluation","local_model")]
    hashes={str(p):sha(p) for p in paths}
    registration=dict(contexts=specs,source_sha256=hashes,feature_names=FEATURE_NAMES,target_names=TARGET_NAMES,
        max_option_seconds=MAX_OPTION_SECONDS,seed=args.seed,controller_gains="fixed defaults; calibrated masses supplied to every controller and predictor",
        supervision="Complete local recovery options after independent physical preludes; no composed mission trajectories or mission goals",
        candidates=dict(rearrange_seconds=REARRANGE,retension_seconds=RETENSION,reconfigure=[False,True]),
        split_unit="Physical context, shared by all counterfactual candidates; test_action shares only held-out test contexts")
    reg=args.out.with_name(args.out.name+"_registration.json")
    if reg.exists():raise FileExistsError(reg)
    reg.write_text(json.dumps(plain(registration),indent=2)+"\n")
    started=time.perf_counter();parts=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(collect_context,spec,args.out) for spec in specs]):
            part=future.result();parts.append(part)
            print(json.dumps(dict(completed=len(parts),total=len(specs),**part)),flush=True)
    arrays={};rows=[]
    for part in sorted(parts,key=lambda x:x["context_id"]):
        with Path(part["path"]).open("rb") as stream:data=pickle.load(stream)
        for index,row in enumerate(data["rows"]):
            split=row["split"]
            for key,value in (("X",data["X"][index]),("Y",data["Y"][index]),("groups",row["context_id"])):
                arrays.setdefault(key+"_"+split,[]).append(value)
            rows.append(row)
    arrays={key:np.asarray(value) for key,value in arrays.items()}
    np.savez_compressed(args.out.with_suffix(".npz"),**arrays)
    for path,digest in hashes.items():
        if sha(path)!=digest:raise AssertionError("Source changed during collection: "+path)
    report=dict(registration,registration_path=str(reg),registration_sha256=sha(reg),parts=sorted(parts,key=lambda x:x["context_id"]),
        array_path=str(args.out.with_suffix(".npz")),array_sha256=sha(args.out.with_suffix(".npz")),rows=rows,
        examples_by_split={split:len(arrays['X_'+split]) for split in ('train','val','test','test_action')},
        completed=sum(bool(r['summary']['completion']) for r in rows),examples=len(rows),wall_seconds=time.perf_counter()-started)
    args.out.with_suffix(".json").write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.out),counts=report["examples_by_split"],completed=report["completed"],seconds=report["wall_seconds"])),flush=True)


if __name__=="__main__":main()
