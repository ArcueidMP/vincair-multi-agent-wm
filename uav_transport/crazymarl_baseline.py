"""Run the pinned, official CrazyMARL environment and IPPO implementation.

This adapter keeps upstream dynamics, observations, rewards and PPO updates
intact. It omits one unused ONNX-export import when loading the official PPO
module. Small training runs are explicitly smoke/unconverged checkpoints;
performance comparisons require a converged policy or a verified released one.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np


UPSTREAM_URL="https://github.com/IMRCLab/CrazyMARL.git"
UPSTREAM_COMMIT="fca7b6e05f3b5e816be7c982c91d69eb92925f91"
CORE_FILES=("crazymarl/envs/multi_quad_env.py","crazymarl/builders/quad_env_builder.py",
    "crazymarl/observations/multi_quad_observation.py","crazymarl/rewards/multi_quad_reward.py",
    "crazymarl/utils/multi_quad_sampling.py","crazymarl/utils/multi_quad_utils.py",
    "jaxmarl/environments/mabrax/mabrax_env.py","baselines/IPPO/ippo_ff_mabrax.py",
    "crazymarl/configs/defaults/training.yaml","crazymarl/train/configs/three_quad.yaml","LICENSE")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(x) for x in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def upstream_provenance(checkout):
    checkout=Path(checkout).resolve()
    head=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
    if head!=UPSTREAM_COMMIT:raise ValueError("Expected pinned official CrazyMARL commit")
    tracked_diff=subprocess.check_output(["git","-C",str(checkout),"diff","HEAD","--name-only"],text=True).strip()
    if tracked_diff:raise ValueError("Keep the official tracked source unchanged: "+tracked_diff)
    hashes={}
    for filename in CORE_FILES:
        path=checkout/filename
        content=subprocess.check_output(["git","-C",str(checkout),"show","HEAD:"+filename])
        expected=hashlib.sha256(content).hexdigest()
        if sha(path)!=expected:raise AssertionError("Upstream source differs: "+filename)
        hashes[filename]=expected
    files=subprocess.check_output(["git","-C",str(checkout),"ls-tree","-r","--name-only","HEAD"],text=True).splitlines()
    checkpoints=[x for x in files if Path(x).suffix.lower() in (".tflite",".onnx",".safetensors",".msgpack",".pth",".pt",".ckpt")]
    return dict(url=UPSTREAM_URL,commit=head,checkout=str(checkout),license="Apache-2.0",
        core_source_sha256=hashes,tracked_source_unchanged=True,tracked_checkpoints=checkpoints,
        checkpoint_release_issue="https://github.com/IMRCLab/CrazyMARL/issues/1")


@contextmanager
def upstream_context(checkout):
    checkout=str(Path(checkout).resolve());old_cwd=os.getcwd();old_path=sys.path.copy()
    os.chdir(checkout);sys.path.insert(0,checkout)
    try:yield
    finally:
        os.chdir(old_cwd);sys.path[:]=old_path


def load_upstream_ippo(checkout):
    """Load unchanged algorithm bodies while skipping an unused exporter import."""
    source=Path(checkout)/"baselines/IPPO/ippo_ff_mabrax.py"
    text=source.read_text();tree=ast.parse(text,filename=str(source))
    imports=[node for node in tree.body if isinstance(node,ast.ImportFrom) and node.module=="jax2onnx"]
    if len(imports)!=1:raise ValueError("Upstream optional-export import no longer matches the adapter contract")
    imported={name.asname or name.name for name in imports[0].names}
    if imported!={"to_onnx","onnx_function"}:raise ValueError("Unexpected ONNX-export import")
    tree.body=[node for node in tree.body if node is not imports[0]]
    used={node.id for node in ast.walk(tree) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load)}
    if imported&used:raise ValueError("The optional exporter is used by this upstream algorithm; omission would change behavior")
    name="_official_crazymarl_ippo_"+UPSTREAM_COMMIT[:12]
    module=types.ModuleType(name);module.__file__=str(source);module.__package__="baselines.IPPO"
    sys.modules[name]=module
    exec(compile(tree,str(source),"exec"),module.__dict__)
    bodies={node.name:hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
            for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    return module,dict(source_sha256=sha(source),omitted_unused_import="from jax2onnx import to_onnx, onnx_function",
        omitted_line=imports[0].lineno,algorithm_bodies_sha256=bodies,training_and_network_bodies_unchanged=True)


def official_config(checkout):
    import yaml
    def merge(left,right):
        result=deepcopy(left)
        for key,value in right.items():
            result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else deepcopy(value)
        return result
    root=Path(checkout)
    base=yaml.safe_load((root/"crazymarl/configs/defaults/training.yaml").read_text())["training"]
    three=yaml.safe_load((root/"crazymarl/train/configs/three_quad.yaml").read_text())["training"]
    return merge(base,three)


def _finite_tree(tree,jax):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree)
               if np.issubdtype(np.asarray(x).dtype,np.number))


def run_environment_smoke(checkout,*,steps=32,seed=0):
    import jax
    import jax.numpy as jp
    with upstream_context(checkout):
        from crazymarl.envs.multi_quad_env import MultiQuadEnv
        env=MultiQuadEnv(num_quads=3,obs_noise=0.,act_noise=0.,target_start_ratio=1.)
        reset=jax.jit(env.reset);step=jax.jit(env.step)
        started=time.perf_counter();state=reset(jax.random.PRNGKey(seed));jax.block_until_ready(state)
        reset_seconds=time.perf_counter()-started
        initial=state;states=[np.asarray(state.pipeline_state.qpos)];rewards=[];dones=[]
        velocities=[np.asarray(state.pipeline_state.qvel)];motors=[np.asarray(state.metrics["filtered_rpm_proxy"])]
        started=time.perf_counter()
        for _ in range(steps):
            state=step(state,jp.zeros(12));jax.block_until_ready(state)
            states.append(np.asarray(state.pipeline_state.qpos));rewards.append(float(state.reward));dones.append(bool(state.done))
            velocities.append(np.asarray(state.pipeline_state.qvel));motors.append(np.asarray(state.metrics["filtered_rpm_proxy"]))
        elapsed=time.perf_counter()-started
        # The complete immutable Brax/MJX state includes all motor/randomization
        # metrics; replay the same action from it and compare every numeric leaf.
        one=step(initial,jp.zeros(12));two=step(initial,jp.zeros(12));jax.block_until_ready((one,two))
        replay=all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
            zip(jax.tree_util.tree_leaves(one),jax.tree_util.tree_leaves(two)))
        return dict(summary=dict(kind="official_environment_interface_smoke",policy="constant neutral normalized motor input",
            controller_training_status="no policy-performance claim",num_quads=3,nq=env.sys.nq,nv=env.sys.nv,nu=env.sys.nu,
            global_observation_dimension=int(state.obs.size),local_observation_dimension=34,
            control_frequency_hz=env.cfg.policy_freq,physics_steps=steps,simulated_seconds=float(state.pipeline_state.time),
            reset_compile_seconds=reset_seconds,step_compile_and_execution_seconds=elapsed,
            finite_states=bool(np.isfinite(states).all() and np.isfinite(velocities).all() and np.isfinite(motors).all()),finite_rewards=bool(np.isfinite(rewards).all()),
            deterministic_state_replay=replay,first_done_step=next((i+1 for i,x in enumerate(dones) if x),None)),
            qpos=np.asarray(states),qvel=np.asarray(velocities),motor_rpm_proxy=np.asarray(motors),rewards=np.asarray(rewards),done=np.asarray(dones))


def _evaluate_actor(checkout,ippo,network,params,env_config,seeds,steps):
    import jax
    import jax.numpy as jp
    from crazymarl.envs.multi_quad_env import MultiQuadEnv
    from crazymarl.observations.multi_quad_observation import get_ix4_mappings
    kwargs=deepcopy(env_config);kwargs.pop("episode_length",None)
    env=MultiQuadEnv(**kwargs)
    _,mapping=get_ix4_mappings(3)
    indices=jp.stack([mapping[f"agent_{i}"] for i in range(3)])
    reset=jax.jit(env.reset);step=jax.jit(env.step)
    actor=jax.jit(lambda p,observation:network.apply(p,observation[indices],method=network.actor_forward).reshape(12))
    cases=[];traces=[]
    for seed in seeds:
        state=reset(jax.random.PRNGKey(seed));jax.block_until_ready(state)
        positions=[np.asarray(state.pipeline_state.xpos[env.ids["payload_body_id"]])];actions=[];rewards=[];dones=[]
        qpos=[np.asarray(state.pipeline_state.qpos)];qvel=[np.asarray(state.pipeline_state.qvel)]
        motors=[np.asarray(state.metrics["filtered_rpm_proxy"])];clock=[float(state.pipeline_state.time)]
        started=time.perf_counter()
        for _ in range(steps):
            action=actor(params,state.obs)
            state=step(state,action);jax.block_until_ready(state)
            positions.append(np.asarray(state.pipeline_state.xpos[env.ids["payload_body_id"]]));actions.append(np.asarray(action))
            rewards.append(float(state.reward));dones.append(bool(state.done));qpos.append(np.asarray(state.pipeline_state.qpos));clock.append(float(state.pipeline_state.time))
            qvel.append(np.asarray(state.pipeline_state.qvel));motors.append(np.asarray(state.metrics["filtered_rpm_proxy"]))
            if bool(state.done):break
        target=np.asarray(state.metrics["target_position"])
        cases.append(dict(seed=int(seed),executed_steps=len(actions),simulated_seconds=clock[-1],
            terminated=bool(state.done),finite_states=bool(np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(motors).all()),finite_rewards=bool(np.isfinite(rewards).all()),
            final_payload_error=float(np.linalg.norm(positions[-1]-target)),reward_sum=float(np.sum(rewards)),
            wall_seconds=time.perf_counter()-started,status="short-trained upstream IPPO smoke; unconverged"))
        traces.append(dict(payload_positions=np.asarray(positions),qpos=np.asarray(qpos),qvel=np.asarray(qvel),
                           motor_rpm_proxy=np.asarray(motors),actions=np.asarray(actions),
                           rewards=np.asarray(rewards),done=np.asarray(dones),time=np.asarray(clock)))
    return cases,traces


def run_ippo_smoke(checkout,output,*,num_envs=2,num_steps=16,updates=2,epochs=1,minibatches=2,
                   seed=0,evaluation_seeds=(101,102,103),evaluation_steps=256):
    import jax
    import jax.numpy as jp
    from flax import serialization
    import wandb
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    artifacts=[output.with_suffix(suffix) for suffix in (".json",".msgpack",".npz",".log")]
    if any(path.exists() for path in artifacts):raise FileExistsError("Preserve previous external-baseline outputs")
    provenance=upstream_provenance(checkout)
    source_snapshot=output.with_name(output.name+"_wrapper.py")
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    wrapper_digest=sha(source_snapshot)
    full=official_config(checkout)
    config=deepcopy(full)
    config.update(NUM_ENVS=int(num_envs),NUM_STEPS=int(num_steps),TOTAL_TIMESTEPS=int(num_envs*num_steps*updates),
        NUM_MINIBATCHES=int(minibatches),UPDATE_EPOCHS=int(epochs),SEED=int(seed),WANDB_MODE="disabled")
    if min(num_envs,num_steps,updates,epochs,minibatches)<1 or 3*num_envs*num_steps%minibatches:
        raise ValueError("Positive smoke budget with evenly divisible actor minibatches required")
    output.with_name(output.name+"_official_full_config.json").write_text(json.dumps(full,indent=2)+"\n")
    output.with_name(output.name+"_smoke_config.json").write_text(json.dumps(config,indent=2)+"\n")
    started=time.perf_counter()
    with upstream_context(checkout),output.with_suffix(".log").open("w") as log,redirect_stdout(log):
        ippo,adapter=load_upstream_ippo(checkout)
        wandb.init(project="crazymarl-local-integration",mode="disabled")
        rng=jax.random.PRNGKey(seed);rng,rng_init=jax.random.split(rng)
        train=jax.jit(ippo.make_train(config,rng_init))
        network=ippo.ActorCritic(action_dim=4,activation=config["ACTIVATION"],
            actor_arch=config["ACTOR_ARCH"],critic_arch=config["CRITIC_ARCH"])
        initial_params=network.init(rng_init,jp.zeros((34,)))
        begin=time.perf_counter();result=train(rng);jax.block_until_ready(result)
        if hasattr(jax,"effects_barrier"):jax.effects_barrier()
        training_seconds=time.perf_counter()-begin
        state=result["runner_state"][0];params=state.params
        if not _finite_tree(params,jax) or not _finite_tree(result["metrics"],jax):
            raise FloatingPointError("Official IPPO smoke produced nonfinite parameters or metrics")
        differences=[np.asarray(a)-np.asarray(b) for a,b in
            zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(initial_params))]
        update_norm=float(np.sqrt(sum(np.sum(x*x) for x in differences)))
        if update_norm<=0 or int(state.step)==0:raise AssertionError("No genuine optimizer update occurred")
        template=dict(params=params,opt_state=state.opt_state,step=state.step)
        payload=serialization.to_bytes(template);checkpoint=output.with_suffix(".msgpack");checkpoint.write_bytes(payload)
        restored=serialization.from_bytes(template,checkpoint.read_bytes())
        exact=all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
            zip(jax.tree_util.tree_leaves(template),jax.tree_util.tree_leaves(restored)))
        if not exact:raise AssertionError("Upstream IPPO checkpoint reload differs")
        cases,traces=_evaluate_actor(checkout,ippo,network,restored["params"],config["ENV_KWARGS"],evaluation_seeds,evaluation_steps)
        wandb.finish()
    metrics={key:np.asarray(value) for key,value in result["metrics"].items()}
    arrays={"metric_"+key:value for key,value in metrics.items()}
    for index,trace in enumerate(traces):
        arrays.update({f"case{index}_"+key:value for key,value in trace.items()})
    np.savez_compressed(output.with_suffix(".npz"),**arrays)
    report=dict(kind="official_upstream_ippo_training_smoke",training_status="smoke_unconverged",
        performance_baseline_ready=False,upstream=provenance,adapter=adapter,config=config,
        official_full_config_path=str(output.with_name(output.name+"_official_full_config.json")),
        smoke_config_path=str(output.with_name(output.name+"_smoke_config.json")),
        requested_environment_transitions=int(num_envs*num_steps*updates),agent_transitions=int(3*num_envs*num_steps*updates),
        optimizer_steps=int(state.step),parameter_update_l2=update_norm,
        parameter_count=sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)),
        checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),checkpoint_reload_exact=exact,
        raw_path=str(output.with_suffix(".npz")),raw_sha256=sha(output.with_suffix(".npz")),
        training_compile_and_execution_seconds=training_seconds,total_wall_seconds=time.perf_counter()-started,
        jax_backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],
        versions={key:importlib.metadata.version(key) for key in ("jax","jaxlib","brax","mujoco","mujoco-mjx","flax","optax","numpy","scipy")},
        evaluation_cases=cases,metrics={key:value.tolist() for key,value in metrics.items()},
        source_path=str(source_snapshot),source_sha256=wrapper_digest)
    if sha(__file__)!=wrapper_digest:raise AssertionError("Wrapper source changed during execution")
    for path,digest in provenance["core_source_sha256"].items():
        if sha(Path(checkout)/path)!=digest:raise AssertionError("Upstream source changed during execution")
    output.with_suffix(".json").write_text(json.dumps(_plain(report),indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout",type=Path,default=Path("data/external/CrazyMARL"))
    parser.add_argument("--out",type=Path,default=Path("data/uav_transport_external/crazymarl_ippo_smoke"))
    parser.add_argument("--environment-only",action="store_true")
    parser.add_argument("--num-envs",type=int,default=2)
    parser.add_argument("--num-steps",type=int,default=16)
    parser.add_argument("--updates",type=int,default=2)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--minibatches",type=int,default=2)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--evaluation-steps",type=int,default=256)
    parser.add_argument("--platform",choices=("cpu","gpu"),default="cpu")
    args=parser.parse_args()
    os.environ["JAX_PLATFORMS"]="cuda,cpu" if args.platform=="gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    os.environ.setdefault("MUJOCO_GL","egl")
    os.environ["WANDB_MODE"]="disabled"
    args.checkout=args.checkout.resolve();args.out=args.out.resolve()
    if args.environment_only:
        provenance=upstream_provenance(args.checkout)
        result=run_environment_smoke(args.checkout,steps=args.num_steps,seed=args.seed)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        if args.out.with_suffix(".json").exists() or args.out.with_suffix(".npz").exists():raise FileExistsError(args.out)
        np.savez_compressed(args.out.with_suffix(".npz"),**{key:value for key,value in result.items() if key!="summary"})
        report=dict(result["summary"],upstream=provenance,raw_path=str(args.out.with_suffix(".npz")),
            raw_sha256=sha(args.out.with_suffix(".npz")),source_sha256=sha(__file__))
        args.out.with_suffix(".json").write_text(json.dumps(_plain(report),indent=2)+"\n")
    else:
        report=run_ippo_smoke(args.checkout,args.out,num_envs=args.num_envs,num_steps=args.num_steps,
            updates=args.updates,epochs=args.epochs,minibatches=args.minibatches,seed=args.seed,
            evaluation_steps=args.evaluation_steps)
    print(json.dumps(_plain(report),indent=2),flush=True)


if __name__=="__main__":main()
