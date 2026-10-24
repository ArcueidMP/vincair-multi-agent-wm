"""Headless reproduction of the team's fixed two-agent VMAS notebook baselines.

Planning functions are loaded verbatim as AST FunctionDefs: no Colab/install or
cached notebook output is executed. Safety instrumentation never clips dynamics.
"""
from __future__ import annotations
import argparse,ast,hashlib,importlib.metadata,json,math,subprocess,time,types
from pathlib import Path
import numpy as np
import torch
import vmas

HERE=Path(__file__).resolve().parent
BASE_COMMIT='a4ff32623f4287966aef878d44f01d3521e660af'
SCENARIOS={
 'S1_symmetric':([[-.8,-.8],[-.8,.8]],[[.8,.8],[.8,-.8]]),
 'S2_perpendicular':([[-.8,0],[.2,.8]],[[.8,0],[.2,-.8]]),
 'S3_asymmetric':([[-.8,.5],[-.8,-.8]],[[.8,-.5],[.8,.7]])}
CONFIG=dict(horizon=8,population=64,iterations=3,elite_fraction=.125,max_steps=200,goal_weight=5.,action_weight=.01,safety_weight=100.,seed=42)
METHODS=['independent_pd','priority_pd','independent_cem','centralized_no_safety','centralized_safety']
SAFE=.30;TOL=.12
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def functions(namespace,file,cells,names):
    nb=json.loads((HERE/'notebooks'/file).read_text(encoding='utf-8'))
    nodes=[]
    for i in cells:
        tree=ast.parse(''.join(nb['cells'][i]['source']))
        nodes.extend(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),namespace)
def context(scene,device='cuda',max_steps=200):
    starts,goals=SCENARIOS[scene];cfg=dict(CONFIG,max_steps=max_steps)
    ns=dict(vmas=vmas,torch=torch,np=np,time=time,math=math,DEVICE=device,SEED=42,MAX_STEPS=max_steps,
        GOAL_TOL=TOL,SAFE_DISTANCE=SAFE,STARTS=np.array(starts,np.float32),GOALS=np.array(goals,np.float32),CEM_CONFIG=cfg)
    functions(ns,'03_Centralized_CEM.ipynb',[1,2],['create_scenario','set_cem_state','centralized_cem'])
    functions(ns,'02_Independent_CEM.ipynb',[4],['independent_cem'])
    return ns
def state(env):return torch.stack([torch.cat([a.state.pos,a.state.vel],-1) for a in env.agents],1)
def trace_substeps(env):
    world=env.scenario.world;original=world._integrate_state;trace=[]
    def integrate(w,entity,substep):
        original(entity,substep)
        if entity is w.agents[-1]:trace.append(state(env)[0].detach().clone())
    world._integrate_state=types.MethodType(integrate,world)
    return trace
def calculate_metrics(states,substates,goals,radii):
    sep=np.linalg.norm(states[:,0,:2]-states[:,1,:2],axis=-1)
    allstates=np.concatenate([states[:1],substates]) if len(substates) else states[:1]
    sub=np.linalg.norm(allstates[:,0,:2]-allstates[:,1,:2],axis=-1)
    finaldist=np.linalg.norm(states[-1,:,:2]-goals,axis=1)
    success=bool((finaldist<TOL).all())
    return dict(joint_success=success,minimum_separation=float(sep.min()),loss_of_separation=bool((sep<SAFE).any()),
        collision_proxy=bool((sep<sum(radii)).any()),violation_steps=int((sep<SAFE).sum()),
        cumulative_violation=float(np.maximum(SAFE-sep,0).sum()),
        substep_minimum_separation=float(sub.min()),substep_loss_of_separation=bool((sub<SAFE).any()),
        substep_collision_proxy=bool((sub<sum(radii)).any()),safe_joint_success=success and bool((sub>=SAFE).all()),
        final_goal_distances=finaldist.tolist(),final_speeds=np.linalg.norm(states[-1,:,2:],axis=1).tolist(),
        path_lengths=np.linalg.norm(np.diff(states[:,:,:2],axis=0),axis=-1).sum(0).tolist())
def episode(scene,method,seed,device='cuda',max_steps=200):
    assert method in METHODS;ns=context(scene,device,max_steps);env=ns['create_scenario'](1)
    sim=ns['create_scenario'](CONFIG['population']) if 'cem' in method or method.startswith('centralized') else None
    subtrace=trace_substeps(env);states=[state(env)[0].detach().cpu().numpy().copy()];actions=[];latencies=[];phase=0;termination='max_steps';started=time.perf_counter()
    for step in range(max_steps):
        current=state(env)[0].detach().cpu().numpy();dist=np.linalg.norm(current[:,:2]-ns['GOALS'],axis=1)
        if (dist<TOL).all():termination='joint_success';break
        if device.startswith('cuda'):torch.cuda.synchronize()
        begin=time.perf_counter()
        if method in ['independent_pd','priority_pd']:
            if method=='priority_pd' and dist[0]<TOL:phase=1
            a=[]
            for i,agent in enumerate(env.agents):
                moving=method=='independent_pd' or i==phase
                u=1.5*(agent.goal.state.pos-agent.state.pos)-agent.state.vel if moving else -agent.state.vel
                a.append(torch.clamp(u,-float(agent.u_range),float(agent.u_range))[0])
            action=torch.stack(a)
        elif method=='independent_cem':
            action=torch.stack([ns['independent_cem'](sim,current[:,:2],current[:,2:],i,ns['CEM_CONFIG'],seed=seed+step*2+i)[0] for i in range(2)])
        else:
            action,_=ns['centralized_cem'](sim,current[:,:2],current[:,2:],ns['CEM_CONFIG'],with_safety=method=='centralized_safety',seed=seed+step)
        if device.startswith('cuda'):torch.cuda.synchronize()
        latencies.append((time.perf_counter()-begin)*1000)
        _,_,done,_=env.step([action[i][None] for i in range(2)])
        actions.append(action.detach().cpu().numpy().copy());states.append(state(env)[0].detach().cpu().numpy().copy())
        if bool(torch.all(done)):termination='environment_done';break
    substates=torch.stack(subtrace).cpu().numpy() if subtrace else np.empty((0,2,4),np.float32)
    states=np.array(states);actions=np.array(actions,dtype=np.float32).reshape(-1,2,2)
    radii=[float(a.shape.radius) for a in env.agents];metrics=calculate_metrics(states,substates,ns['GOALS'],radii)
    if metrics['joint_success']:termination='joint_success'
    world=env.scenario.world
    physics=dict(dt=world.dt,substeps=world._substeps,radii=radii,action_limits=[float(a.u_range) for a in env.agents],
        mass=[float(a.mass) for a in env.agents],drag=[a.drag for a in env.agents],world_drag=world._drag,max_speed=[a.max_speed for a in env.agents])
    # Independent fresh native environment replays every action; instrumentation has no dynamics intervention.
    replay=ns['create_scenario'](1);error=0.
    for k,a in enumerate(actions):
        at=torch.as_tensor(a,device=device)
        replay.step([at[i][None] for i in range(2)])
        error=max(error,float(abs(state(replay)[0].cpu().numpy()-states[k+1]).max()))
    assert error<=1e-6
    result=dict(scene=scene,method=method,seed=seed,steps=len(actions),termination=termination,**metrics,
        mean_planning_ms=float(np.mean(latencies)) if latencies else 0.,p95_planning_ms=float(np.percentile(latencies,95)) if latencies else 0.,
        wall_seconds=time.perf_counter()-started,native_replay_max_abs_error=error,physics=physics)
    # No renderer is initialized; VMAS1.5.2 Environment has no close() method.
    return result,dict(state=states,substep_state=substates,action=actions,goals=ns['GOALS'],latency_ms=np.array(latencies))
def tasks(suite):
    if suite=='smoke':return [('S1_symmetric',m,42) for m in METHODS]
    return [(s,m,seed) for s in SCENARIOS for seed in [42,43,44] for m in ['centralized_no_safety','centralized_safety']]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=['smoke','full'],required=True);parser.add_argument('--device',default='cuda',choices=['cpu','cuda']);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(base_commit=BASE_COMMIT,config=CONFIG,scenarios=SCENARIOS,device=args.device,step_safety=SAFE,goal_tolerance=TOL,
        suites=dict(smoke=tasks('smoke'),full=tasks('full')),source_sha256=sha(Path(__file__)),
        notebook_sha256={p.name:sha(p) for p in (HERE/'notebooks').glob('*.ipynb')},
        versions={n:importlib.metadata.version(n) for n in ['vmas','torch','numpy']},
        notes=['Original native VMAS: no bounded-acceleration intervention.','Seeds vary CEM RNG on three fixed geometries, not nine independent scene geometries.',
        'Substep metrics are additional diagnostics; original position-only success and step metrics preserved.','True simulator baseline only; no learned Navigation model.'])
    if (args.output/'protocol.json').exists():assert json.loads((args.output/'protocol.json').read_text())==json.loads(json.dumps(protocol))
    else:write(args.output/'protocol.json',protocol)
    start=time.perf_counter()
    for scene,method,seed in tasks(args.suite):
        dest=args.output/f'{scene}_{method}_{seed}'
        if (dest/'result.json').exists():
            saved=json.loads((dest/'result.json').read_text());assert saved['native_replay_max_abs_error']<=1e-6 and (dest/'trajectory.npz').exists();continue
        assert not dest.exists(),'Preserve incomplete attempt for inspection'
        if time.perf_counter()-start>1800:raise RuntimeError('Suite wall budget1800s exceeded; retain completed runs')
        dest.mkdir();result,arrays=episode(scene,method,seed,args.device)
        np.savez_compressed(dest/'trajectory.npz',**arrays);result['trajectory_sha256']=sha(dest/'trajectory.npz');write(dest/'result.json',result)
        print('EPISODE',json.dumps(result),flush=True)
    write(args.output/f'{args.suite}_completed.json',dict(status='passed',wall_seconds=time.perf_counter()-start,suite=args.suite))
if __name__=='__main__':main()
