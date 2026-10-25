"""Native Navigation collection; whole-geometry split and exact action replay."""
import types
import numpy as np
import torch
from experiments.e4_navigation import reproduce_baselines as native

MODES=['noisy_pd','priority_pd','yield_pd','random_ou']
def layouts():
    rng=np.random.default_rng(2026105301);rows=[]
    for i in range(16):
        angle=float(rng.uniform(45,135));theta=float(rng.uniform(-np.pi,np.pi));center=rng.uniform(-.08,.08,2);separation=float(rng.uniform(.34,1.10))
        d=np.array([[np.cos(theta),np.sin(theta)],[np.cos(theta+np.radians(angle)),np.sin(theta+np.radians(angle))]],np.float32)
        offset=separation/(2*np.sin(np.radians(angle)/2));velocity=rng.uniform(0,.45,(2,1))*d
        rows.append(dict(geometry_id=f'nav_trainpilot_{i:02}',split='train' if i<12 else 'validation',angle_deg=angle,
            starts=(center-offset*d).astype(np.float32).tolist(),goals=(center+.75*d).astype(np.float32).tolist(),velocity=velocity.astype(np.float32).tolist()))
    return rows

def observation(env):return torch.stack([env.scenario.observation(a) for a in env.agents],1).detach().clone()

def trace_substeps(env):
    w=env.scenario.world;original=w._integrate_state;trace=[]
    def integrate(world,entity,substep):
        original(entity,substep)
        if entity is world.agents[-1]:trace.append(native.state(env).detach().clone())
    w._integrate_state=types.MethodType(integrate,w)
    return trace

def collect(geometries,steps=128):
    states=[];actions=[];observations=[];substates=[];metadata=[]
    for gi,geo in enumerate(geometries):
        ns=native.context('S1_symmetric','cpu',steps+1);ns['STARTS']=np.array(geo['starts'],np.float32);ns['GOALS']=np.array(geo['goals'],np.float32)
        env=ns['create_scenario'](4);initial=np.column_stack([geo['starts'],geo['velocity']]).astype(np.float32)
        ns['set_cem_state'](env,initial[:,:2],initial[:,2:]);rng=np.random.default_rng(2026180000+gi);original_trace=trace_substeps(env)
        ss=[native.state(env).detach().clone()];oo=[observation(env)];aa=[];ou=np.zeros((2,2),np.float32);phase=0
        for t in range(steps):
            s=native.state(env).detach().numpy();goal=np.array(geo['goals'],np.float32);pd=1.5*(goal[None]-s[:,:,:2])-s[:,:,2:]
            act=pd.copy();act[0]+=rng.normal(0,.05,(2,2))
            if np.linalg.norm(s[1,0,:2]-goal[0])<.12:phase=1
            act[1,1-phase]=-s[1,1-phase,2:];act[1]+=rng.normal(0,.02,(2,2))
            if (t//20)%2:act[2,0]=-s[2,0,2:]
            else:act[2,1]*=.3
            act[2]+=rng.normal(0,.25,(2,2));ou=.8*ou+rng.normal(0,.25,(2,2));act[3]=ou
            act=np.clip(act,-1,1).astype(np.float32);at=torch.from_numpy(act)
            env.step([at[:,i] for i in range(2)]);aa.append(at.clone());ss.append(native.state(env).detach().clone());oo.append(observation(env))
        s=torch.stack(ss,1);a=torch.stack(aa,1);o=torch.stack(oo,1)
        assert s.shape==(4,steps+1,2,4) and o.shape==(4,steps+1,2,18)
        assert torch.equal(s,o[...,:4]) and torch.isfinite(s).all() and torch.isfinite(o).all() and a.abs().max()<=1
        replay=ns['create_scenario'](4);ns['set_cem_state'](replay,initial[:,:2],initial[:,2:])
        trace=trace_substeps(replay)
        error=0.
        for t in range(steps):
            replay.step([a[:,t,i] for i in range(2)]);error=max(error,float((native.state(replay)-s[:,t+1]).abs().max()))
        assert error==0;sub=torch.stack(original_trace,1);assert torch.equal(sub,torch.stack(trace,1)) and torch.equal(sub[:,1::2],s[:,1:])
        states.append(s);actions.append(a);observations.append(o);substates.append(sub)
        metadata.extend(dict(episode_id=f"{geo['geometry_id']}_{mode}",geometry_id=geo['geometry_id'],split=geo['split'],policy=mode,seed=2026180000+gi) for mode in MODES)
        print('COLLECT',gi+1,'/16',flush=True)
    return dict(state=torch.cat(states),action=torch.cat(actions),native_observation=torch.cat(observations),substep_state=torch.cat(substates),
        goals=torch.tensor([g['goals'] for g in geometries for _ in MODES],dtype=torch.float32)),metadata

def sample(states,actions,episodes,batch,history,k,generator):
    total=states.shape[1]-1;frames=history+k
    if total<frames-1:raise ValueError('Episode shorter than requested context/rollout')
    e=episodes[torch.randint(len(episodes),(batch,),generator=generator)];t=torch.randint(total-frames+2,(batch,),generator=generator)
    idx=e.to(states.device);rows=(t[:,None]+torch.arange(frames)[None]).to(states.device)
    return states[idx[:,None],rows],actions[idx[:,None],rows[:,:-1]],e,t

def free_rollout(initial,actions):
    """Known native free motion baseline, explicitly ignores inter-agent contact."""
    pos=initial[...,:2].clone();vel=initial[...,2:].clone();out=[]
    for t in range(actions.shape[1]):
        vel=vel*.75
        for _ in range(2):vel=vel+actions[:,t]*.05;pos=pos+vel*.05
        out.append(torch.cat([pos,vel],-1))
    return torch.stack(out,1)
