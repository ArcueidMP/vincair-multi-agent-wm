"""Frozen direct-scorer receding-horizon pilot in native VMAS; not LeWM."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import torch
import conflict_ranking as cr
import reproduce_baselines as base
import repair_safety as repair
import state_action_ranker as sr

ARMS=['pd','original','raw_repair','invariant_repair','true_sim_safety']

def geometries():
    rng=np.random.default_rng(2026104801);rows=[]
    for i in range(6):
        angle=float(rng.uniform(45,135));theta=float(rng.uniform(-np.pi,np.pi));center=rng.uniform(-.08,.08,2)
        d=np.array([[np.cos(theta),np.sin(theta)],[np.cos(theta+np.radians(angle)),np.sin(theta+np.radians(angle))]],np.float32)
        rows.append(dict(name=f'closed_{i:02}',angle_deg=angle,directions=d.tolist(),
            starts=(center-.75*d).tolist(),goals=(center+.75*d).tolist(),velocity=(np.array([.12,.24])[:,None]*d).tolist()))
    return rows

def pd(state,goals):return np.clip(1.5*(goals-state[:,:2])-state[:,2:],-1,1).astype(np.float32)

def candidates(ns,state,goals,directions,seed):
    # Avoid zero route vector when one agent happens to be exactly at its goal.
    starts=state[:,:2].copy();atgoal=np.linalg.norm(goals-starts,axis=1)<1e-6
    starts[atgoal]=goals[atgoal]-np.asarray(directions)[atgoal]*1e-5
    route=ns['generate_conflict_actions'](dict(starts=starts,goals=goals),64,8,seed).cpu().numpy()
    pool=repair.proposals(state,route);feedback=pd(state,goals);brake=np.clip(-state[:,2:]/.1,-1,1)
    extra=np.repeat(feedback[None,None],4,axis=0).repeat(8,axis=1)
    extra[1,:,1]=brake[1];extra[2,:,0]=brake[0];extra[3]*=.5
    return np.concatenate([pool,extra]).astype(np.float32)

def load_models(oldroot,newroot):
    oldfile=oldroot/'state_action_scorer.pt';freeze=json.loads((newroot/'selection_frozen.json').read_text())
    assert sr.sha(oldfile)==freeze['baseline_sha256']
    result={};hashes={}
    for name,path,kind in [('original',oldfile,'old'),('raw_repair',newroot/'raw.pt','raw'),('invariant_repair',newroot/'invariant.pt','invariant')]:
        if kind!='old':assert sr.sha(path)==freeze['models'][kind]['checkpoint_sha256']
        c=torch.load(path,map_location='cpu',weights_only=False);net=sr.Scorer().eval();net.load_state_dict(c['state_dict'])
        mean=c['mean'].numpy() if torch.is_tensor(c['mean']) else c['mean'];std=c['std'].numpy() if torch.is_tensor(c['std']) else c['std']
        result[name]=dict(net=net,checkpoint=c,kind=kind,mean=mean,std=std);hashes[name]=sr.sha(path)
    return result,hashes

def decide(model,state,goals,actions):
    c=model['checkpoint'];kind=model['kind']
    x=repair.invariant_features(state,goals,actions) if kind=='invariant' else sr.features(state,goals,actions)[:,None]
    raw=repair.infer(model['net'],x,model['mean'],model['std'])
    if kind=='old':idx,fb,*_=sr.select(raw[None],c['cost_mean'],c['cost_std'],c['threshold'])
    else:idx,fb=repair.choose(raw[None],c['cost_mean'],c['cost_std'],**c['policy'])
    return int(idx[0]),bool(fb[0]),raw

def episode(geo,arm,seed,models,steps):
    ns=cr.namespace('S1_symmetric',geo['goals']);ns['MAX_STEPS']=steps
    env=ns['create_scenario'](1);initial=np.concatenate([geo['starts'],geo['velocity']],axis=1).astype(np.float32)
    ns['set_cem_state'](env,initial[:,:2],initial[:,2:]);trace=base.trace_substeps(env)
    goals=np.array(geo['goals'],np.float32);states=[initial.copy()];actions=[];pools=[];predictions=[];indices=[];fallbacks=[];latency=[];oraclecost=[];oraclemin=[]
    for step in range(steps):
        current=base.state(env)[0].cpu().numpy().copy()
        if (np.linalg.norm(current[:,:2]-goals,axis=1)<.12).all():break
        torch.cuda.synchronize();started=time.perf_counter();raw=np.zeros((72,3),np.float32);fallback=False
        if arm=='pd':action=pd(current,goals);index=-1
        else:
            pool=candidates(ns,current,goals,geo['directions'],seed+step)
            if arm=='true_sim_safety':
                at=torch.tensor(pool,device='cuda');path,sub=repair.rollout(ns,current,at)
                cost=ns['calculate_ranking_labels'](path,at,goals)['total_cost']
                allsub=torch.cat([path[:,:1],sub],1);sep=torch.linalg.vector_norm(allsub[:,:,0,:2]-allsub[:,:,1,:2],dim=-1)
                minimum=sep.min(1).values.cpu().numpy();exposure=torch.relu(.3-sep).square().sum(1).cpu().numpy()
                index=int(cr.preference_order(cost,minimum,exposure)[0]);oraclecost.append(cost);oraclemin.append(minimum)
            else:index,fallback,raw=decide(models[arm],current,goals,pool)
            action=pool[index,0];pools.append(pool);predictions.append(raw)
        torch.cuda.synchronize();latency.append((time.perf_counter()-started)*1000)
        assert action.shape==(2,2) and np.isfinite(action).all() and abs(action).max()<=1.000001
        at=torch.tensor(action,device='cuda');env.step([at[i][None] for i in range(2)])
        actions.append(action.copy());indices.append(index);fallbacks.append(fallback);states.append(base.state(env)[0].cpu().numpy().copy())
    states=np.array(states);substates=torch.stack(trace).cpu().numpy();actions=np.array(actions,np.float32)
    metrics=base.calculate_metrics(states,substates,goals,[.1,.1])
    replay=ns['create_scenario'](1);ns['set_cem_state'](replay,initial[:,:2],initial[:,2:]);replaytrace=base.trace_substeps(replay);error=0.
    for i,a in enumerate(actions):
        at=torch.tensor(a,device='cuda');replay.step([at[k][None] for k in range(2)])
        error=max(error,float(abs(base.state(replay)[0].cpu().numpy()-states[i+1]).max()))
    suberror=float(abs(torch.stack(replaytrace).cpu().numpy()-substates).max());assert max(error,suberror)==0
    result=dict(geometry=geo['name'],arm=arm,seed=seed,steps=len(actions),**metrics,
        fallback_steps=int(sum(fallbacks)),mean_planning_ms=float(np.mean(latency)),p95_planning_ms=float(np.percentile(latency,95)),
        state_replay_error=error,substep_replay_error=suberror)
    arrays=dict(state=states,substep_state=substates,action=actions,goals=goals,selected=np.array(indices),fallback=np.array(fallbacks),
        latency_ms=np.array(latency),candidates=np.array(pools,np.float32),prediction=np.array(predictions,np.float32),
        oracle_cost=np.array(oraclecost,np.float32),oracle_minimum=np.array(oraclemin,np.float32))
    return result,arrays

def aggregate(rows):
    result={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];scenes=[]
        for geo in sorted({r['geometry'] for r in rr}):
            g=[r for r in rr if r['geometry']==geo]
            scenes.append(dict(geometry=geo,**{k:float(np.mean([q[k] for q in g])) for k in ['joint_success','safe_joint_success','substep_loss_of_separation','substep_collision_proxy','steps','substep_minimum_separation','mean_planning_ms']}))
        result[arm]=dict(episodes=len(rr),scenes=scenes,mean={k:float(np.mean([q[k] for q in scenes])) for k in scenes[0] if k!='geometry'})
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--repair',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Preserve prior attempt')
    torch.set_num_threads(1);models,hashes=load_models(a.baseline,a.repair);geos=geometries();started=time.perf_counter()
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sr.sha(Path(__file__)),
        parent_commit='1f73fee5913fa0d420088c69dea1653ffb5ffc4b',model_sha256=hashes,geometry_specs=geos,seeds=[2026130000,2026131000],arms=ARMS,
        episodes=60,max_steps=200,horizon=8,candidates=72,replan='Every step, execute first action only; current observed state only',
        proposals='64 route/noise controls +4 previous repair zero/brake +4 current-PD joint/agent0-priority/agent1-priority/half controls, repeated H8; no dynamics rollout for learned arms',
        matching='Same initial states, seed schedule, limits and candidate generation code. Pools diverge when policies visit different states. PD has no candidate pool; oracle has privileged true-simulator rollouts.',
        native='VMAS1.5.2 dt.1 two substeps radius.1 drag.25 action components[-1,1]; separation.30; joint position tolerance.12, no speed criterion',
        oracle='True substep safety-first pool ranking; finite H8 reference, not certified safe or a globally optimal controller; no learned WM',
        primary='Safe joint success by200 steps, averaged seeds within geometry then geometries equally. Also unsafe/collision/goal/latency and all paired geometries.',
        gate='Descriptive pilot only: repair safe-joint-success > original AND substep loss-of-separation <= original; no significance or deployment claim',
        smoke='Separate fixed geometry,5steps per arm; verify finite/legal controls, first-action execution and exact native state+substep replay before full run',
        budget='60 full episodes plus5 smoke; max200 steps each; abort between episodes after900 seconds; preserve partial outputs',
        no_training=True,no_policy_tuning=True,limitations=['Only6 independent geometries; two candidate seeds are nested repeats','Fresh starts farther from conflict than training offsets; PD candidates also shift proposal distribution','Synthetic two-agent task only, not Navigation LeWM or three-agent transfer'])
    sr.write(out/'protocol.json',protocol)
    smoke=dict(name='smoke',starts=[[-.7,-.3],[-.3,.7]],goals=[[.7,.3],[.3,-.7]],velocity=[[.1,.04],[.04,-.1]],directions=[[.919145,.393919],[.393919,-.919145]])
    smoke_results=[]
    def save(geo,arm,seed,steps,section):
        r,z=episode(geo,arm,seed,models,steps);p=out/section/f"{geo['name']}_{arm}_{seed}";p.mkdir(parents=True)
        np.savez_compressed(p/'trajectory.npz',**z);r['trajectory_sha256']=sr.sha(p/'trajectory.npz');sr.write(p/'result.json',r);return r
    for arm in ARMS:smoke_results.append(save(smoke,arm,2026139999,5,'smoke'))
    sr.write(out/'smoke_passed.json',dict(results=smoke_results,all_replay_exact=True));rows=[]
    for geo in geos:
        for seed in protocol['seeds']:
            for arm in ARMS:
                if time.perf_counter()-started>900:raise RuntimeError('Pilot wall budget reached; preserve partial run')
                r=save(geo,arm,seed,200,'episodes');rows.append(r);sr.write(out/'episode_results.json',rows)
                print('EPISODE',len(rows),'/60',geo['name'],arm,r['steps'],r['safe_joint_success'],flush=True)
    _,finalhash=load_models(a.baseline,a.repair);assert finalhash==hashes
    methods=aggregate(rows);old=methods['original']['mean'];new=methods['invariant_repair']['mean']
    sr.write(out/'summary.json',dict(methods=methods,gate_passed=bool(new['safe_joint_success']>old['safe_joint_success'] and new['substep_loss_of_separation']<=old['substep_loss_of_separation']),
        episodes=len(rows),geometry_count=6,model_hashes_unchanged=True,no_training=True,wall_seconds=time.perf_counter()-started))
    sr.write(out/'EVALUATION_CONSUMED.json',dict(reason='Frozen closed-loop pilot consumed; no tuning on these geometries',protocol_sha256=sr.sha(out/'protocol.json')))
    print('COMPLETE',json.dumps(methods),flush=True)
if __name__=='__main__':main()
