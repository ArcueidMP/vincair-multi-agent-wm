"""Gate-controlled bounded paired closed-loop follow-up; freeze details before confirmation."""
import argparse,datetime,time,shutil
import numpy as np
import torch
from phase22_common import *
from world import MultiPushT
from policies import CoopPolicy
parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
plan=OUT/'closed_loop_protocol.json'
if args.freeze_only:
    assert not plan.exists() and not (OUT/'confirmation_summary.json').exists()
    write(plan,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),
        seeds=list(range(2026103100,2026103104)),arms=['readout','selected'],budget=150,push_execute=10,
        ready_px=5,approach_timeout=30,tabu_steps=30,finish_position_px=20,finish_angle_deg=20,
        worker='Same64 mixed pool, fixed16x3 global-CEM arm; either original global readout or frozen selected residual.',
        coordinator='Previous privileged _choose/_walk approach/push/finish mechanism, identical across arms.',
        success='Original environment eval_state; independently verify norm([agent0,T]position_error)<20 and circular Tangle<20deg.',
        gate='Exactly main protocol and confirmation_summary; do not run when false.',
        limitations=['Four fresh scenes only.','Privileged coordinator/candidates, not pure image planning.',
            'Offline candidate improvement need not improve closed-loop success.']))
    shutil.copy2(Path(__file__),ROOT/'scripts/phase22_v1'/Path(__file__).name);print('CLOSED LOOP DETAILS FROZEN');raise SystemExit(0)
spec=json.loads(plan.read_text());assert sha(Path(__file__))==spec['source_sha256']
confirmation=json.loads((OUT/'confirmation_summary.json').read_text());folder=OUT/'closed_loop';folder.mkdir(exist_ok=True)
assert not (folder/'summary.json').exists()
if not confirmation['closed_loop_gate']:
    write(folder/'summary.json',dict(status='skipped',reason='Predeclared confirmation gate not met',conditions=confirmation['gate_conditions']))
    print('CLOSED LOOP SKIPPED BY GATE');raise SystemExit(0)
ck=torch.load(OUT/'selected_scorer.pt',map_location='cpu',weights_only=False)
assert sha(OUT/'selected_scorer.pt')==confirmation['checkpoint_sha256']
if ck['kind']=='mlp':net=ResidualMLP().eval();net.load_state_dict(ck['state_dict'])
def learned_score(features,pc):
    nx=(features-ck['mean'].numpy())/ck['std'].numpy()
    if ck['kind']=='zero':return pc/100
    if ck['kind']=='ridge':return pc/100+np.clip(nx@ck['beta'].numpy(),-.25,.25)
    with torch.no_grad():return pc/100+net(torch.from_numpy(nx)).numpy()
engine=Engine();results=[];start=time.perf_counter()
for seed in spec['seeds']:
    for arm in spec['arms']:
        env=MultiPushT(max_episode_steps=10000,**ENV);env.reset(seed=seed);goal=env.core.goal_state.copy();t=0;pair=None;walk=0
        cp=CoopPolicy(np.random.default_rng([seed,111]),noise=0,smooth=0,p_solo=0);cp.reset(env.core,0)
        goalz=engine.encode([env.core.goals[0]]);pg=engine.decode(goalz)[0]
        states=[env.state().copy()];bodies=[body(env.core)];aa=[];cc=[];phases=[];windows=[];success=None
        dest=folder/f'{arm}_{seed}';dest.mkdir();calls=dest/'calls';calls.mkdir()
        def errors():
            s=env.state();return float(np.linalg.norm(s[4:6]-goal[4:6])),float(abs(np.arctan2(np.sin(s[6]-goal[6]),np.cos(s[6]-goal[6]))))
        def push(a,phase):
            global t,success
            assert np.isfinite(a).all() and abs(a).max()<=1.000001
            env.core.simulate(a);t+=1;aa.append(a.copy());cc.append(env.core.block_contact.copy());states.append(env.state().copy());bodies.append(body(env.core));phases.append(phase)
            if success is None and env.core.eval_state(goal,env.state())[0]:success=t
        while t<150 and success is None:
            cp.step=t;e,ang=errors()
            if e<20 and ang<np.radians(20):
                pair=None;push(np.stack([cp._walk(env.core,i,goal[2*i:2*i+2]) for i in range(2)]).astype(np.float32),'finish');continue
            if pair is None:
                pair=cp._choose(env.core);walk=0
                if pair is None:push(np.zeros((2,2),np.float32),'no_pair');continue
            pts,nrm=cp.contacts;targets=np.array([env.core.block.local_to_world(tuple(pts[j]+nrm[j]*(cp.radius+cp.standoff))) for j in pair])
            pos=np.array([b.position for b in env.core.agents]);ready=np.linalg.norm(targets-pos,axis=1)<5
            if not ready.all():
                push(np.stack([np.zeros(2) if ready[i] else cp._walk(env.core,i,targets[i]) for i in range(2)]).astype(np.float32),'approach');walk+=1
                if walk>30:cp.tabu[tuple(sorted(pair))]=t+30;pair=None
                continue
            inside=np.array([env.core.block.local_to_world(tuple(pts[j]-nrm[j]*30)) for j in pair]);direction=inside-pos
            direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
            z0=engine.encode([env.core.render_view(0)]);p0=engine.decode(z0)[0];actions,_=engine.pools(z0,pg,seed,t,direction)
            zp=engine.predict(z0,actions);pp=engine.decode(zp[:,-1]);pc=physical_cost(pp,pg);features=low_features(pp,pg,p0,actions)
            score=pc/100 if arm=='readout' else learned_score(features,pc);chosen=int(score.argmin());begin=t
            np.savez_compressed(calls/f'call{len(windows):03d}.npz',actions=actions,features=features,readout_cost=pc,score=score,selected=chosen,t=t)
            for a in actions[chosen,:min(10,150-t)]:
                push(a,'push');pe,pa=errors()
                if success is not None or (pe<20 and pa<np.radians(20)):break
            pe,pa=errors();windows.append(dict(t_before=begin,t_after=t,selected_index=chosen,proposal=ARMS[chosen//16]))
            if pe>e-2 and pa>ang-np.radians(2):cp.tabu[tuple(sorted(pair))]=t+30
            pair=None
        replay=MultiPushT(max_episode_steps=10000,**ENV);replay.reset(seed=seed);maxerr=0.;maxbody=0.
        for k,a in enumerate(aa):
            replay.core.simulate(a);maxerr=max(maxerr,float(abs(replay.state()-states[k+1]).max()));maxbody=max(maxbody,float(abs(body(replay.core)-bodies[k+1]).max()))
            assert np.array_equal(replay.core.block_contact,cc[k])
        assert maxerr==maxbody==0
        arr=np.array(states);spos=np.linalg.norm(arr[:,[0,1,4,5]]-goal[[0,1,4,5]],axis=1)
        sang=abs(np.arctan2(np.sin(arr[:,6]-goal[6]),np.cos(arr[:,6]-goal[6])));ok=(spos<20)&(sang<np.radians(20))
        assert (success is not None)==bool(ok[1:].any())
        record=dict(seed=seed,arm=arm,success=success is not None,success_step=success,steps=t,calls=len(windows),
            actual_learned_scores=len(windows)*112,final_global_cost=float(physical_cost(pose(arr[-1]),pose(goal))),
            final_T_cost=float(tcost(pose(arr[-1]),pose(goal))),native_replay_error=maxerr,body_replay_error=maxbody)
        np.savez_compressed(dest/'trajectory.npz',state=arr,body=np.array(bodies),action=np.array(aa),contact=np.array(cc),phase=np.array(phases),goal=goal)
        write(dest/'results.json',record);write(dest/'windows.json',windows);results.append(record);write(folder/'episodes.json',results)
        replay.close();env.close();print('CLOSED EPISODE',json.dumps(record),flush=True)
assert len(results)==8
write(folder/'summary.json',dict(status='passed',episodes=results,wall_seconds=time.perf_counter()-start,
    methods={arm:dict(n=4,successes=sum(q['success'] for q in results if q['arm']==arm),mean_final_global_cost=float(np.mean([q['final_global_cost'] for q in results if q['arm']==arm]))) for arm in spec['arms']}))
