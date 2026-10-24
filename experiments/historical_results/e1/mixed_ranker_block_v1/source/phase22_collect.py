"""Collect fixed train or post-freeze confirmation mixed pools with separated inputs/labels."""
import argparse,time,datetime,shutil,importlib.metadata,subprocess
import numpy as np
import torch
from phase22_common import *
from world import MultiPushT
from policies import CoopPolicy
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['train','confirmation'],required=True);args=parser.parse_args()
protocol=json.loads((OUT/'protocol.json').read_text());seeds=protocol[f'{args.split}_seeds']
folder=OUT/f'{args.split}_data';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'metadata.json').exists()
if args.split=='confirmation':
    frozen=json.loads((OUT/'selection_frozen.json').read_text());assert sha(OUT/'selected_scorer.pt')==frozen['checkpoint_sha256']
meta=dict(status='running',split=args.split,seeds=seeds,protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(Path(__file__)),
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),packages={n:importlib.metadata.version(n) for n in ['torch','numpy','gymnasium','pymunk']},
    gpu=torch.cuda.get_device_name(),git_commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip(),
    git_diff=subprocess.check_output(['git','-C',str(CODE),'diff'],text=True),old_test_candidate_npz_opened=0)
write(folder/'metadata.json',meta);start=time.perf_counter();engine=Engine();records=[];maxbranch=0.;maxreplay=0.
prior=json.loads((ROOT/'results/e1/ranking_v2/data/scene_manifest.json').read_text())+json.loads((ROOT/'results/e1/proposal_coverage_v1/scene_manifest.json').read_text())
assert not set(seeds)&{q['seed'] for q in prior}
prior_states=[q['initial_state'] for q in prior]
if args.split=='confirmation':
    tr=json.loads((OUT/'train_data/manifest.json').read_text());assert not set(seeds)&{q['seed'] for q in tr}
    prior_states += [q['initial_state'] for q in tr]
try:
    for scene,seed in enumerate(seeds):
        env=MultiPushT(max_episode_steps=10000,**ENV);sim=MultiPushT(max_episode_steps=10000,**ENV)
        env.reset(seed=seed);initial=env.state().copy();goal=env.core.goal_state.copy()
        assert not np.any(np.all(np.isclose(np.array(prior_states),initial,rtol=0,atol=1e-3),axis=1))
        policy=CoopPolicy(np.random.default_rng(seed),noise=.03,smooth=.5,p_solo=.2)
        for i in range(2):policy.reset(env.core,i)
        states=[initial];bodies=[body(env.core)];actions=[];contacts=[];ready=None
        for t in range(160):
            a=np.stack([policy(env.core,i) for i in range(2)]).astype(np.float32)
            if ready is None and policy.mode=='push' and policy.plan is not None:ready=t
            env.core.simulate(a);actions.append(a);contacts.append(env.core.block_contact.copy());states.append(env.state().copy());bodies.append(body(env.core))
        states,bodies,actions,contacts=map(np.array,(states,bodies,actions,contacts));second=ready if ready is not None and 1<=ready<=140 else 75
        np.savez_compressed(folder/f'scene{scene:02d}_source.npz',state=states,body=bodies,action=actions,contact=contacts,goal=goal)
        rec=dict(scene=scene,seed=seed,split=args.split,initial_state=initial.tolist(),ready_t=ready,fallback=second!=ready,groups=[])
        for t in [0,int(second)]:
            def restore():
                global maxbranch
                sim.reset(seed=seed)
                for a in actions[:t]:sim.core.simulate(a)
                e=float(abs(body(sim.core)-bodies[t]).max());maxbranch=max(maxbranch,e);assert e==0
            restore();pixels=sim.core.render_view(0);goal_pixels=sim.core.goals[0].copy();z0,zg=engine.encode([pixels,goal_pixels]);z0=z0[None]
            pg=engine.decode(zg[None])[0];p0=engine.decode(z0)[0]
            cp=CoopPolicy(np.random.default_rng([seed,t,91]),noise=0,smooth=0,p_solo=0);cp.reset(sim.core,0);pair=cp._choose(sim.core)
            if pair is None:targets=np.repeat(np.array(sim.core.block.position)[None],2,axis=0)
            else:
                pts,norm=cp.contacts;targets=np.array([sim.core.block.local_to_world(tuple(pts[j]-norm[j]*30)) for j in pair])
            d=targets-np.array([b.position for b in sim.core.agents]);d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-9)
            a,trace=engine.pools(z0,pg,seed,t,d);assert a.shape==(64,10,2,2) and abs(a).max()<=1.000001
            zp=engine.predict(z0,a);pp=engine.decode(zp[:,-1]);high,pc,lc=planning_features(zp,zg,a,engine.probe)
            features=low_features(pp,pg,p0,a);paths=[];ct=[];images=[]
            for candidate in a:
                restore();path=[sim.state().copy()];touch=[]
                for action in candidate:sim.core.simulate(action);path.append(sim.state().copy());touch.append(sim.core.block_contact.copy())
                paths.append(path);ct.append(touch);images.append(sim.core.render_view(0))
            paths=np.array(paths);ct=np.array(ct);truepose=pose(paths[:,-1]);y=physical_cost(truepose,pose(goal));ty=tcost(truepose,pose(goal))
            realz=engine.encode(images);realpose=engine.decode(realz)
            for i,candidate in enumerate(a):
                restore();check=[sim.state().copy()]
                for action in candidate:sim.core.simulate(action);check.append(sim.state().copy())
                e=float(abs(np.array(check)-paths[i]).max());maxreplay=max(maxreplay,e);assert e==0
            group=folder/f'scene{scene:02d}_t{t:03d}';group.mkdir()
            np.savez_compressed(group/'inputs.npz',features=features,features821=high,actions=a,pixels=pixels,goal_pixels=goal_pixels,
                predicted_latents=zp.cpu().numpy(),goal_latent=zg.cpu().numpy(),predicted_pose=pp,initial_predicted_pose=p0,decoded_goal=pg,
                predicted_cost=pc,predicted_T_cost=tcost(pp,pg),latent_cost=lc,proposal_id=np.repeat(np.arange(4),16))
            np.savez_compressed(group/'labels.npz',state_path=paths,contact=ct,true_cost=y,true_T_cost=ty,true_pose=truepose,
                real_encoded_latents=realz.cpu().numpy(),real_encoded_pose=realpose,real_encoded_cost=physical_cost(realpose,pg),
                initial_state=states[t],goal=goal,true_cost_start=physical_cost(pose(states[t]),pose(goal)),true_T_cost_start=tcost(pose(states[t]),pose(goal)),
                prefix_actions=actions[:t],direction=d)
            np.savez_compressed(group/'cem_intermediate_UNLABELED.npz',actions=np.array([q['actions'] for q in trace[:2]]),predicted_latents=np.array([q['latents'] for q in trace[:2]]),predicted_cost=np.array([q['score'] for q in trace[:2]]))
            rec['groups'].append(dict(name=group.name,t=t,input_sha256=sha(group/'inputs.npz'),labels_sha256=sha(group/'labels.npz')))
        env.close();sim.close();records.append(rec);write(folder/'manifest.json',records)
        print('SCENE',args.split,scene+1,'/',len(seeds),flush=True)
    assert all(sha(Path(p))==h for p,h in protocol['checkpoint_hashes'].items())
    meta.update(status='passed',scenes=len(records),groups=2*len(records),labeled_candidates=128*len(records),
        max_branch_error=maxbranch,max_second_replay_error=maxreplay,wall_seconds=time.perf_counter()-start,
        actual_learned_scores=224*len(records),unlabeled_intermediate_predictions=64*len(records),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
except BaseException as e:meta.update(status='failed',error=repr(e),wall_seconds=time.perf_counter()-start);raise
finally:
    write(folder/'metadata.json',meta);shutil.copy2(Path(__file__),ROOT/'scripts/phase22_v1'/Path(__file__).name)
print('COLLECTION_COMPLETE',args.split,meta['status'],flush=True)
