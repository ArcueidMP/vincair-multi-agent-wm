"""Independent numerical audit plus portable scorer equality and smoke replay."""
import argparse,json,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
import shared_navigation_model as shared
import repair_navigation_cost as cost
import repair_safety as repair
import conflict_ranking as cr
import state_action_ranker as sr
import verify_safety_repair as independent
import verify_navigation_cost as free

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--repair',type=Path,required=True);ap.add_argument('--portable-smoke',type=Path,required=True);a=ap.parse_args();root=a.run
    torch.set_num_threads(1);p=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text());rows=json.loads((root/'episode_results.json').read_text());assert len(rows)==72
    models,obj=shared.load();original=torch.load(a.repair/'invariant.pt',map_location='cpu',weights_only=False)
    assert sr.sha(a.repair/'invariant.pt')==obj['source_checkpoint_sha256']==p['model_sha256']['invariant_repair']
    for k,v in original['state_dict'].items():assert torch.equal(v,models['invariant_repair']['net'].state_dict()[k])
    np.testing.assert_array_equal(original['mean'],models['invariant_repair']['mean']);np.testing.assert_array_equal(original['std'],models['invariant_repair']['std'])
    assert obj['policy']==original['policy'];assert obj['cost_mean']==original['cost_mean'] and obj['cost_std']==original['cost_std']
    for arm in p['arms']:
        parent=root/'smoke'/f'smoke_{arm}_2026169999'/'trajectory.npz';portable=a.portable_smoke/f'portable_smoke_{arm}_2026169999'/'trajectory.npz'
        with np.load(parent) as x,np.load(portable) as y:
            for key in ['state','substep_state','action','candidates','prediction','selected','fallback']:np.testing.assert_array_equal(x[key],y[key])
    with np.load(a.repair/'invariant_portable.npz') as z:net={k:z[k] for k in z.files}
    cm=float(net['cost_mean']);cs=float(net['cost_std']);policy=obj['policy'];maxforward=0.;maxcost=0.;decisions=0;oracledecisions=0;steps=0;hashes={}
    for row in rows:
        f=root/'episodes'/f"{row['geometry']}_{row['arm']}_{row['seed']}"/'trajectory.npz';assert sr.sha(f)==row['trajectory_sha256'];hashes[str(f.relative_to(root))]=sr.sha(f)
        geo=next(g for g in p['geometry_specs'] if g['name']==row['geometry']);assert row['base_id']==geo['base_id'] and row['condition']==geo['condition']
        with np.load(f) as z:
            s=z['state'];sub=z['substep_state'];actions=z['action'];goals=z['goals'];pool=z['candidates'];raw=z['prediction'];n=len(actions);assert n==row['steps']<=200
            np.testing.assert_array_equal(s[0],np.concatenate([geo['starts'],geo['velocity']],1).astype(np.float32));np.testing.assert_array_equal(sub[1::2],s[1:]);np.testing.assert_array_equal(actions,pool[np.arange(n),z['selected'],0]);assert abs(actions).max()<=1.000001
            allsub=np.concatenate([s[:1],sub]);ds=np.linalg.norm(allsub[:,0,:2]-allsub[:,1,:2],axis=-1);success=bool((np.linalg.norm(s[-1,:,:2]-goals,axis=-1)<.12).all())
            assert success==row['joint_success'] and bool((ds<.3).any())==row['substep_loss_of_separation'] and bool((ds<.2).any())==row['substep_collision_proxy'];assert (success and bool((ds>=.3).all()))==row['safe_joint_success'];assert success or n==200
            assert row['state_replay_error']==row['substep_replay_error']==0
            ns=cr.namespace('S1_symmetric',goals) if row['arm']=='true_sim_safety' else None
            for t in range(n):
                idx=int(z['selected'][t])
                if ns is not None:
                    path,subs=repair.rollout(ns,s[t],torch.tensor(pool[t],device='cuda'));path=path.cpu().numpy();subs=subs.cpu().numpy();sep=np.linalg.norm(path[:,1:,0,:2]-path[:,1:,1,:2],axis=-1)
                    c=5*np.linalg.norm(path[:,-1,:,:2]-goals,axis=-1).sum(-1)+100*(np.maximum(.3-sep,0)**2).sum(-1)+.01*(pool[t]**2).mean((1,2,3));maxcost=max(maxcost,float(abs(c-z['oracle_cost'][t]).max()));np.testing.assert_allclose(c,z['oracle_cost'][t],atol=1e-5,rtol=1e-5)
                    ss=np.concatenate([path[:,:1],subs],1);d=np.linalg.norm(ss[:,:,0,:2]-ss[:,:,1,:2],axis=-1);minimum=d.min(-1);exposure=(np.maximum(.3-d,0)**2).sum(-1);np.testing.assert_allclose(minimum,z['oracle_minimum'][t],atol=1e-7,rtol=0)
                    best=min(range(72),key=lambda j:(minimum[j]<.3,0 if minimum[j]>=.3 else -minimum[j],0 if minimum[j]>=.3 else exposure[j],z['oracle_cost'][t,j],j));assert best==idx;oracledecisions+=1;continue
                expected=independent.infer(independent.canonical(s[t],goals,pool[t]),net)
                if row['arm']=='analytic_goal':expected[:,1]=(free.free_goal(s[t],goals,pool[t])[0]-cm)/cs
                maxforward=max(maxforward,float(abs(expected-raw[t]).max()));np.testing.assert_allclose(expected,raw[t],atol=6e-5,rtol=2e-5)
                r=raw[t];risk=1/(1+np.exp(-np.clip(r[:,0],-50,50)));c=r[:,1]*cs+cm;minimum=r[:,2]*.3;valid=np.flatnonzero((risk<policy['threshold'])&(minimum>=.3+policy['margin']))
                best=min(valid,key=lambda j:(c[j],j)) if len(valid) else min(range(72),key=lambda j:(risk[j],-minimum[j],c[j],j));assert best==idx and bool(z['fallback'][t])==(not bool(len(valid)));decisions+=1
            steps+=n
        print('VERIFIED',row['geometry'],row['arm'],flush=True)
    for cond,arms in summary['by_condition'].items():
        for arm,stats in arms.items():
            rr=[r for r in rows if r['condition']==cond and r['arm']==arm];assert len(rr)==6
            for key,value in stats['mean'].items():assert abs(float(np.mean([r[key] for r in rr]))-value)<1e-12
    gate=all(summary['by_condition'][c]['analytic_goal']['mean']['safe_joint_success']>=.95 and summary['by_condition'][c]['analytic_goal']['mean']['substep_loss_of_separation']<=summary['by_condition'][c]['true_sim_safety']['mean']['substep_loss_of_separation'] for c in p['conditions']);assert gate==summary['robustness_gate_passed']
    sr.write(root/'trajectory_sha256.json',hashes);sr.write(root/'verification.json',dict(status='passed',episodes=72,independent_base_geometries=6,executed_steps=steps,independent_model_choices=decisions,independent_oracle_choices=oracledecisions,
        max_numpy_forward_error=maxforward,max_oracle_cost_error=maxcost,physical_outcomes_and_condition_means_verified=True,native_replay_error=0,
        portable_parameters_exact=True,portable_three_arm_smoke_arrays_exact=True,portable_json_sha256=sr.sha(shared.DEFAULT_MODEL),no_training=True))
    sr.write(root/'tested_environment.json',dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','numpy','vmas','matplotlib']},platform='WSL2 Ubuntu / RTX4060 Laptop8GB',device='cuda'))
    print((root/'verification.json').read_text())
if __name__=='__main__':main()
