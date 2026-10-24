"""Independent labels/choices audit plus paired initial-pool failure diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import shared_navigation_model as shared
import repair_safety as repair
import conflict_ranking as cr
import state_action_ranker as sr
import verify_safety_repair as independent
import verify_navigation_cost as free

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();root=a.run
    torch.set_num_threads(1);protocol=json.loads((root/'protocol.json').read_text());rows=json.loads((root/'episode_results.json').read_text());summary=json.loads((root/'summary.json').read_text());assert len(rows)==36
    _,obj=shared.load();assert sr.sha(shared.DEFAULT_MODEL)==protocol['portable_model_sha256'];net={k:np.array(v,np.float32) for k,v in obj['arrays'].items()};net.update(cost_mean=obj['cost_mean'],cost_std=obj['cost_std'])
    cm,cs=obj['cost_mean'],obj['cost_std'];policy=obj['policy'];maxforward=0.;maxcost=0.;decisions=0;oracledecisions=0;steps=0;hashes={};initial={};failures=[]
    for row in rows:
        file=root/'episodes'/f"{row['geometry']}_{row['arm']}_{row['seed']}"/'trajectory.npz';assert sr.sha(file)==row['trajectory_sha256'];hashes[str(file.relative_to(root))]=sr.sha(file)
        geo=next(g for g in protocol['geometry_specs'] if g['name']==row['geometry'])
        with np.load(file) as z:
            s=z['state'];sub=z['substep_state'];actions=z['action'];goals=z['goals'];pool=z['candidates'];raw=z['prediction'];n=len(actions);assert n==row['steps']<=200
            np.testing.assert_array_equal(s[0],np.concatenate([geo['starts'],geo['velocity']],1).astype(np.float32));np.testing.assert_array_equal(sub[1::2],s[1:]);np.testing.assert_array_equal(actions,pool[np.arange(n),z['selected'],0]);assert abs(actions).max()<=1.000001
            if geo['name'] in initial:np.testing.assert_array_equal(initial[geo['name']],pool[0])
            else:initial[geo['name']]=pool[0].copy()
            ss=np.concatenate([s[:1],sub]);ds=np.linalg.norm(ss[:,0,:2]-ss[:,1,:2],axis=-1);success=bool((np.linalg.norm(s[-1,:,:2]-goals,axis=-1)<.12).all())
            assert success==row['joint_success'] and bool((ds<.3).any())==row['substep_loss_of_separation'] and bool((ds<.2).any())==row['substep_collision_proxy'];assert (success and bool((ds>=.3).all()))==row['safe_joint_success'];assert success or n==200
            assert row['state_replay_error']==row['substep_replay_error']==0
            ns=cr.namespace('S1_symmetric',goals) if row['arm']=='true_sim_safety' else None
            for t in range(n):
                idx=int(z['selected'][t])
                if ns is not None:
                    path,subs=repair.rollout(ns,s[t],torch.tensor(pool[t],device='cuda'));path=path.cpu().numpy();subs=subs.cpu().numpy();d=np.linalg.norm(path[:,1:,0,:2]-path[:,1:,1,:2],axis=-1)
                    truth=5*np.linalg.norm(path[:,-1,:,:2]-goals,axis=-1).sum(-1)+100*(np.maximum(.3-d,0)**2).sum(-1)+.01*(pool[t]**2).mean((1,2,3));maxcost=max(maxcost,float(abs(truth-z['oracle_cost'][t]).max()));np.testing.assert_allclose(truth,z['oracle_cost'][t],atol=1e-5,rtol=1e-5)
                    ss2=np.concatenate([path[:,:1],subs],1);d=np.linalg.norm(ss2[:,:,0,:2]-ss2[:,:,1,:2],axis=-1);minimum=d.min(-1);exposure=(np.maximum(.3-d,0)**2).sum(-1);np.testing.assert_allclose(minimum,z['oracle_minimum'][t],atol=1e-7,rtol=0)
                    best=min(range(72),key=lambda j:(minimum[j]<.3,0 if minimum[j]>=.3 else -minimum[j],0 if minimum[j]>=.3 else exposure[j],z['oracle_cost'][t,j],j));assert idx==best;oracledecisions+=1;continue
                expected=independent.infer(independent.canonical(s[t],goals,pool[t]),net)
                if row['arm']=='analytic_goal':expected[:,1]=(free.free_goal(s[t],goals,pool[t])[0]-cm)/cs
                maxforward=max(maxforward,float(abs(expected-raw[t]).max()));np.testing.assert_allclose(expected,raw[t],atol=6e-5,rtol=2e-5)
                r=raw[t];risk=1/(1+np.exp(-np.clip(r[:,0],-50,50)));c=r[:,1]*cs+cm;minimum=r[:,2]*.3;valid=np.flatnonzero((risk<policy['threshold'])&(minimum>=.3+policy['margin']))
                best=min(valid,key=lambda j:(c[j],j)) if len(valid) else min(range(72),key=lambda j:(risk[j],-minimum[j],c[j],j));assert idx==best and bool(z['fallback'][t])==(not bool(len(valid)));decisions+=1
            if row['arm']=='analytic_goal':
                oracle=root/'episodes'/f"{row['geometry']}_true_sim_safety_{row['seed']}"/'trajectory.npz'
                with np.load(oracle) as ref:
                    np.testing.assert_array_equal(pool[0],ref['candidates'][0]);mins=ref['oracle_minimum'][0];idx=int(z['selected'][0]);violation=np.flatnonzero(ds<.3)
                    failures.append(dict(geometry=row['geometry'],condition=row['condition'],initial_pool_safe_candidates=int((mins>=.3).sum()),selected_initial_index=idx,
                        selected_initial_H8_true_minimum=float(mins[idx]),selected_initial_predicted_risk=float(1/(1+np.exp(-raw[0,idx,0]))),selected_initial_predicted_minimum=float(raw[0,idx,2]*.3),
                        selected_initial_fallback=bool(z['fallback'][0]),episode_violation=bool(len(violation)),first_violation_execution_step=int((violation[0]-1)//2) if len(violation) else None,
                        interpretation='Post-hoc local H8 diagnostic; unsafe candidate rollout does not alone prove the executed first action must violate'))
            steps+=n
        print('VERIFIED',row['geometry'],row['arm'],flush=True)
    for condition,arms in summary['by_condition'].items():
        for arm,stats in arms.items():
            rr=[r for r in rows if r['condition']==condition and r['arm']==arm];assert len(rr)==6
            for key,value in stats['mean'].items():assert abs(float(np.mean([r[key] for r in rr]))-value)<1e-12
    gate=all(summary['by_condition'][c]['analytic_goal']['mean']['safe_joint_success']>=.95 and summary['by_condition'][c]['analytic_goal']['mean']['substep_loss_of_separation']<=summary['by_condition'][c]['true_sim_safety']['mean']['substep_loss_of_separation'] for c in protocol['conditions']);assert gate==summary['stress_gate_passed']
    sr.write(root/'initial_pool_diagnostics.json',failures);sr.write(root/'trajectory_sha256.json',hashes);sr.write(root/'verification.json',dict(status='passed',episodes=36,executed_steps=steps,independent_model_choices=decisions,independent_oracle_choices=oracledecisions,
        max_numpy_forward_error=maxforward,max_oracle_cost_error=maxcost,paired_initial_pools_exact=True,physical_outcomes_and_aggregates_verified=True,native_replay_error=0,portable_model_hash_unchanged=True,no_training=True))
    print((root/'verification.json').read_text())
if __name__=='__main__':main()
