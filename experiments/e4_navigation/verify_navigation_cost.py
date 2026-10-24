"""Audit train-only ridge and frozen closed-loop cost substitutions."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import repair_navigation_cost as m
import repair_safety as repair
import conflict_ranking as cr
import state_action_ranker as sr
import verify_safety_repair as independent

def free_goal(state,goals,actions):
    pos=np.broadcast_to(state[None,:,:2],(len(actions),2,2)).copy();vel=np.broadcast_to(state[None,:,2:],pos.shape).copy();minimum=np.linalg.norm(pos[:,0]-pos[:,1],axis=-1)
    for force in actions.transpose(1,0,2,3):
        vel=vel*np.float32(.75)
        for _ in range(2):
            vel=vel+force*np.float32(.05);pos=pos+vel*np.float32(.05);minimum=np.minimum(minimum,np.linalg.norm(pos[:,0]-pos[:,1],axis=-1))
    distance=np.linalg.norm(pos-goals,axis=-1).sum(-1);goal=5*distance+.01*(actions*actions).mean((1,2,3))
    return goal,minimum,np.linalg.norm(state[:,:2]-goals,axis=-1).sum()-distance

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--repair',type=Path,required=True);a=ap.parse_args();root=a.run
    torch.set_num_threads(1);protocol=json.loads((root/'protocol.json').read_text());freeze=json.loads((root/'selection_frozen.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert sr.sha(root/'protocol.json')==freeze['protocol_sha256'];assert sr.sha(root/'goal_correction.npz')==freeze['correction_sha256'];assert sr.sha(a.repair/'invariant.pt')==freeze['risk_sha256']
    with np.load(root/'goal_correction.npz') as z:w={k:z[k] for k in z.files}
    with np.load(a.repair/'invariant_portable.npz') as z:net={k:z[k] for k in z.files}
    policy=json.loads((a.repair/'selection_frozen.json').read_text())['models']['invariant']['policy'];cm=float(net['cost_mean']);cs=float(net['cost_std'])
    records=json.loads((a.data/'records.json').read_text());assert sr.sha(a.data/'records.json')==protocol['data_records_sha256']
    trainx=[];trainy=[];weights=[];maxfeature=0.;maxraw=0.;controls={};validation=[]
    for r in records:
        p=a.data/'snapshots'/r['name'];assert sr.sha(p/'inputs.npz')==r['input_sha256'] and sr.sha(p/'labels.npz')==r['label_sha256']
        with np.load(p/'inputs.npz') as z,np.load(p/'labels.npz') as lab,np.load(p/'prediction.npz') as pred:
            s,g,act=z['initial_state'],z['goals'],z['actions'];raw=pred['raw'];raw2=independent.infer(independent.canonical(s,g,act),net)
            maxraw=max(maxraw,float(abs(raw-raw2).max()));np.testing.assert_allclose(raw,raw2,atol=5e-5,rtol=2e-5)
            x=m.correction_features(s,g,act,raw,cm,cs);views=independent.canonical(s,g,act);goal,sep,prog=free_goal(s,g,act)
            independent_x=np.column_stack([views.mean(1),(views*views).mean(1),goal,sep,raw[:,1]*cs+cm,prog]);maxfeature=max(maxfeature,float(abs(x-independent_x).max()));np.testing.assert_allclose(x,independent_x,atol=2e-5,rtol=2e-5)
            target=5*np.linalg.norm(lab['state_path'][:,-1,:,:2]-g,axis=-1).sum(-1)+.01*(act*act).mean((1,2,3));residual=target-(raw[:,1]*cs+cm)
            if r['split']=='train':
                n=sum(q['split']=='train' and q['geometry']==r['geometry'] and q['origin']==r['origin'] for q in records)
                trainx.append(x);trainy.append(residual);weights.extend([1/(8*2*n*72)]*72)
            else:validation.append((r,x,raw,target,lab['min_separation'].copy(),np.linalg.norm(s[:,:2]-g,axis=-1).sum()-np.linalg.norm(lab['state_path'][:,-1,:,:2]-g,axis=-1).sum(-1),goal))
    X=np.concatenate(trainx);Y=np.concatenate(trainy);wt=np.array(weights);mean=(X*wt[:,None]).sum(0);std=np.maximum(np.sqrt(((X-mean)**2*wt[:,None]).sum(0)),.01);intercept=float((Y*wt).sum())
    np.testing.assert_allclose(mean,w['mean'],atol=1e-10,rtol=1e-10);np.testing.assert_allclose(std,w['std'],atol=1e-10,rtol=1e-10);assert abs(intercept-float(w['intercept']))<1e-10
    X=(X-mean)/std;Y=Y-intercept;gradient=X.T@(wt*(X@w['coef']-Y))+float(w['lambda_value'])*w['coef'];assert abs(gradient).max()<1e-9
    # Rebuild full validation selection grid using the independently checked features.
    vr=[r[0] for r in validation];vx=np.stack([r[1] for r in validation]);vraw=np.stack([r[2] for r in validation]);target=np.stack([r[3] for r in validation]);minimum=np.stack([r[4] for r in validation]);progress=np.stack([r[5] for r in validation]);goal=np.stack([r[6] for r in validation]);c=dict(cost_mean=cm,cost_std=cs,policy=policy)
    def val_stats(raw):
        selected=repair.choose(raw,cm,cs,**policy)[0];per=[]
        for gid in sorted({r['geometry'] for r in vr}):
            rr=[]
            for origin in ['invariant_repair','true_sim_safety']:
                inds=[i for i,r in enumerate(vr) if r['geometry']==gid and r['origin']==origin]
                rr.append([np.mean([minimum[i,selected[i]]<.3 for i in inds]),np.mean([minimum[i,selected[i]]>=.3 and progress[i,selected[i]]>.05 for i in inds]),np.mean([float(target[i,selected[i]]-target[i].min()) for i in inds])])
            per.append(np.mean(rr,axis=0))
        return dict(zip(['unsafe','safe_progress','goal_cost_regret'],map(float,np.mean(per,axis=0))))
    grid=json.loads((root/'validation_grid.json').read_text());best=None
    for entry in grid:
        lam=entry['lambda_value'];coef=np.linalg.solve(X.T@(wt[:,None]*X)+lam*np.eye(92),X.T@(wt*Y));pred=vraw.copy();pred[...,1]+=(((vx-mean)/std)@coef+intercept)/cs;stats=val_stats(pred)
        for key,value in stats.items():assert abs(value-entry[key])<1e-10,(lam,key,value,entry[key])
        key=(stats['unsafe'],-stats['safe_progress'],stats['goal_cost_regret'],lam)
        if best is None or key<best:best=key
    assert list(best)==freeze['validation_key'] and best[-1]==float(w['lambda_value'])
    pred=vraw.copy();pred[...,1]=(goal-cm)/cs;controls['analytic_goal']=val_stats(pred);controls['invariant_repair']=val_stats(vraw);sr.write(root/'validation_controls.json',controls)
    rows=json.loads((root/'episode_results.json').read_text());assert len(rows)==48;decisions=0;oracledecisions=0;totalsteps=0;hashes={};maxcosterror=0.
    for row in rows:
        p=root/'episodes'/f"{row['geometry']}_{row['arm']}_{row['seed']}"/'trajectory.npz';assert sr.sha(p)==row['trajectory_sha256'];hashes[str(p.relative_to(root))]=sr.sha(p)
        with np.load(p) as z:
            states=z['state'];sub=z['substep_state'];actions=z['action'];goals=z['goals'];pool=z['candidates'];raw=z['prediction'];n=len(actions);assert n==row['steps']<=200
            np.testing.assert_array_equal(sub[1::2],states[1:]);np.testing.assert_array_equal(actions,pool[np.arange(n),z['selected'],0]);assert row['state_replay_error']==row['substep_replay_error']==0
            ss=np.concatenate([states[:1],sub]);ds=np.linalg.norm(ss[:,0,:2]-ss[:,1,:2],axis=-1);success=bool((np.linalg.norm(states[-1,:,:2]-goals,axis=-1)<.12).all())
            assert success==row['joint_success'] and bool((ds<.3).any())==row['substep_loss_of_separation'] and bool((ds<.2).any())==row['substep_collision_proxy'];assert (success and (ds>=.3).all())==row['safe_joint_success'];assert success or n==200
            ns=cr.namespace('S1_symmetric',goals) if row['arm']=='true_sim_safety' else None
            for t in range(n):
                idx=int(z['selected'][t])
                if ns is not None:
                    path,subs=repair.rollout(ns,states[t],torch.tensor(pool[t],device='cuda'));path=path.cpu().numpy();subs=subs.cpu().numpy()
                    dist=np.linalg.norm(path[:,1:,0,:2]-path[:,1:,1,:2],axis=-1);cost=5*np.linalg.norm(path[:,-1,:,:2]-goals,axis=-1).sum(-1)+100*(np.maximum(.3-dist,0)**2).sum(-1)+.01*(pool[t]**2).mean((1,2,3))
                    maxcosterror=max(maxcosterror,float(abs(cost-z['oracle_cost'][t]).max()));np.testing.assert_allclose(cost,z['oracle_cost'][t],atol=1e-5,rtol=1e-5)
                    ss=np.concatenate([path[:,:1],subs],1);ds=np.linalg.norm(ss[:,:,0,:2]-ss[:,:,1,:2],axis=-1);sep=ds.min(-1);exposure=(np.maximum(.3-ds,0)**2).sum(-1);np.testing.assert_allclose(sep,z['oracle_minimum'][t],atol=1e-7,rtol=0)
                    truecost=z['oracle_cost'][t];best=min(range(72),key=lambda j:(sep[j]<.3,0 if sep[j]>=.3 else -sep[j],0 if sep[j]>=.3 else exposure[j],truecost[j],j));assert idx==best;oracledecisions+=1;continue
                frozenraw=independent.infer(independent.canonical(states[t],goals,pool[t]),net);expected=frozenraw.copy()
                if row['arm']=='analytic_goal':predicted=free_goal(states[t],goals,pool[t])[0];expected[:,1]=(predicted-cm)/cs
                elif row['arm']=='ridge_goal':
                    features=m.correction_features(states[t],goals,pool[t],frozenraw,cm,cs);predicted=frozenraw[:,1]*cs+cm+((features-w['mean'])/w['std'])@w['coef']+w['intercept'];expected[:,1]=(predicted-cm)/cs
                maxraw=max(maxraw,float(abs(expected-raw[t]).max()));np.testing.assert_allclose(expected,raw[t],atol=2e-4,rtol=3e-5)
                # Selection reconstructed from audited stored heads to preserve float tie behavior.
                r=raw[t];risk=1/(1+np.exp(-np.clip(r[:,0],-50,50)));cost=r[:,1]*cs+cm;sep=r[:,2]*.3;accepted=np.flatnonzero((risk<policy['threshold'])&(sep>=.3+policy['margin']))
                best=min(accepted,key=lambda j:(cost[j],j)) if len(accepted) else min(range(72),key=lambda j:(risk[j],-sep[j],cost[j],j))
                assert idx==best and bool(z['fallback'][t])==(not bool(len(accepted)));decisions+=1
            totalsteps+=n
        print('VERIFIED',row['geometry'],row['arm'],row['seed'],flush=True)
    for arm,stats in summary['methods'].items():
        for key,value in stats['mean'].items():
            actual=np.mean([np.mean([r[key] for r in rows if r['arm']==arm and r['geometry']==g['name']]) for g in protocol['geometry_specs']]);assert abs(actual-value)<1e-12
    old=summary['methods']['invariant_repair']['mean'];new=summary['methods']['ridge_goal']['mean'];assert summary['gate_passed']==bool(new['safe_joint_success']>old['safe_joint_success'] and new['substep_loss_of_separation']<=old['substep_loss_of_separation'])
    sr.write(root/'trajectory_sha256.json',hashes);sr.write(root/'verification.json',dict(status='passed',episodes=48,executed_steps=totalsteps,independent_model_choices=decisions,independent_oracle_choices=oracledecisions,
        train_only_weighted_moments_exact=True,ridge_normal_equation_max_error=float(abs(gradient).max()),validation_grid_and_selection_verified=True,max_feature_formula_difference=maxfeature,max_numpy_prediction_difference=maxraw,max_oracle_cost_error=maxcosterror,
        frozen_risk_heads_verified=True,first_action_execution_exact=True,physical_metrics_and_aggregates_verified=True,native_replay_error=0))
    print((root/'verification.json').read_text())
if __name__=='__main__':main()
