"""Independent NumPy metric/forward/choice audit and fresh oracle verification."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import conflict_ranking as cr
import repair_safety as repair
import state_action_ranker as sr
import verify_safety_repair as independent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--repair',type=Path,required=True);a=ap.parse_args();root=a.run
    torch.set_num_threads(1);protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text());rows=json.loads((root/'episode_results.json').read_text());assert len(rows)==60
    weights={};frozen=json.loads((a.repair/'selection_frozen.json').read_text())
    for arm,path in [('original',a.baseline/'portable_scorer.npz'),('raw_repair',a.repair/'raw_portable.npz'),('invariant_repair',a.repair/'invariant_portable.npz')]:
        with np.load(path) as z:weights[arm]={k:z[k] for k in z.files}
    paths=dict(original=a.baseline/'state_action_scorer.pt',raw_repair=a.repair/'raw.pt',invariant_repair=a.repair/'invariant.pt')
    for arm,p in paths.items():assert sr.sha(p)==protocol['model_sha256'][arm]
    maxforward=0.;oracleerror=0.;steps_checked=0;modeldecisions=0;oracledecisions=0;hashes={};initials={}
    for row in rows:
        geo=next(g for g in protocol['geometry_specs'] if g['name']==row['geometry']);arm=row['arm'];p=root/'episodes'/f"{row['geometry']}_{arm}_{row['seed']}"/'trajectory.npz'
        assert sr.sha(p)==row['trajectory_sha256'];hashes[str(p.relative_to(root))]=sr.sha(p)
        with np.load(p) as z:
            s=z['state'];sub=z['substep_state'];actions=z['action'];goals=z['goals'];n=len(actions);assert n==row['steps'] and n<=200
            np.testing.assert_array_equal(sub[1::2],s[1:]);np.testing.assert_array_equal(s[0],np.concatenate([geo['starts'],geo['velocity']],1).astype(np.float32))
            assert abs(actions).max()<=1.000001 and np.isfinite(s).all() and len(sub)==2*n
            if row['geometry'] in initials:np.testing.assert_array_equal(initials[row['geometry']],s[0])
            else:initials[row['geometry']]=s[0].copy()
            allsub=np.concatenate([s[:1],sub]);sep=np.linalg.norm(allsub[:,0,:2]-allsub[:,1,:2],axis=-1)
            success=bool((np.linalg.norm(s[-1,:,:2]-goals,axis=-1)<.12).all());unsafe=bool((sep<.3).any());collision=bool((sep<.2).any())
            assert success==row['joint_success'] and unsafe==row['substep_loss_of_separation'] and collision==row['substep_collision_proxy']
            assert (success and not unsafe)==row['safe_joint_success'];assert abs(sep.min()-row['substep_minimum_separation'])<1e-7
            assert row['state_replay_error']==row['substep_replay_error']==0
            assert abs(z['latency_ms'].mean()-row['mean_planning_ms'])<1e-9
            # Both termination and budget are checked, without imposing a speed criterion.
            assert success or n==200
            if arm=='pd':
                expected=np.clip(1.5*(goals[None]-s[:-1,:,:2])-s[:-1,:,2:],-1,1)
                np.testing.assert_array_equal(actions,expected);assert (z['selected']==-1).all()
            else:
                pools=z['candidates'];raws=z['prediction'];assert pools.shape==(n,72,8,2,2)
                np.testing.assert_array_equal(actions,pools[np.arange(n),z['selected'],0])
                ns=cr.namespace('S1_symmetric',goals) if arm=='true_sim_safety' else None
                for t in range(n):
                    pool=pools[t];idx=int(z['selected'][t]);current=s[t]
                    if arm=='true_sim_safety':
                        at=torch.tensor(pool,device='cuda');path,subs=repair.rollout(ns,current,at);path=path.cpu().numpy();subs=subs.cpu().numpy()
                        ds=np.linalg.norm(path[:,1:,0,:2]-path[:,1:,1,:2],axis=-1)
                        cost=5*np.linalg.norm(path[:,-1,:,:2]-goals,axis=-1).sum(-1)+100*(np.maximum(.3-ds,0)**2).sum(-1)+.01*(pool*pool).mean((1,2,3))
                        ss=np.concatenate([path[:,:1],subs],1);sd=np.linalg.norm(ss[:,:,0,:2]-ss[:,:,1,:2],axis=-1);minimum=sd.min(-1);exposure=(np.maximum(.3-sd,0)**2).sum(-1)
                        oracleerror=max(oracleerror,float(abs(cost-z['oracle_cost'][t]).max()));np.testing.assert_allclose(cost,z['oracle_cost'][t],atol=1e-5,rtol=1e-5)
                        np.testing.assert_allclose(minimum,z['oracle_minimum'][t],atol=1e-7,rtol=0)
                        # Use stored cost after independently checking it, to avoid float rounding at ties.
                        c=z['oracle_cost'][t];best=min(range(72),key=lambda j:(minimum[j]<.3,0 if minimum[j]>=.3 else -minimum[j],0 if minimum[j]>=.3 else exposure[j],c[j],j))
                        assert idx==best;oracledecisions+=1;continue
                    w=weights[arm];x=independent.canonical(current,goals,pool) if arm=='invariant_repair' else independent.raw_features(current,goals,pool)[:,None]
                    raw=independent.infer(x,w);maxforward=max(maxforward,float(abs(raw-raws[t]).max()));np.testing.assert_allclose(raw,raws[t],atol=5e-5,rtol=2e-5)
                    raw=raws[t];risk=1/(1+np.exp(-np.clip(raw[:,0],-50,50)));cost=raw[:,1]*w['cost_std']+w['cost_mean'];minimum=raw[:,2]*.3
                    if arm=='original':threshold=float(w['threshold']);margin=0
                    else:policy=frozen['models']['invariant' if arm=='invariant_repair' else 'raw']['policy'];threshold=policy['threshold'];margin=policy['margin']
                    accepted=[j for j in range(72) if risk[j]<threshold and (arm=='original' or minimum[j]>=.3+margin)]
                    if accepted:best=min(accepted,key=lambda j:(cost[j],j))
                    elif arm=='original':best=min(range(72),key=lambda j:(-minimum[j],cost[j],j))
                    else:best=min(range(72),key=lambda j:(risk[j],-minimum[j],cost[j],j))
                    assert best==idx and bool(z['fallback'][t])==(not bool(accepted));modeldecisions+=1
            steps_checked+=n
        print('VERIFIED',row['geometry'],arm,row['seed'],flush=True)
    for arm,stats in summary['methods'].items():
        for key,expected in stats['mean'].items():
            means=[]
            for geo in protocol['geometry_specs']:
                g=[r for r in rows if r['arm']==arm and r['geometry']==geo['name']];assert len(g)==2
                means.append(float(np.mean([r[key] for r in g])))
            assert abs(float(np.mean(means))-expected)<1e-12
    old=summary['methods']['original']['mean'];new=summary['methods']['invariant_repair']['mean']
    assert summary['gate_passed']==bool(new['safe_joint_success']>old['safe_joint_success'] and new['substep_loss_of_separation']<=old['substep_loss_of_separation'])
    sr.write(root/'trajectory_sha256.json',hashes);sr.write(root/'verification.json',dict(status='passed',episodes=60,geometries=6,executed_steps=steps_checked,
        independent_model_decisions=modeldecisions,independent_native_oracle_decisions=oracledecisions,max_numpy_forward_error=maxforward,
        max_oracle_cost_formula_error=oracleerror,first_action_execution_exact=True,metrics_and_aggregate_verified=True,model_hashes_unchanged=True,native_replay_error=0))
    print((root/'verification.json').read_text())
if __name__=='__main__':main()
