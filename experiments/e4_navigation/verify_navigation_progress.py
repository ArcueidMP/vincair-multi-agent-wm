"""Independent NumPy counterfactual audit on all new development snapshots."""
import argparse,json
from pathlib import Path
import numpy as np
import state_action_ranker as sr
import verify_safety_repair as independent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--repair',type=Path,required=True);a=ap.parse_args();root=a.run
    protocol=json.loads((root/'protocol.json').read_text());records=json.loads((root/'records.json').read_text());summary=json.loads((root/'summary.json').read_text());frozen=json.loads((a.repair/'selection_frozen.json').read_text())
    assert sr.sha(a.repair/'invariant.pt')==protocol['model_sha256']['invariant_repair']
    with np.load(a.repair/'invariant_portable.npz') as z:w={k:z[k] for k in z.files}
    policy=frozen['models']['invariant']['policy'];maxcost=0.;maxraw=0.;decisions=0;hashes={};reconstructed=[]
    for r in records:
        p=root/'snapshots'/r['name']
        for file,key in [('inputs.npz','input_sha256'),('labels.npz','label_sha256'),('prediction.npz','prediction_sha256')]:
            assert sr.sha(p/file)==r[key];hashes[str((p/file).relative_to(root))]=r[key]
        with np.load(p/'inputs.npz') as z,np.load(p/'labels.npz') as lab,np.load(p/'prediction.npz') as pred:
            s,g,actions=z['initial_state'],z['goals'],z['actions'];path=lab['state_path'];sub=lab['substep_state_path'];raw=pred['raw'];cost=lab['true_cost'];minimum=lab['min_separation'];exposure=lab['exposure']
            assert actions.shape==(72,8,2,2);np.testing.assert_array_equal(sub[:,1::2],path[:,1:]);np.testing.assert_array_equal(path[:,0],np.repeat(s[None],72,0))
            ds=np.linalg.norm(path[:,1:,0,:2]-path[:,1:,1,:2],axis=-1)
            true=5*np.linalg.norm(path[:,-1,:,:2]-g,axis=-1).sum(-1)+100*(np.maximum(.3-ds,0)**2).sum(-1)+.01*(actions*actions).mean((1,2,3))
            maxcost=max(maxcost,float(abs(true-cost).max()));np.testing.assert_allclose(true,cost,atol=1e-5,rtol=1e-5)
            allsub=np.concatenate([path[:,:1],sub],1);sep=np.linalg.norm(allsub[:,:,0,:2]-allsub[:,:,1,:2],axis=-1)
            np.testing.assert_allclose(sep.min(-1),minimum,atol=1e-7,rtol=0);np.testing.assert_allclose((np.maximum(.3-sep,0)**2).sum(-1),exposure,atol=1e-7,rtol=1e-5)
            raw2=independent.infer(independent.canonical(s,g,actions),w);maxraw=max(maxraw,float(abs(raw-raw2).max()));np.testing.assert_allclose(raw,raw2,atol=5e-5,rtol=2e-5)
            # The producer uses Python scalar checkpoint statistics with float32 heads.
            risk=1/(1+np.exp(-np.clip(raw[:,0],-50,50)));pc=raw[:,1]*float(w['cost_std'])+float(w['cost_mean']);ps=raw[:,2]*.3
            accepted=(risk<policy['threshold'])&(ps>=.3+policy['margin']);safe=minimum>=.3
            progress=np.linalg.norm(s[:,:2]-g,axis=-1).sum()-np.linalg.norm(path[:,-1,:,:2]-g,axis=-1).sum(-1);good=safe&(progress>.05)
            valid=np.flatnonzero(accepted);selected=min(valid,key=lambda j:(pc[j],j)) if len(valid) else min(range(72),key=lambda j:(risk[j],-ps[j],pc[j],j))
            def choose(mask,values):
                idx=np.flatnonzero(mask);return min(idx,key=lambda j:(values[j],j)) if len(idx) else selected
            choices=dict(learned=selected,true_cost_predicted_filter=choose(accepted,cost),true_safety_predicted_cost=choose(safe,pc),
                true_safety_oracle=min(range(72),key=lambda j:(not safe[j],0 if safe[j] else -minimum[j],0 if safe[j] else exposure[j],cost[j],j)),safe_progress_oracle=choose(safe,-progress))
            flags=dict(safe_available=bool(safe.any()),safe_progress_available=bool(good.any()),filter_excludes_all_progress=bool(good.any() and not (good&accepted).any()),
                cost_miss_within_filter=bool((good&accepted).any() and not good[selected]),fallback=not bool(len(valid)),cost_head_rmse=float(np.sqrt(np.mean((pc-cost)**2))))
            for key,value in flags.items():assert value==r[key],(r['name'],key,value,r[key])
            for method,idx in choices.items():
                m=r['methods'][method];assert m['index']==idx and m['unsafe']==bool(not safe[idx]) and m['safe_progress']==bool(good[idx])
                assert abs(m['progress']-float(progress[idx]))<1e-6 and abs(m['soft_regret']-float(cost[idx]-cost.min()))<1e-6;decisions+=1
            # Snapshot must match its separately hashed parent, without future features.
            parent=root/'parents'/f"{r['geometry']}_{r['origin']}"/'trajectory.npz'
            with np.load(parent) as z:
                np.testing.assert_array_equal(s,z['state'][r['step']]);np.testing.assert_array_equal(actions,z['candidates'][r['step']])
            reconstructed.append(r)
    episodes=json.loads((root/'episode_results.json').read_text());assert len(episodes)==24
    for r in episodes:
        p=root/'parents'/f"{r['geometry']}_{r['arm']}"/'trajectory.npz';assert sr.sha(p)==r['trajectory_sha256'];hashes[str(p.relative_to(root))]=sr.sha(p)
        assert r['state_replay_error']==r['substep_replay_error']==0
        with np.load(p) as z:
            goal=bool((np.linalg.norm(z['state'][-1,:,:2]-z['goals'],axis=-1)<.12).all());ss=np.concatenate([z['state'][:1],z['substep_state']],0);unsafe=bool((np.linalg.norm(ss[:,0,:2]-ss[:,1,:2],axis=-1)<.3).any())
            assert goal==r['joint_success'] and (goal and not unsafe)==r['safe_joint_success']
    for group,stats in summary['by_split_origin'].items():
        split,origin=group.split('/');rr=[r for r in records if r['split']==split and r['origin']==origin]
        assert len(rr)==stats['snapshots']
        def mean_geometry(value):return float(np.mean([np.mean([value(r) for r in rr if r['geometry']==gid]) for gid in sorted({r['geometry'] for r in rr})]))
        for key,value in stats['mean'].items():assert abs(mean_geometry(lambda r:r[key])-value)<1e-12
        for method,values in stats['methods'].items():
            for key,value in values.items():assert abs(mean_geometry(lambda r:r['methods'][method][key])-value)<1e-12
    assert summary['candidate_labels']==72*len(records) and summary['max_native_replay_error']==0
    sr.write(root/'dataset_sha256.json',hashes);sr.write(root/'verification.json',dict(status='passed',snapshots=len(records),candidate_labels=len(records)*72,parent_episodes=24,
        independent_counterfactual_choices=decisions,max_cost_formula_error=maxcost,max_numpy_forward_error=maxraw,parent_snapshot_matching_exact=True,
        split_origin_geometry_aggregation_verified=True,native_double_replay_error=0,no_training=True))
    print((root/'verification.json').read_text())
if __name__=='__main__':main()
