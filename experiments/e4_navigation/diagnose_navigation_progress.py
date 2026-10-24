"""New development trajectories for counterfactual scoring diagnostics; no fitting."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import torch
import closed_loop_pilot as pilot
import conflict_ranking as cr
import repair_safety as repair
import state_action_ranker as sr

SNAPSHOTS=[0,10,30,60,100,150,199]
METHODS=['learned','true_cost_predicted_filter','true_safety_predicted_cost','true_safety_oracle','safe_progress_oracle']
def geometries():
    rng=np.random.default_rng(2026104901);rows=[]
    for i in range(12):
        angle=float(rng.uniform(40,140));theta=float(rng.uniform(-np.pi,np.pi));center=rng.uniform(-.08,.08,2);distance=float(rng.uniform(.6,.9))
        d=np.array([[np.cos(theta),np.sin(theta)],[np.cos(theta+np.radians(angle)),np.sin(theta+np.radians(angle))]],np.float32)
        rows.append(dict(name=f'progress_{i:02}',split='train' if i<8 else 'validation',angle_deg=angle,directions=d.tolist(),
            starts=(center-distance*d).tolist(),goals=(center+.75*d).tolist(),velocity=(rng.uniform(.08,.25,(2,1))*d).tolist()))
    return rows

def analyze(initial,goals,raw,cost,minimum,exposure,terminal_positions,checkpoint):
    cm,cs=checkpoint['cost_mean'],checkpoint['cost_std'];policy=checkpoint['policy']
    selected,fallback=repair.choose(raw[None],cm,cs,**policy);selected=int(selected[0]);fallback=bool(fallback[0])
    predicted_cost=raw[:,1]*cs+cm;risk=1/(1+np.exp(-np.clip(raw[:,0],-50,50)));predicted_sep=raw[:,2]*.3
    accepted=(risk<policy['threshold'])&(predicted_sep>=.3+policy['margin']);safe=minimum>=.3
    initial_distance=np.linalg.norm(initial[:,:2]-goals,axis=-1).sum()
    progress=initial_distance-np.linalg.norm(terminal_positions-goals,axis=-1).sum(-1)
    good=safe&(progress>.05)
    def best(mask,values):
        inds=np.flatnonzero(mask);return int(inds[values[inds].argmin()]) if len(inds) else selected
    choices=dict(learned=selected,true_cost_predicted_filter=best(accepted,cost),true_safety_predicted_cost=best(safe,predicted_cost),
        true_safety_oracle=int(cr.preference_order(cost,minimum,exposure)[0]),safe_progress_oracle=best(safe,-progress))
    metrics={name:dict(index=idx,unsafe=bool(not safe[idx]),progress=float(progress[idx]),safe_progress=bool(good[idx]),soft_regret=float(cost[idx]-cost.min())) for name,idx in choices.items()}
    return dict(safe_available=bool(safe.any()),safe_progress_available=bool(good.any()),accepted_count=int(accepted.sum()),
        safe_progress_candidates=int(good.sum()),filter_excludes_all_progress=bool(good.any() and not (good&accepted).any()),
        cost_miss_within_filter=bool((good&accepted).any() and not good[selected]),fallback=fallback,
        cost_head_rmse=float(np.sqrt(np.mean((predicted_cost-cost)**2))),methods=metrics)

def aggregate(rows):
    groups=[]
    for gid in sorted({r['geometry'] for r in rows}):
        rr=[r for r in rows if r['geometry']==gid]
        groups.append(dict(geometry=gid,snapshots=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['safe_available','safe_progress_available','filter_excludes_all_progress','cost_miss_within_filter','fallback','cost_head_rmse']},
            methods={m:{k:float(np.mean([r['methods'][m][k] for r in rr])) for k in ['unsafe','progress','safe_progress','soft_regret']} for m in METHODS}))
    return dict(snapshots=len(rows),geometries=len(groups),scenes=groups,
        mean={k:float(np.mean([g[k] for g in groups])) for k in ['safe_available','safe_progress_available','filter_excludes_all_progress','cost_miss_within_filter','fallback','cost_head_rmse']},
        methods={m:{k:float(np.mean([g['methods'][m][k] for g in groups])) for k in ['unsafe','progress','safe_progress','soft_regret']} for m in METHODS})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--repair',type=Path,required=True);a=ap.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Preserve previous attempt')
    torch.set_num_threads(1);started=time.perf_counter();models,hashes=pilot.load_models(a.baseline,a.repair);model=models['invariant_repair'];geos=geometries()
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sr.sha(Path(__file__)),parent_commit='3cf1251890e60c70e2ed0ffb18013e4639be1305',
        geometry_specs=geos,split='8 new train/4 new validation whole geometries; development diagnostics, NOT confirmation',
        origins=['invariant_repair','true_sim_safety'],seed=2026140000,snapshot_steps=SNAPSHOTS,max_episode_steps=200,horizon=8,candidates=72,
        model_sha256=hashes,progress_definition='Sum current goal distances minus sum H8 terminal distances; positive threshold0.05 VMAS units',
        counterfactuals=METHODS,filter='Frozen risk and predicted-minsep conjunction. True-cost substitution retains original fallback when no accepted candidate.',
        sampling='Snapshot only if step was executed in its parent trajectory; equal snapshots within geometry, then equal geometries. Report origins separately.',
        native='Same native VMAS1.5.2/dt.1/substeps2/radius.1/drag.25; label safety includes initial state and native substeps',
        protocol_limitations=['One candidate seed per geometry','Counterfactual substitutions are local finite-H interventions, not new closed-loop controllers','Method substitutions need not add up to unique causal attribution','No old closed-loop confirmation/stress files read','No fitting or threshold search'],
        budget='24 bounded new parent episodes plus at most168 snapshot pools, each replayed twice; abort between geometries after900s',no_training=True)
    sr.write(out/'protocol.json',protocol);records=[];episodes=[];maxreplay=0.
    for gi,geo in enumerate(geos):
        if time.perf_counter()-started>900:raise RuntimeError('Budget exceeded; preserve partial outputs')
        ns=cr.namespace('S1_symmetric',geo['goals']);goals=np.array(geo['goals'],np.float32);seed=protocol['seed']+gi*1000
        for origin in protocol['origins']:
            result,z=pilot.episode(geo,origin,seed,models,200);p=out/'parents'/f"{geo['name']}_{origin}";p.mkdir(parents=True)
            np.savez_compressed(p/'trajectory.npz',**z);result['trajectory_sha256']=sr.sha(p/'trajectory.npz');result['split']=geo['split'];episodes.append(result);sr.write(p/'result.json',result)
            for step in SNAPSHOTS:
                if step>=len(z['action']):continue
                initial=z['state'][step];actions=z['candidates'][step];at=torch.tensor(actions,device='cuda')
                path,sub=repair.rollout(ns,initial,at);again,subagain=repair.rollout(ns,initial,at)
                error=max(float((path-again).abs().max()),float((sub-subagain).abs().max()));maxreplay=max(maxreplay,error);assert error==0
                lab=ns['calculate_ranking_labels'](path,at,goals);cost=lab['total_cost'];allsub=torch.cat([path[:,:1],sub],1)
                sep=torch.linalg.vector_norm(allsub[:,:,0,:2]-allsub[:,:,1,:2],dim=-1);minimum=sep.min(1).values.cpu().numpy();exposure=torch.relu(.3-sep).square().sum(1).cpu().numpy()
                _,_,raw=pilot.decide(model,initial,goals,actions);name=f"{geo['name']}_{origin}_t{step:03}";dest=out/'snapshots'/name;dest.mkdir(parents=True)
                np.savez_compressed(dest/'inputs.npz',initial_state=initial,goals=goals,actions=actions)
                np.savez_compressed(dest/'labels.npz',state_path=path.cpu().numpy(),substep_state_path=sub.cpu().numpy(),true_cost=cost,min_separation=minimum,exposure=exposure)
                np.savez_compressed(dest/'prediction.npz',raw=raw)
                record=dict(name=name,geometry=geo['name'],split=geo['split'],origin=origin,step=step,
                    input_sha256=sr.sha(dest/'inputs.npz'),label_sha256=sr.sha(dest/'labels.npz'),prediction_sha256=sr.sha(dest/'prediction.npz'),
                    **analyze(initial,goals,raw,cost,minimum,exposure,path[:,-1,:,:2].cpu().numpy(),model['checkpoint']))
                records.append(record)
        sr.write(out/'records.json',records);sr.write(out/'episode_results.json',episodes);print('GEOMETRY',gi+1,'/12',flush=True)
    _,after=pilot.load_models(a.baseline,a.repair);assert after==hashes
    summary=dict(by_split_origin={f'{split}/{origin}':aggregate([r for r in records if r['split']==split and r['origin']==origin]) for split in ['train','validation'] for origin in protocol['origins']},
        snapshot_count=len(records),candidate_labels=len(records)*72,parent_episodes=len(episodes),max_native_replay_error=maxreplay,model_hashes_unchanged=True,no_training=True,wall_seconds=time.perf_counter()-started)
    sr.write(out/'summary.json',summary);print('COMPLETE',json.dumps({k:v['mean'] for k,v in summary['by_split_origin'].items()}),flush=True)
if __name__=='__main__':main()
