"""Rebuild teammate conflict pilot and geometry-grouped safety ranking targets.

Only true VMAS is used. No learned-model performance is reported.
"""
import argparse,csv,json,hashlib,time,types,importlib.metadata
from pathlib import Path
import numpy as np
import torch
import reproduce_baselines as base

OFFSETS=[.26,.30,.36,.45]
def write(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def preference_order(cost,minimum,violation_integral):
    """Safe first; safe by cost; unsafe by max separation, then exposure, cost, index."""
    cost,minimum,violation_integral=map(np.asarray,(cost,minimum,violation_integral))
    assert cost.ndim==1 and minimum.shape==cost.shape==violation_integral.shape
    assert np.isfinite(cost).all() and np.isfinite(minimum).all() and np.isfinite(violation_integral).all()
    unsafe=minimum<.30;idx=np.arange(len(cost))
    return np.lexsort((idx,cost,np.where(unsafe,violation_integral,0),np.where(unsafe,-minimum,0),unsafe.astype(int)))
def pairs(cost,minimum,exposure):
    order=preference_order(cost,minimum,exposure);rank=np.empty(len(order),int);rank[order]=np.arange(len(order))
    i,j=np.triu_indices(len(cost),1);safe=minimum>=.30
    # Ignore physically indistinguishable within-class pairs; index breaks selection ties only.
    cross=safe[i]!=safe[j];ss=safe[i]&safe[j]&(abs(cost[i]-cost[j])>1e-4)
    uu=(~safe[i])&(~safe[j])&((abs(minimum[i]-minimum[j])>1e-6)|(abs(exposure[i]-exposure[j])>1e-6)|(abs(cost[i]-cost[j])>1e-4))
    keep=cross|ss|uu;i,j=i[keep],j[keep]
    return np.stack([np.where(rank[i]<rank[j],i,j),np.where(rank[i]<rank[j],j,i)],1),np.where(cross[keep],0,np.where(ss[keep],1,2)).astype(np.int8)
def namespace(family,goals):
    ns=base.context(family,'cuda',200);ns['GOALS']=np.asarray(goals,np.float32)
    ns['CONFIG']=dict(base.CONFIG)
    base.functions(ns,'04_WorldModel_Integration.ipynb',[47,48,41],['line_intersection','make_conflict_state','generate_conflict_actions','calculate_ranking_labels'])
    ns['rank_adapter']=types.SimpleNamespace(action_limits=torch.ones(2,device='cuda'))
    return ns
def rollout(ns,initial,actions):
    env=ns['create_scenario'](64);ns['set_cem_state'](env,initial[:,:2],initial[:,2:])
    world=env.scenario.world;original=world._integrate_state;trace=[]
    def integration(w,entity,substep):
        original(entity,substep)
        if entity is w.agents[-1]:trace.append(base.state(env).detach().clone())
    world._integrate_state=types.MethodType(integration,world)
    states=[base.state(env).detach().clone()]
    with torch.no_grad():
        for t in range(8):env.step([actions[:,t,i] for i in range(2)]);states.append(base.state(env).detach().clone())
    return torch.stack(states,1),torch.stack(trace,1)
def layouts():
    rng=np.random.default_rng(2026103200);records=[]
    for family,(starts,goals) in base.SCENARIOS.items():
        for j in range(6):
            theta=rng.uniform(-.3,.3);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            scale=rng.uniform(.96,1.04);translation=rng.uniform(-.08,.08,2)
            split='train' if j<4 else ('validation' if j==4 else 'test')
            records.append(dict(geometry_id=f'{family}_g{j}',family=family,split=split,
                starts=(np.array(starts)@rotation.T*scale+translation).astype(np.float32).tolist(),
                goals=(np.array(goals)@rotation.T*scale+translation).astype(np.float32).tolist()))
    return records
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists(),'Preserve prior attempts'
    torch.set_num_threads(1);new=layouts()
    pilot=[dict(geometry_id=family,family=family,split='public_pilot',starts=starts,goals=goals) for family,(starts,goals) in base.SCENARIOS.items()]
    protocol=dict(status='frozen',source_sha256=sha(Path(__file__)),notebook_sha256=sha(base.HERE/'notebooks/04_WorldModel_Integration.ipynb'),
        base_commit=base.BASE_COMMIT,offsets=OFFSETS,horizon=8,candidates=64,speed=.2,safe_distance=.30,
        native_physics='VMAS1.5.2 defaults, dt.1,substeps2,radius.1,world_drag.25; no extra limits',
        layouts=new,pilot=pilot,split_unit='Whole geometry: all four offsets and64 candidates remain together',
        semantics='Original notebook step soft labels preserved; new safety order uses all native substeps incl initial state.',
        unsafe_fallback='Maximize minimum separation, then minimize cumulative squared violation, soft cost, index.',
        targets='safe>unsafe; within safe cost; within unsafe separation/exposure/cost. No learning.',
        test_policy='Generate and validate schema/hash/labels only. No test method metrics, pair fitting or model selection.',
        versions={n:importlib.metadata.version(n) for n in ['vmas','torch','numpy']})
    write(out/'protocol.json',protocol);start=time.perf_counter();manifest=[];diagnostics=[];maxreplay=0.;pilot_matches=[];paircounts={'train':0,'validation':0}
    expected=list(csv.DictReader((base.HERE/'results/Conflict_Ranking_Dataset/conflict_pilot_diagnostics.csv').open()))
    expected_dec=list(csv.DictReader((base.HERE/'results/Conflict_Ranking_Dataset/cost_vs_safety_first.csv').open()))
    for geo_index,geo in enumerate(pilot+new):
        ns=namespace(geo['family'],geo['goals']);scene=dict(starts=geo['starts'],goals=geo['goals'])
        for offset_index,offset in enumerate(OFFSETS):
            initial=ns['make_conflict_state'](scene,offset,.2);assert np.linalg.norm(initial[0,:2]-initial[1,:2])>.30
            seed=2027+geo_index*4+offset_index if geo['split']=='public_pilot' else 2026110000+(geo_index-3)*4+offset_index
            actions=ns['generate_conflict_actions'](scene,64,8,seed)
            states,sub=rollout(ns,initial,actions);again,subagain=rollout(ns,initial,actions)
            error=float(max((states-again).abs().max().item(),(sub-subagain).abs().max().item()));maxreplay=max(maxreplay,error);assert error==0
            legacy=ns['calculate_ranking_labels'](states,actions,np.array(geo['goals'],np.float32))
            allsub=torch.cat([states[:,:1],sub],1);sep=torch.linalg.vector_norm(allsub[:,:,0,:2]-allsub[:,:,1,:2],dim=-1)
            minimum=sep.min(1).values.cpu().numpy();exposure=torch.relu(.30-sep).square().sum(1).cpu().numpy()
            order=preference_order(legacy['total_cost'],minimum,exposure);rank=np.empty(64,np.int32);rank[order]=np.arange(64)
            oldunsafe=legacy['violation_steps']>0;oldorder=np.lexsort((legacy['total_cost'],oldunsafe.astype(int)))
            name=f"{geo['geometry_id']}_o{offset_index}";folder=out/geo['split']/name;folder.mkdir(parents=True)
            np.savez_compressed(folder/'inputs.npz',initial_state=initial,goals=np.array(geo['goals'],np.float32),actions=actions.cpu().numpy(),candidate_mode=np.arange(64)%4)
            labels={k:v for k,v in legacy.items() if k!='best_candidate'}
            np.savez_compressed(folder/'labels.npz',**labels,state_path=states.cpu().numpy(),substep_state_path=sub.cpu().numpy(),
                substep_minimum_separation=minimum,substep_squared_violation=exposure,safety_order=order,safety_rank=rank,
                legacy_safety_order=oldorder,cost_best_index=legacy['best_candidate'],safety_best_index=order[0])
            if geo['split'] in paircounts:
                pref,kind=pairs(legacy['total_cost'],minimum,exposure);np.savez_compressed(folder/'preferences.npz',preferred_pairs=pref,kind=kind)
                paircounts[geo['split']]+=len(pref)
            record=dict(name=name,geometry_id=geo['geometry_id'],family=geo['family'],split=geo['split'],offset=offset,seed=seed,
                input_sha256=sha(folder/'inputs.npz'),labels_sha256=sha(folder/'labels.npz'))
            manifest.append(record)
            if geo['split']!='test':
                unsafe=minimum<.30;best=legacy['best_candidate'];safe=~unsafe
                diagnostics.append(dict(**record,safe_candidates=int(safe.sum()),unsafe_candidates=int(unsafe.sum()),mixed=bool(safe.any() and unsafe.any()),
                    cost_best_safe=bool(safe[best]),safety_best_safe=bool(safe[order[0]]),legacy_cost_best_safe=bool(not oldunsafe[best]),
                    legacy_safety_best_safe_under_substeps=bool(safe[oldorder[0]]),
                    safety_extra_soft_cost=float(legacy['total_cost'][order[0]]-legacy['total_cost'][best]),
                    step_vs_substep_candidate_disagreements=int((oldunsafe!=unsafe).sum())))
            if geo['split']=='public_pilot':
                old=expected[geo_index*4+offset_index];dec=expected_dec[geo_index*4+offset_index]
                match=int(old['unsafe_candidates'])==int(oldunsafe.sum()) and abs(float(old['minimum_separation'])-float(legacy['minimum_separation'].min()))<1e-5
                match=match and int(dec['cost_best_index'])==legacy['best_candidate'] and int(dec['best_safe_index'])==int(oldorder[0])
                pilot_matches.append(dict(name=name,match=match))
        write(out/'manifest.json',manifest);print('GEOMETRY',geo['split'],geo['geometry_id'],flush=True)
    assert len(manifest)==84 and len(pilot_matches)==12 and all(q['match'] for q in pilot_matches)
    summary={}
    for split in ['public_pilot','train','validation']:
        rows=[q for q in diagnostics if q['split']==split];geos=[]
        for gid in sorted({q['geometry_id'] for q in rows}):
            group=[q for q in rows if q['geometry_id']==gid];available=[q for q in group if q['safe_candidates']>0]
            geos.append(dict(geometry_id=gid,groups=len(group),mixed_groups=sum(q['mixed'] for q in group),no_safe_groups=sum(q['safe_candidates']==0 for q in group),
                soft_unsafe_given_safe_available=float(np.mean([not q['cost_best_safe'] for q in available])) if available else None,
                safety_unsafe_given_safe_available=float(np.mean([not q['safety_best_safe'] for q in available])) if available else None))
        summary[split]=dict(geometries=len(geos),groups=len(rows),mixed_groups=sum(q['mixed'] for q in rows),no_safe_groups=sum(q['safe_candidates']==0 for q in rows),
            soft_unsafe_given_safe_available_scene_mean=float(np.mean([q['soft_unsafe_given_safe_available'] for q in geos if q['soft_unsafe_given_safe_available'] is not None])),
            safety_unsafe_given_safe_available_scene_mean=float(np.mean([q['safety_unsafe_given_safe_available'] for q in geos if q['safety_unsafe_given_safe_available'] is not None])),
            step_substep_disagreements=sum(q['step_vs_substep_candidate_disagreements'] for q in rows),scene_metrics=geos)
    write(out/'group_diagnostics.json',diagnostics);write(out/'pilot_comparison.json',pilot_matches)
    write(out/'summary.json',dict(status='passed',splits=summary,preference_pair_counts=paircounts,groups=84,labeled_candidates=5376,
        new_groups=72,new_labeled_candidates=4608,max_native_replay_error=maxreplay,test_method_evaluations=0,
        wall_seconds=time.perf_counter()-start,limitations=['Oracle targets, not learned scoring.','Finite H8 candidate safety is not closed-loop safety.',
        'Layouts are transformations of three route families; not held-out-family generalization.']))
    print('COMPLETE',json.dumps({k:{a:b for a,b in v.items() if a!='scene_metrics'} for k,v in summary.items()}),flush=True)
if __name__=='__main__':main()
