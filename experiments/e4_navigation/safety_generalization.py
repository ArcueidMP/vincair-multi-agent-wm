"""Post-hoc failure audit and predeclared new-scenario evaluation, frozen scorer."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import torch
import conflict_ranking as cr
import state_action_ranker as sr

CONDITIONS=['angle_orientation','velocity_asymmetry','arrival_offset']
def scenarios():
    rng=np.random.default_rng(2026103400);rows=[]
    for angle in [45,75,105,135]:
        for replica in range(3):
            theta=rng.uniform(-np.pi,np.pi);center=rng.uniform(-.08,.08,2)
            directions=np.array([[np.cos(theta),np.sin(theta)],[np.cos(theta+np.radians(angle)),np.sin(theta+np.radians(angle))]],np.float32)
            rows.append(dict(base_id=f'angle{angle}_rep{replica}',angle_deg=angle,orientation_rad=float(theta),center=center.tolist(),
                directions=directions.tolist(),starts=(center-directions).astype(np.float32).tolist(),goals=(center+directions).astype(np.float32).tolist()))
    return rows
def initial_state(geo,condition,distance):
    d=np.array(geo['directions'],np.float32);center=np.array(geo['center'],np.float32)
    offset=distance/(2*np.sin(np.radians(geo['angle_deg'])/2));offsets=np.array([offset,offset],np.float32)
    velocity=.2*d
    if condition=='velocity_asymmetry':
        perp=np.stack([-d[:,1],d[:,0]],-1);velocity=np.array([.08,.35],np.float32)[:,None]*d+np.array([.08,-.06],np.float32)[:,None]*perp
    elif condition=='arrival_offset':offsets+=np.array([.1,-.1],np.float32)
    elif condition!='angle_orientation':raise ValueError(condition)
    state=np.concatenate([center-offsets[:,None]*d,velocity],1).astype(np.float32)
    assert np.linalg.norm(state[0,:2]-state[1,:2])>.30
    return state
def audit(row,raw,truth,minimum,cm,cs,threshold):
    picked,fallback,cost,risk,sep=sr.select(raw[None],cm,cs,threshold)
    i=int(picked[0]);fallback=bool(fallback[0]);cost,risk,sep=cost[0],risk[0],sep[0]
    safe=minimum>=.30;accepted=risk<threshold
    bestsafe=int(np.flatnonzero(safe)[truth[safe].argmin()]) if safe.any() else None
    gap=float(truth[i]-truth[bestsafe]) if bestsafe is not None else None
    if not safe.any():category='no_safe_candidate'
    elif not safe[i]:category='fallback_miss' if fallback else 'risk_false_accept'
    elif gap<=1e-4:category='safe_optimal'
    elif not accepted[bestsafe]:category='safe_risk_exclusion'
    else:category='safe_cost_ranking'
    predicted_safe=np.flatnonzero(accepted);true_safe=np.flatnonzero(safe)
    risk_oracle=int(true_safe[cost[true_safe].argmin()]) if len(true_safe) else i
    cost_oracle=int(predicted_safe[truth[predicted_safe].argmin()]) if len(predicted_safe) else i
    choices=dict(learned_safety=i,learned_cost_only=int(cost.argmin()),true_risk_predicted_cost=risk_oracle,
        predicted_risk_true_cost=cost_oracle,true_soft_oracle=int(truth.argmin()))
    methods={name:dict(index=idx,unsafe=bool(not safe[idx]),regret=float(truth[idx]-truth.min()),minimum_separation=float(minimum[idx])) for name,idx in choices.items()}
    return dict(**row,category=category,safe_available=bool(safe.any()),safe_candidates=int(safe.sum()),predicted_safe_candidates=int(accepted.sum()),
        selected=i,fallback=fallback,selected_risk=float(risk[i]),selected_predicted_minimum=float(sep[i]),selected_true_minimum=float(minimum[i]),
        best_true_safe_index=bestsafe,safe_cost_gap=gap,false_safe_candidates=int((accepted&~safe).sum()),
        false_unsafe_candidates=int((~accepted&safe).sum()),methods=methods)
def aggregate(rows):
    methods={}
    for method in ['learned_safety','learned_cost_only','true_risk_predicted_cost','predicted_risk_true_cost','true_soft_oracle']:
        scenes=[]
        for gid in sorted({q['base_id'] for q in rows}):
            g=[q for q in rows if q['base_id']==gid];eligible=[q for q in g if q['safe_available']]
            scenes.append(dict(base_id=gid,unsafe_given_safe=float(np.mean([q['methods'][method]['unsafe'] for q in eligible])) if eligible else None,
                unsafe_all=float(np.mean([q['methods'][method]['unsafe'] for q in g])),regret=float(np.mean([q['methods'][method]['regret'] for q in g]))))
        methods[method]=dict(mean={k:float(np.mean([q[k] for q in scenes if q[k] is not None])) if any(q[k] is not None for q in scenes) else None for k in ['unsafe_given_safe','unsafe_all','regret']},scenes=scenes)
    return dict(groups=len(rows),base_geometries=len({q['base_id'] for q in rows}),no_safe_groups=sum(not q['safe_available'] for q in rows),
        category_counts={k:sum(q['category']==k for q in rows) for k in ['no_safe_candidate','risk_false_accept','fallback_miss','safe_optimal','safe_risk_exclusion','safe_cost_ranking']},methods=methods)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    frozen=json.loads((args.model/'selection_frozen.json').read_text());modelpath=args.model/'state_action_scorer.pt'
    assert sr.sha(modelpath)==frozen['checkpoint_sha256'];checkpoint=torch.load(modelpath,map_location='cpu',weights_only=False)
    net=sr.Scorer().eval();net.load_state_dict(checkpoint['state_dict']);torch.set_num_threads(1)
    mean=checkpoint['mean'].numpy();std=checkpoint['std'].numpy();cm=checkpoint['cost_mean'];cs=checkpoint['cost_std'];threshold=checkpoint['threshold']
    geos=scenarios();protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sr.sha(Path(__file__)),checkpoint_sha256=sr.sha(modelpath),threshold=threshold,epoch=checkpoint['epoch'],
        geometry_specs=geos,conditions=CONDITIONS,initial_distances=[.34,.45],horizon=8,candidates=64,new_groups=72,new_candidates=4608,
        paired='Within base geometry/distance, identical goals and64 candidate actions across all3 conditions; only initial state changes.',
        inference='Frozen5059-param direct state/action scorer. CPU inference, native VMAS CUDA labels. No training/threshold changes.',
        diagnostic_categories=['no_safe_candidate','risk_false_accept','fallback_miss','safe_optimal','safe_risk_exclusion','safe_cost_ranking'],
        safe_cost_tie_tolerance=1e-4,old_data_use='Previously consumed validation/test: explicitly post-hoc diagnosis, no model selection.',
        metrics='Within base geometry average states, then equal-weight base geometries. Show all categories and conditions; no significance claim.',
        case_selection='Old test: lexicographically first unsafe selected case with safe option, first all-unsafe, first safe selection (deduplicated). No best-looking selection.',
        new_geometry_limitations=['Changes in orientation/action distribution can jointly shift raw44D features.','No learned world model or closed-loop safety claim.'],
        no_refit=True,no_threshold_search=True)
    sr.write(out/'protocol.json',protocol);start=time.perf_counter()
    def predict(initial,goals,actions):
        x=sr.features(initial,goals,actions)
        with torch.no_grad():raw=net(torch.from_numpy((x-mean)/std)).numpy()
        return x,raw
    old=[];oldpred=[];prior_initial=[]
    for row in json.loads((args.data/'manifest.json').read_text()):
        p=args.data/row['split']/row['name']
        with np.load(p/'inputs.npz') as z:
            initial=z['initial_state'];goals=z['goals'];actions=z['actions'];prior_initial.append(initial.copy())
            if row['split'] not in ['validation','test']:continue
            x,raw=predict(initial,goals,actions)
        with np.load(p/'labels.npz') as z:truth=z['total_cost'];minimum=z['substep_minimum_separation']
        record=audit(dict(name=row['name'],base_id=row['geometry_id'],split=row['split']),raw,truth,minimum,cm,cs,threshold)
        old.append(record);oldpred.append(raw)
    sr.write(out/'old_failure_records.json',old)
    sr.write(out/'old_failure_summary.json',{split:aggregate([q for q in old if q['split']==split]) for split in ['validation','test']})
    records=[];predictions=[];maxreplay=0.;saved_actions={}
    for gi,geo in enumerate(geos):
        ns=cr.namespace('S1_symmetric',geo['goals']);scene=dict(starts=geo['starts'],goals=geo['goals'])
        for di,distance in enumerate([.34,.45]):
            actions=ns['generate_conflict_actions'](scene,64,8,2026111000+gi*2+di);an=actions.cpu().numpy()
            for condition in CONDITIONS:
                initial=initial_state(geo,condition,distance)
                assert not any(np.allclose(initial,s,rtol=0,atol=1e-5) for s in prior_initial)
                states,sub=cr.rollout(ns,initial,actions);again,subagain=cr.rollout(ns,initial,actions)
                error=max(float((states-again).abs().max()),float((sub-subagain).abs().max()));maxreplay=max(maxreplay,error);assert error==0
                legacy=ns['calculate_ranking_labels'](states,actions,np.array(geo['goals'],np.float32))
                allsub=torch.cat([states[:,:1],sub],1);ds=torch.linalg.vector_norm(allsub[:,:,0,:2]-allsub[:,:,1,:2],dim=-1)
                minimum=ds.min(1).values.cpu().numpy();truth=legacy['total_cost']
                x,raw=predict(initial,np.array(geo['goals'],np.float32),an)
                name=f"{geo['base_id']}_{condition}_d{di}";folder=out/'new_data'/name;folder.mkdir(parents=True)
                np.savez_compressed(folder/'inputs.npz',initial_state=initial,goals=np.array(geo['goals'],np.float32),actions=an)
                np.savez_compressed(folder/'labels.npz',state_path=states.cpu().numpy(),substep_state_path=sub.cpu().numpy(),true_cost=truth,min_separation=minimum)
                np.savez_compressed(folder/'predictions.npz',features=x,raw=raw)
                row=dict(name=name,base_id=geo['base_id'],condition=condition,angle_deg=geo['angle_deg'],distance=distance,
                    input_sha256=sr.sha(folder/'inputs.npz'),labels_sha256=sr.sha(folder/'labels.npz'),prediction_sha256=sr.sha(folder/'predictions.npz'))
                records.append(audit(row,raw,truth,minimum,cm,cs,threshold))
        sr.write(out/'new_failure_records.json',records);print('BASE',gi+1,'/12',flush=True)
    assert len(records)==72 and sr.sha(modelpath)==frozen['checkpoint_sha256']
    summary=dict(status='passed',conditions={c:aggregate([q for q in records if q['condition']==c]) for c in CONDITIONS},
        all_conditions=aggregate(records),new_groups=72,new_candidates=4608,max_native_replay_error=maxreplay,
        checkpoint_unchanged=True,no_refit=True,no_threshold_search=True,wall_seconds=time.perf_counter()-start)
    sr.write(out/'summary.json',summary);print('COMPLETE',json.dumps({c:v['methods']['learned_safety']['mean'] for c,v in summary['conditions'].items()}),flush=True)
if __name__=='__main__':main()
