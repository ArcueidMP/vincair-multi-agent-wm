"""Independent saved-array validation and publishable reproduction summary."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run
    here=Path(__file__).resolve().parent;dest=here/'reproduction_results/v1';dest.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((run/'protocol.json').read_text());assert (run/'full_completed.json').exists() and (run/'smoke_completed.json').exists()
    records=[];manifest=[];maxendpoint=0.;maxmetrics=0.
    for p in sorted(run.glob('*/result.json')):
        result=json.loads(p.read_text());traj=p.with_name('trajectory.npz');assert sha(traj)==result['trajectory_sha256']
        with np.load(traj,allow_pickle=False) as z:
            state=z['state'];sub=z['substep_state'];action=z['action'];goals=z['goals'];steps=len(action)
            assert state.shape==(steps+1,2,4) and sub.shape==(steps*2,2,4) and action.shape==(steps,2,2)
            assert np.isfinite(state).all() and np.isfinite(sub).all() and abs(action).max()<=1.000001
            endpoint=float(abs(sub[1::2]-state[1:]).max());maxendpoint=max(maxendpoint,endpoint);assert endpoint==0
            sep=np.linalg.norm(state[:,0,:2]-state[:,1,:2],axis=-1);ssep=np.linalg.norm(sub[:,0,:2]-sub[:,1,:2],axis=-1)
            submin=float(min(sep[0],ssep.min()));success=bool((np.linalg.norm(state[-1,:,:2]-goals,axis=1)<.12).all())
            assert success==result['joint_success'] and steps==result['steps']
            assert bool((sep<.30).any())==result['loss_of_separation']
            assert (submin<.30)==result['substep_loss_of_separation']
            assert result['safe_joint_success']==(success and submin>=.30)
            assert int((sep<.30).sum())==result['violation_steps']
            maxmetrics=max(maxmetrics,abs(float(sep.min())-result['minimum_separation']),abs(submin-result['substep_minimum_separation']))
            assert result['native_replay_max_abs_error']<=1e-6
        records.append(result);manifest.append(dict(episode=p.parent.name,trajectory_sha256=sha(traj),result_sha256=sha(p),bytes=traj.stat().st_size))
    assert len(records)==21
    originals=list(csv.DictReader((here/'results/CEM_baselines/E4B_multi_scenario_results.csv').open()))
    comparison=[]
    for old in originals:
        method='centralized_safety' if old['with_safety']=='True' else 'centralized_no_safety'
        new=next(q for q in records if q['scene']==old['scenario'] and q['method']==method and q['seed']==int(old['seed']))
        delta=abs(float(old['min_separation'])-new['minimum_separation'])
        match=new['steps']==int(old['steps']) and new['joint_success']==(old['joint_success']=='True') and delta<1e-5 and new['violation_steps']==int(old['violation_steps'])
        comparison.append(dict(scene=new['scene'],seed=new['seed'],method=method,reference_match=match,
            reference_steps=int(old['steps']),local_steps=new['steps'],minimum_separation_abs_difference=delta))
    assert len(comparison)==18
    summary={}
    for method in sorted({q['method'] for q in records}):
        rr=[q for q in records if q['method']==method]
        summary[method]=dict(episodes=len(rr),scene_geometries=len({q['scene'] for q in rr}),successes=sum(q['joint_success'] for q in rr),
            safe_successes=sum(q['safe_joint_success'] for q in rr),loss_of_separation_episodes=sum(q['substep_loss_of_separation'] for q in rr),
            collision_proxy_episodes=sum(q['substep_collision_proxy'] for q in rr),
            mean_steps=float(np.mean([q['steps'] for q in rr])),mean_planning_ms=float(np.mean([q['mean_planning_ms'] for q in rr])),
            min_substep_separation=float(min(q['substep_minimum_separation'] for q in rr)))
    verification=dict(status='passed',episodes=len(records),all_native_replayed=True,max_native_replay_error=max(q['native_replay_max_abs_error'] for q in records),
        substep_endpoint_max_error=maxendpoint,saved_metric_max_error=maxmetrics,input_arrays_finite_and_actions_bounded=True,
        independent_numpy_verification=True,upstream_comparisons=18,upstream_matches=sum(q['reference_match'] for q in comparison),
        note='Cached upstream comparison excludes latency; three fixed layouts with three planner RNG seeds are not nine independently sampled geometries.')
    write(dest/'protocol.json',protocol);write(dest/'episode_metrics.json',records);write(dest/'summary.json',summary)
    write(dest/'verification.json',verification);write(dest/'trajectory_manifest.json',manifest);write(dest/'upstream_comparison.json',comparison)
    cols=['scene','method','seed','steps','joint_success','safe_joint_success','minimum_separation','substep_minimum_separation','violation_steps','mean_planning_ms','native_replay_max_abs_error']
    with (dest/'episode_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,lineterminator='\n');w.writeheader();w.writerows({k:q[k] for k in cols} for q in records)
    lines=['| Method | Episodes | Goal success | Safe goal success | Substep loss of separation | Mean latency(ms) |','|---|---:|---:|---:|---:|---:|']
    for name,q in summary.items():lines.append(f"| {name} | {q['episodes']} | {q['successes']} | {q['safe_successes']} | {q['loss_of_separation_episodes']} | {q['mean_planning_ms']:.2f} |")
    doc=f'''# E4 baseline reproduction v1

Actual local runs based on teammate commit `{protocol['base_commit']}`. This is true-VMAS reproduction, not learned world-model planning.

## Results

{chr(10).join(lines)}

Two PD policies and independent CEM cover S1 seed42 only. Centralized arms each cover three fixed layouts and three planner RNG seeds. These are different evaluation scopes: do not compare the counts as a common generalization benchmark. Safety means separation >=0.30 at every recorded native integration substep, including initial state; goal success preserves the original position-only tolerance0.12. There is no terminal-speed requirement in this reproduction.

Cached upstream comparisons: **{verification['upstream_matches']}/18** match step count, goal success, step-level violation count and minimum separation within1e-5. Latency is hardware dependent and is not expected to match. Substep metrics are additional diagnostics, not retroactively attributed to the team's old results.

## Validation

All21 action sequences were replayed in fresh native VMAS environments, maximum state error **{verification['max_native_replay_error']}**. Saved-array checks independently recompute success, separation, bounds, substep endpoints and hashes. Four targeted unit tests compare original notebook functions against this entry point and check scenario-goal isolation and substep safety. Initial development errors in optional VMAS metadata access (`substeps`, absent `close()`, nullable agent drag) were corrected before collecting results; no physics or planner parameters changed.

## Environment and limitations

Versions: `{json.dumps(protocol['versions'])}`. Local runtime is CUDA on an RTX4060 laptop; planning uses native VMAS with dt0.1, two substeps, agent radius0.1, action component limits±1, world drag0.25, and no added speed/acceleration cap. This differs from earlier local bounded-navigation pilots; their success figures must not be pooled with these.

The learned Navigation adapter in notebook04 remains unimplemented. No Navigation weight was downloaded or trained. Soft safety cost is not a safety guarantee. Existing decision-ranking data lack close conflicts; conflict-pilot labels are the next priority, with raw NPZ provenance and scene-group splits checked first.

## Raw artifacts

All state, velocity, executed action, native-substep state, goals and latency arrays are retained in the local run directory. Large/raw artifacts are excluded from Git; `trajectory_manifest.json` identifies every NPZ by SHA256 and bytes. No model weights, private chat screenshots, or absolute personal filesystem paths are included in this result bundle. Full rerun commands are in the parent `REPRODUCTION.md`.
'''
    (dest/'README.md').write_text(doc,encoding='utf-8')
    write(run/'verification.json',verification);write(run/'summary.json',summary)
    (run/'REPRODUCTION_REPORT.md').write_text(doc,encoding='utf-8')
    print(json.dumps(dict(verification=verification,summary=summary),indent=2))
if __name__=='__main__':main()
