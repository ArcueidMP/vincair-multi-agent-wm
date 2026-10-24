"""New near-conflict/inertia screen using ONLY the published direct scorer JSON."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import torch
import shared_navigation_model as shared
import repair_navigation_cost as cost
import closed_loop_pilot as pilot
import state_action_ranker as sr

ARMS=['invariant_repair','analytic_goal','true_sim_safety']
CONDITIONS=['near_conflict','high_inertia']
def layouts():
    rng=np.random.default_rng(2026105201);rows=[]
    for i in range(6):
        angle=float(rng.uniform(75,135));theta=float(rng.uniform(-np.pi,np.pi));center=rng.uniform(-.08,.08,2)
        d=np.array([[np.cos(theta),np.sin(theta)],[np.cos(theta+np.radians(angle)),np.sin(theta+np.radians(angle))]],np.float32)
        for condition,separation,speeds in [('near_conflict',.42,[.10,.20]),('high_inertia',.36,[.35,.50])]:
            offset=separation/(2*np.sin(np.radians(angle)/2))
            rows.append(dict(name=f'pressure_{i:02}_{condition}',base_id=f'pressure_{i:02}',condition=condition,angle_deg=angle,
                directions=d.tolist(),starts=(center-offset*d).tolist(),goals=(center+.75*d).tolist(),velocity=(np.array(speeds)[:,None]*d).tolist(),initial_separation=separation))
    return rows

def aggregate(rows):
    result={}
    for condition in CONDITIONS:
        result[condition]={}
        for arm in ARMS:
            rr=[r for r in rows if r['condition']==condition and r['arm']==arm]
            result[condition][arm]=dict(episodes=len(rr),mean={k:float(np.mean([r[k] for r in rr])) for k in ['joint_success','safe_joint_success','substep_loss_of_separation','substep_collision_proxy','steps','mean_planning_ms']})
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
    if out.exists():raise RuntimeError('Preserve prior attempt; use a NEW output directory')
    out.mkdir(parents=True);torch.set_num_threads(1);models,obj=shared.load();pilot.decide=cost.decision;geos=layouts();started=time.perf_counter()
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_commit='d2d0af77c1710bccfd70d4158aab1bde6141ecee',source_sha256=sr.sha(Path(__file__)),portable_model_sha256=sr.sha(shared.DEFAULT_MODEL),
        geometry_specs=geos,arms=ARMS,conditions=CONDITIONS,episodes=36,max_steps=200,horizon=8,candidates=72,
        pairing='Six NEW independent base geometries x2 correlated conditions; candidate seed2026170000+1000*base index, plus execution step',
        contract='Unchanged native VMAS1.5.2 dt.1 substeps2 radius.1 drag.25, separation.30 incl initial/native substeps, goal tolerance.12 position only',
        interventions='Near conflict: initial separation.42, route speeds.10/.20. High inertia: separation.36, route speeds.35/.50. Both initially safe.',
        metrics='Joint/safe joint goal success, episode separation/collision proxy, first true-sim pool safe candidate count and max minimum separation',
        gate='Descriptive stress screen: analytic safe joint success>=.95 AND violation no worse than finite-pool true-sim reference in EACH condition',
        interpretation='Zero initial-pool safe candidates is a H8/72 candidate-coverage diagnostic, NOT proof of physical impossibility or global unavoidability.',
        no_training=True,no_threshold_or_proposal_changes=True,budget='36 episodes max200 steps,900s wall checked between episodes; no post-result changes',
        limitations=['Only6 independent geometry clusters','Two conditions also change velocity and separation jointly','Finite candidate reference is privileged, not optimal','No Navigation LeWM or three-agent claim'])
    sr.write(out/'protocol.json',protocol);rows=[]
    for geo in geos:
        seed=2026170000+1000*int(geo['base_id'].split('_')[-1])
        for arm in ARMS:
            if time.perf_counter()-started>900:raise RuntimeError('Wall budget reached; preserve partial outputs')
            result,z=pilot.episode(geo,arm,seed,models,200);p=out/'episodes'/f"{geo['name']}_{arm}_{seed}";p.mkdir(parents=True)
            np.savez_compressed(p/'trajectory.npz',**z);result.update(base_id=geo['base_id'],condition=geo['condition'],trajectory_sha256=sr.sha(p/'trajectory.npz'))
            if arm=='true_sim_safety':result.update(initial_pool_safe_candidates=int((z['oracle_minimum'][0]>=.3).sum()),initial_pool_best_minimum=float(z['oracle_minimum'][0].max()))
            sr.write(p/'result.json',result);rows.append(result);sr.write(out/'episode_results.json',rows);print('EPISODE',len(rows),'/36',geo['condition'],arm,result['safe_joint_success'],flush=True)
    stats=aggregate(rows);gate=all(stats[c]['analytic_goal']['mean']['safe_joint_success']>=.95 and stats[c]['analytic_goal']['mean']['substep_loss_of_separation']<=stats[c]['true_sim_safety']['mean']['substep_loss_of_separation'] for c in CONDITIONS)
    sr.write(out/'summary.json',dict(by_condition=stats,episodes=36,independent_base_geometries=6,stress_gate_passed=gate,model_unchanged=sr.sha(shared.DEFAULT_MODEL)==protocol['portable_model_sha256'],wall_seconds=time.perf_counter()-started))
    sr.write(out/'EVALUATION_CONSUMED.json',dict(reason='New pressure evaluation consumed; no tuning on these6 bases or variants',protocol_sha256=sr.sha(out/'protocol.json')))
    print('COMPLETE',json.dumps(stats),flush=True)
if __name__=='__main__':main()
