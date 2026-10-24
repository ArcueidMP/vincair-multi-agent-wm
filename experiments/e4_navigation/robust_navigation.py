"""Frozen analytic-goal/risk hybrid robustness; no training or parameter tuning."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import torch
import closed_loop_pilot as pilot
import repair_navigation_cost as cost
import state_action_ranker as sr

CONDITIONS=['nominal','fast_approach','arrival_offset','terminal_neighborhood']
ARMS=['invariant_repair','analytic_goal','true_sim_safety']
def layouts():
    rng=np.random.default_rng(2026105101);rows=[]
    for i in range(6):
        angle=float(rng.uniform(45,135));theta=float(rng.uniform(-np.pi,np.pi));center=rng.uniform(-.08,.08,2)
        d=np.array([[np.cos(theta),np.sin(theta)],[np.cos(theta+np.radians(angle)),np.sin(theta+np.radians(angle))]],np.float32);goals=center+.75*d
        for condition in CONDITIONS:
            offsets=np.array([.75,.75]);speeds=np.array([.12,.24]);starts=center-offsets[:,None]*d
            if condition=='fast_approach':speeds=np.array([.45,.60])
            elif condition=='arrival_offset':starts=center-np.array([.90,.45])[:,None]*d;speeds=np.array([.20,.20])
            elif condition=='terminal_neighborhood':starts=goals-.20*d;speeds=np.array([.30,.45])
            assert np.linalg.norm(starts[0]-starts[1])>.30
            rows.append(dict(name=f'robust_{i:02}_{condition}',base_id=f'robust_{i:02}',condition=condition,angle_deg=angle,
                directions=d.tolist(),starts=starts.tolist(),goals=goals.tolist(),velocity=(speeds[:,None]*d).tolist()))
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
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--repair',type=Path,required=True);a=ap.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Preserve prior attempt')
    torch.set_num_threads(1);models,hashes=pilot.load_models(a.baseline,a.repair);models['analytic_goal']=dict(models['invariant_repair'],progress_mode='analytic');geos=layouts();start=time.perf_counter()
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_commit='7e9650d687ebbc6fc7e1eda41062d92eac08e4a0',source_sha256=sr.sha(Path(__file__)),
        geometry_specs=geos,model_sha256=hashes,conditions=CONDITIONS,arms=ARMS,episodes=72,max_steps=200,horizon=8,candidates=72,
        pairing='Six independent base geometries; four correlated condition variants per base, one candidate seed per base shared across conditions/arms. State-dependent pools diverge.',
        seed='2026160000+1000*base index; plus execution step, unchanged candidate generator',
        interventions='Nominal route speeds.12/.24; fast.45/.60; arrival distances.90/.45 and speeds.20/.20; terminal starts.20 before each goal and speeds.30/.45',
        caveat='Arrival condition changes both positions and speeds vs nominal; terminal condition tests final approach, not a crossing conflict.',
        contract='Native VMAS1.5.2 dt.1 substeps2 drag.25 radius.1; separation.30 incl initial/native substeps; joint position tolerance.12; no terminal speed criterion',
        gate='Descriptive robustness screen: analytic safe-joint-success >=.95 in EACH condition and separation-violation rate <= true-sim reference in EACH condition. No population safety guarantee.',
        training='No fitting, threshold/physics/proposal changes, no prior confirmation data read',budget='72 episodes max200 steps +3 separate5-step smoke;900s wall cap checked between episodes',
        limitations=['Only6 independent geometry clusters; do not count24 conditions as independent scenes','Single candidate seed per base','Hand-coded free-motion goal dynamics, frozen direct risk scorer; not LeWM'])
    sr.write(out/'protocol.json',protocol);pilot.decide=cost.decision
    def run(geo,arm,seed,steps,section):
        r,z=pilot.episode(geo,arm,seed,models,steps);p=out/section/f"{geo['name']}_{arm}_{seed}";p.mkdir(parents=True)
        np.savez_compressed(p/'trajectory.npz',**z);r['trajectory_sha256']=sr.sha(p/'trajectory.npz');r.update(base_id=geo.get('base_id','smoke'),condition=geo.get('condition','smoke'));sr.write(p/'result.json',r);return r
    smoke=dict(name='smoke',starts=[[-.7,-.3],[-.3,.7]],goals=[[.7,.3],[.3,-.7]],velocity=[[.1,.04],[.04,-.1]],directions=[[.919145,.393919],[.393919,-.919145]])
    sr.write(out/'smoke_passed.json',dict(results=[run(smoke,arm,2026169999,5,'smoke') for arm in ARMS]));rows=[]
    for geo in geos:
        gi=int(geo['base_id'].split('_')[-1]);seed=2026160000+gi*1000
        for arm in ARMS:
            if time.perf_counter()-start>900:raise RuntimeError('Budget exceeded; preserve partial run')
            rows.append(run(geo,arm,seed,200,'episodes'));sr.write(out/'episode_results.json',rows);print('EPISODE',len(rows),'/72',geo['condition'],arm,rows[-1]['safe_joint_success'],flush=True)
    _,after=pilot.load_models(a.baseline,a.repair);assert after==hashes;stats=aggregate(rows)
    gate=all(stats[c]['analytic_goal']['mean']['safe_joint_success']>=.95 and stats[c]['analytic_goal']['mean']['substep_loss_of_separation']<=stats[c]['true_sim_safety']['mean']['substep_loss_of_separation'] for c in CONDITIONS)
    sr.write(out/'summary.json',dict(by_condition=stats,episodes=72,independent_base_geometries=6,robustness_gate_passed=gate,model_hashes_unchanged=True,no_training=True,wall_seconds=time.perf_counter()-start))
    sr.write(out/'EVALUATION_CONSUMED.json',dict(reason='Frozen robustness screen consumed; do not tune on these6 bases or their24 variants',protocol_sha256=sr.sha(out/'protocol.json')))
    print('COMPLETE',json.dumps(stats),flush=True)
if __name__=='__main__':main()
