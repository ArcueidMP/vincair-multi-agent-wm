"""Portable inference-only loader for our trained direct safety scorer, NOT LeWM."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import state_action_ranker as sr
import closed_loop_pilot as pilot
import repair_navigation_cost as cost

DEFAULT_MODEL=Path(__file__).resolve().parent/'shared_models/invariant_safety_v1.json'

def load(path=DEFAULT_MODEL):
    obj=json.loads(Path(path).read_text());assert obj['schema']=='direct_safety_mlp_v1' and obj['architecture']==[44,64,32,3]
    arrays={k:np.array(v,np.float32) for k,v in obj['arrays'].items()};net=sr.Scorer().eval()
    state={k:torch.from_numpy(arrays[k.replace('.','_')]) for k in net.state_dict()};net.load_state_dict(state,strict=True)
    c=dict(state_dict=state,cost_mean=obj['cost_mean'],cost_std=obj['cost_std'],policy=obj['policy'],epoch=obj['epoch'])
    risk=dict(net=net,checkpoint=c,kind='invariant',mean=arrays['mean'],std=arrays['std'])
    assert (risk['std']>=.01).all() and sum(p.numel() for p in net.parameters())==5059
    return dict(invariant_repair=risk,analytic_goal=dict(risk,progress_mode='analytic')),obj

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,default=DEFAULT_MODEL);ap.add_argument('--protocol',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--full',action='store_true');a=ap.parse_args()
    if a.output.exists():raise RuntimeError('Use a NEW output directory; preserve existing runs')
    a.output.mkdir(parents=True);torch.set_num_threads(1);models,obj=load(a.model);pilot.decide=cost.decision
    if a.full:
        if a.protocol is None:raise ValueError('--full requires frozen robustness --protocol')
        protocol=json.loads(a.protocol.read_text());geos=protocol['geometry_specs'];tasks=[(geo,arm,2026160000+1000*int(geo['base_id'].split('_')[-1]),200) for geo in geos for arm in protocol['arms']]
    else:
        geo=dict(name='portable_smoke',starts=[[-.7,-.3],[-.3,.7]],goals=[[.7,.3],[.3,-.7]],velocity=[[.1,.04],[.04,-.1]],directions=[[.919145,.393919],[.393919,-.919145]])
        tasks=[(geo,arm,2026169999,5) for arm in ['invariant_repair','analytic_goal','true_sim_safety']]
    sr.write(a.output/'reproduction_protocol.json',dict(model_json_sha256=sr.sha(a.model),source_checkpoint_sha256=obj['source_checkpoint_sha256'],full=a.full,tasks=len(tasks),
        kind='Portable direct-scorer/analytic hybrid, NOT Navigation LeWM',original_protocol_sha256=sr.sha(a.protocol) if a.protocol else None))
    results=[]
    for geo,arm,seed,steps in tasks:
        result,arrays=pilot.episode(geo,arm,seed,models,steps);p=a.output/f"{geo['name']}_{arm}_{seed}";p.mkdir()
        np.savez_compressed(p/'trajectory.npz',**arrays);sr.write(p/'result.json',result);results.append(result)
        print('REPRODUCED',len(results),'/',len(tasks),geo['name'],arm,result['safe_joint_success'],flush=True)
    sr.write(a.output/'results.json',results);print('Complete. All trajectories include fresh native replay checks.')
if __name__=='__main__':main()
