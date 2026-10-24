"""Goal-cost correction with frozen risk heads; direct scoring, not LeWM."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import torch
import closed_loop_pilot as pilot
import repair_safety as repair
import state_action_ranker as sr

ORIGINAL_DECIDE=pilot.decide
LAMBDAS=[.0001,.001,.01,.1,1.,10.]

def analytic(state,goals,actions):
    """Native free-motion recurrence; deliberately ignores inter-agent contact forces."""
    n=len(actions);pos=np.repeat(np.asarray(state)[None,:,:2],n,0).copy();vel=np.repeat(np.asarray(state)[None,:,2:],n,0).copy()
    initial=float(np.linalg.norm(state[:,:2]-goals,axis=-1).sum());minimum=np.linalg.norm(pos[:,0]-pos[:,1],axis=-1)
    for t in range(8):
        vel*=.75
        for _ in range(2):
            vel+=actions[:,t]*.05;pos+=vel*.05;minimum=np.minimum(minimum,np.linalg.norm(pos[:,0]-pos[:,1],axis=-1))
    distance=np.linalg.norm(pos-goals,axis=-1).sum(-1)
    return (5*distance+.01*(actions*actions).mean((1,2,3))).astype(np.float32),minimum.astype(np.float32),(initial-distance).astype(np.float32)

def correction_features(state,goals,actions,raw,cm,cs):
    views=repair.invariant_features(state,goals,actions);goal,minimum,progress=analytic(state,goals,actions)
    # All92 components use current inputs or frozen predictions, never observed future labels.
    return np.column_stack([views.mean(1),(views*views).mean(1),goal,minimum,raw[:,1]*cs+cm,progress]).astype(np.float64)

def decision(model,state,goals,actions):
    idx,fb,raw=ORIGINAL_DECIDE(model,state,goals,actions);mode=model.get('progress_mode')
    if mode is None:return idx,fb,raw
    c=model['checkpoint'];cm,cs=c['cost_mean'],c['cost_std'];raw=raw.copy()
    if mode=='analytic':cost=analytic(state,goals,actions)[0]
    else:
        w=model['correction'];x=correction_features(state,goals,actions,raw,cm,cs)
        cost=raw[:,1]*cs+cm+(x-w['mean'])/w['std']@w['coef']+w['intercept']
    raw[:,1]=((cost-cm)/cs).astype(np.float32)
    choice,fallback=repair.choose(raw[None],cm,cs,**c['policy'])
    return int(choice[0]),bool(fallback[0]),raw

def data(root,split,model):
    rows=[r for r in json.loads((root/'records.json').read_text()) if r['split']==split];xx=[];yy=[];raws=[];truth=[];minimum=[];progress=[]
    for r in rows:
        p=root/'snapshots'/r['name'];assert sr.sha(p/'inputs.npz')==r['input_sha256'] and sr.sha(p/'labels.npz')==r['label_sha256']
        with np.load(p/'inputs.npz') as z:
            s,g,a=z['initial_state'],z['goals'],z['actions'];_,_,raw=ORIGINAL_DECIDE(model,s,g,a)
            c=model['checkpoint'];xx.append(correction_features(s,g,a,raw,c['cost_mean'],c['cost_std']));raws.append(raw)
        with np.load(p/'labels.npz') as z:
            distance=np.linalg.norm(z['state_path'][:,-1,:,:2]-g,axis=-1).sum(-1)
            target=5*distance+.01*(a*a).mean((1,2,3));truth.append(target);yy.append(target-(raw[:,1]*c['cost_std']+c['cost_mean']))
            minimum.append(z['min_separation']);progress.append(np.linalg.norm(s[:,:2]-g,axis=-1).sum()-distance)
    # Equal geometry, then equal policy origin, then equal snapshots and candidates.
    gids=sorted({r['geometry'] for r in rows});weights=[]
    for r in rows:
        n=sum(q['geometry']==r['geometry'] and q['origin']==r['origin'] for q in rows)
        weights.append(1/(len(gids)*2*n*72))
    return rows,np.stack(xx),np.stack(yy),np.stack(raws),np.stack(truth),np.stack(minimum),np.stack(progress),np.repeat(weights,72)

def stats(rows,raw,cost,minimum,progress,c):
    idx,fb=repair.choose(raw,c['cost_mean'],c['cost_std'],**c['policy']);rec=[]
    for j,(r,i) in enumerate(zip(rows,idx)):
        rec.append(dict(name=r['name'],geometry=r['geometry'],origin=r['origin'],selected=int(i),unsafe=bool(minimum[j,i]<.3),
            safe_progress=bool(minimum[j,i]>=.3 and progress[j,i]>.05),goal_cost_regret=float(cost[j,i]-cost[j].min()),fallback=bool(fb[j])))
    groups=[]
    for gid in sorted({r['geometry'] for r in rows}):
        origins=[]
        for origin in ['invariant_repair','true_sim_safety']:
            rr=[r for r in rec if r['geometry']==gid and r['origin']==origin]
            origins.append({k:float(np.mean([r[k] for r in rr])) for k in ['unsafe','safe_progress','goal_cost_regret']})
        groups.append(dict(geometry=gid,**{k:float(np.mean([r[k] for r in origins])) for k in origins[0]}))
    return dict(mean={k:float(np.mean([g[k] for g in groups])) for k in ['unsafe','safe_progress','goal_cost_regret']},scenes=groups,records=rec)

def geometries():
    rng=np.random.default_rng(2026105001);rows=[]
    for i in range(6):
        angle=float(rng.uniform(40,140));theta=float(rng.uniform(-np.pi,np.pi));center=rng.uniform(-.08,.08,2);dist=float(rng.uniform(.6,.9))
        d=np.array([[np.cos(theta),np.sin(theta)],[np.cos(theta+np.radians(angle)),np.sin(theta+np.radians(angle))]],np.float32)
        rows.append(dict(name=f'cost_confirm_{i:02}',angle_deg=angle,directions=d.tolist(),starts=(center-dist*d).tolist(),
            goals=(center+.75*d).tolist(),velocity=(rng.uniform(.08,.25,(2,1))*d).tolist()))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--repair',type=Path,required=True);a=ap.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Preserve prior attempt')
    torch.set_num_threads(1);started=time.perf_counter();models,hashes=pilot.load_models(a.baseline,a.repair);risk=models['invariant_repair'];c=risk['checkpoint'];geos=geometries()
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sr.sha(Path(__file__)),parent_commit='15453877e0148d550c438e765d20d83e0b8980e1',
        data_records_sha256=sr.sha(a.data/'records.json'),risk_model_sha256=hashes['invariant_repair'],geometry_specs=geos,lambdas=LAMBDAS,
        target='True H8 terminal goal cost5*sum distances+.01*mean actions^2; no soft safety cost in correction target',
        features='92: mean canonical44, mean squared canonical44, free-motion goal cost/minsep, frozen cost prediction, free-motion progress',
        analytic='Known dt.1/substeps2/drag.25/mass1 recurrence; ignores contact forces. Hand-coded physics baseline, NOT a learned WM',
        fit='Weighted ridge residual to frozen cost, train8 whole geometries only; equal geometry/origin/snapshot/candidate weights; train-only weighted moments; weighted unpenalized intercept',
        selection='Validation4 geometries only; unsafe rate primary, negative safe-progress rate secondary, goal-cost regret third, lambda fourth; equal geometry and origin means',
        invariants='Frozen risk/minsep heads, threshold, margin and fallback; same72-proposal generator. Only cost ranking changes, including fallback cost tie-break.',
        confirmation='Freeze coefficients,lambda,protocol and model hashes before any NEW confirmation rollout;6 geometries x2 seeds x4 arms=48 episodes max200 steps, H8/72',
        arms=['invariant_repair','analytic_goal','ridge_goal','true_sim_safety'],seeds=[2026150000,2026151000],
        gate='Ridge safe-joint-success > frozen invariant AND episode separation violation <= frozen invariant; descriptive only. Report analytic competitor regardless of outcome.',
        budget='Four5-step smoke runs, then48 episodes;900s between-episode wall cap; no post-confirmation tuning',
        limitations=['One fitted model and6 confirmation geometries','Simple analytic baseline knows native free-motion parameters','Risk remains imperfect; ranking changes can increase unsafe choices','Local validation ranking does not establish closed-loop success'])
    sr.write(out/'protocol.json',protocol)
    tr,x,y,raw,truth,minimum,progress,weight=data(a.data,'train',risk);vr,vx,vy,vraw,vtruth,vmin,vprogress,_=data(a.data,'validation',risk)
    flat=x.reshape(-1,92);target=y.ravel();assert abs(weight.sum()-1)<1e-12
    mean=(flat*weight[:,None]).sum(0);std=np.maximum(np.sqrt(((flat-mean)**2*weight[:,None]).sum(0)),.01);intercept=float((target*weight).sum());X=(flat-mean)/std;Y=target-intercept
    gram=X.T@(weight[:,None]*X);rhs=X.T@(weight*Y);grid=[];best=None
    for lam in LAMBDAS:
        coef=np.linalg.solve(gram+lam*np.eye(92),rhs);prediction=vraw.copy();correction=((vx-mean)/std)@coef+intercept;prediction[...,1]+=correction/c['cost_std']
        s=stats(vr,prediction,vtruth,vmin,vprogress,c);m=s['mean'];key=(m['unsafe'],-m['safe_progress'],m['goal_cost_regret'],lam)
        grid.append(dict(lambda_value=lam,**m))
        if best is None or key<best:best=key;chosen=dict(mean=mean,std=std,coef=coef,intercept=np.array(intercept),lambda_value=np.array(lam));beststats=s
    np.savez(out/'goal_correction.npz',**chosen);sr.write(out/'validation_grid.json',grid);sr.write(out/'validation_selected.json',beststats)
    models['analytic_goal']=dict(risk,progress_mode='analytic');models['ridge_goal']=dict(risk,progress_mode='ridge',correction=chosen)
    sr.write(out/'selection_frozen.json',dict(status='frozen',protocol_sha256=sr.sha(out/'protocol.json'),correction_sha256=sr.sha(out/'goal_correction.npz'),
        risk_sha256=hashes['invariant_repair'],lambda_value=float(chosen['lambda_value']),train_snapshots=len(tr),validation_snapshots=len(vr),validation_key=list(best)))
    # Reuse tested native episode runner with this explicit local policy injection.
    pilot.decide=decision
    def run(geo,arm,seed,steps,section):
        r,z=pilot.episode(geo,arm,seed,models,steps);p=out/section/f"{geo['name']}_{arm}_{seed}";p.mkdir(parents=True)
        np.savez_compressed(p/'trajectory.npz',**z);r['trajectory_sha256']=sr.sha(p/'trajectory.npz');sr.write(p/'result.json',r);return r
    smoke=dict(name='smoke',starts=[[-.7,-.3],[-.3,.7]],goals=[[.7,.3],[.3,-.7]],velocity=[[.1,.04],[.04,-.1]],directions=[[.919145,.393919],[.393919,-.919145]])
    sr.write(out/'smoke_passed.json',dict(results=[run(smoke,arm,2026159999,5,'smoke') for arm in protocol['arms']]))
    rows=[]
    for geo in geos:
        for seed in protocol['seeds']:
            for arm in protocol['arms']:
                if time.perf_counter()-started>900:raise RuntimeError('Budget reached; preserve partial run')
                rows.append(run(geo,arm,seed,200,'episodes'));sr.write(out/'episode_results.json',rows);print('EPISODE',len(rows),'/48',arm,rows[-1]['safe_joint_success'],flush=True)
    groups={}
    for arm in protocol['arms']:
        scenes=[]
        for geo in geos:
            rr=[r for r in rows if r['arm']==arm and r['geometry']==geo['name']]
            scenes.append(dict(geometry=geo['name'],**{k:float(np.mean([r[k] for r in rr])) for k in ['joint_success','safe_joint_success','substep_loss_of_separation','substep_collision_proxy','steps','mean_planning_ms']}))
        groups[arm]=dict(scenes=scenes,mean={k:float(np.mean([r[k] for r in scenes])) for k in scenes[0] if k!='geometry'})
    _,after=pilot.load_models(a.baseline,a.repair);assert after==hashes;freeze=json.loads((out/'selection_frozen.json').read_text());assert sr.sha(out/'goal_correction.npz')==freeze['correction_sha256']
    old=groups['invariant_repair']['mean'];new=groups['ridge_goal']['mean']
    sr.write(out/'summary.json',dict(methods=groups,episodes=48,gate_passed=bool(new['safe_joint_success']>old['safe_joint_success'] and new['substep_loss_of_separation']<=old['substep_loss_of_separation']),
        model_hashes_unchanged=True,wall_seconds=time.perf_counter()-started))
    sr.write(out/'CONFIRMATION_CONSUMED.json',dict(reason='Frozen goal-cost repair evaluated; no tuning on these6 geometries',selection_sha256=sr.sha(out/'selection_frozen.json')))
    print('COMPLETE',json.dumps({k:v['mean'] for k,v in groups.items()}),flush=True)
if __name__=='__main__':main()
