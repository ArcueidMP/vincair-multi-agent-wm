"""Fresh, bounded direct-scorer repair; no Navigation world model is used."""
import argparse, copy, datetime, json, time, types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import conflict_ranking as cr
import reproduce_baselines as base
import safety_generalization as sg
import state_action_ranker as sr

def invariant_features(state, goals, actions):
    """Center positions and align inter-agent displacement; average both agent orders."""
    views=[]
    for order in ([0,1],[1,0]):
        s=np.asarray(state)[order].copy();g=np.asarray(goals)[order];a=np.asarray(actions)[:,:,order,:]
        delta=s[1,:2]-s[0,:2];length=np.linalg.norm(delta)
        if length<1e-6:raise ValueError('Coincident agents have no canonical frame')
        u=delta/length;rotation=np.array([[u[0],-u[1]],[u[1],u[0]]],np.float32)
        center=s[:,:2].mean(0);s[:,:2]=(s[:,:2]-center)@rotation;s[:,2:]=s[:,2:]@rotation
        # Rotation can exceed component limits in this coordinate frame; validate original actions only.
        sr.features(state,goals,actions)
        views.append(np.concatenate([np.repeat(s.reshape(1,8),len(a),0),
            np.repeat(((g-center)@rotation-s[:,:2]).reshape(1,4),len(a),0),(a@rotation).reshape(len(a),32)],1))
    return np.stack(views,1).astype(np.float32)

def proposals(state, actions):
    brake=np.clip(-np.asarray(state)[:,2:]/.1,-1,1)
    extra=np.zeros((4,8,2,2),np.float32)
    # Fixed open-loop controls, not a state-feedback controller or certified stop.
    extra[1]=brake;extra[2,:,0]=brake[0];extra[3,:,1]=brake[1]
    return np.concatenate([actions,extra])

def rollout(ns, initial, actions):
    env=ns['create_scenario'](len(actions));ns['set_cem_state'](env,initial[:,:2],initial[:,2:])
    world=env.scenario.world;original=world._integrate_state;trace=[]
    def integration(w,entity,substep):
        original(entity,substep)
        if entity is w.agents[-1]:trace.append(base.state(env).detach().clone())
    world._integrate_state=types.MethodType(integration,world);states=[base.state(env).detach().clone()]
    with torch.no_grad():
        for t in range(8):env.step([actions[:,t,i] for i in range(2)]);states.append(base.state(env).detach().clone())
    return torch.stack(states,1),torch.stack(trace,1)

def geometries():
    rng=np.random.default_rng(2026104701);rows=[]
    for split,n in [('train',24),('validation',8),('confirmation',12)]:
        for i in range(n):
            angle=float(rng.uniform(40,140));theta=float(rng.uniform(-np.pi,np.pi));center=rng.uniform(-.08,.08,2)
            d=np.array([[np.cos(theta),np.sin(theta)],[np.cos(theta+np.radians(angle)),np.sin(theta+np.radians(angle))]],np.float32)
            rows.append(dict(base_id=f'{split}_{i:02}',split=split,angle_deg=angle,orientation_rad=theta,
                center=center.tolist(),directions=d.tolist(),starts=(center-d).tolist(),goals=(center+d).tolist()))
    return rows

def choose(raw,cm,cs,threshold,margin,consistent=True):
    risk=1/(1+np.exp(-np.clip(raw[...,0],-50,50)));cost=raw[...,1]*cs+cm;sep=raw[...,2]*.30
    result=[];fallback=[]
    for r,c,s in zip(risk,cost,sep):
        valid=np.flatnonzero((r<threshold)&((s>=.30+margin) if consistent else True))
        if len(valid):i=int(valid[c[valid].argmin()]);fallback.append(False)
        else:
            # Conservative agreement heuristic: minimize risk, then maximize predicted separation.
            # It is evaluated empirically and carries no safety guarantee.
            i=int(np.lexsort((np.arange(len(c)),c,-s,r))[0]);fallback.append(True)
        result.append(i)
    return np.array(result),np.array(fallback)

def metric(rows,chosen,cost,minimum):
    records=[]
    for row,i,c,s in zip(rows,chosen,cost,minimum):
        records.append(dict(name=row['name'],base_id=row['base_id'],condition=row['condition'],selected=int(i),
            safe_available=bool((s>=.30).any()),unsafe=bool(s[i]<.30),regret=float(c[i]-c.min())))
    scenes=[]
    for gid in sorted({r['base_id'] for r in records}):
        g=[r for r in records if r['base_id']==gid];eligible=[r for r in g if r['safe_available']]
        scenes.append(dict(base_id=gid,unsafe_given_safe=float(np.mean([r['unsafe'] for r in eligible])) if eligible else None,
            unsafe_all=float(np.mean([r['unsafe'] for r in g])),regret=float(np.mean([r['regret'] for r in g]))))
    means={k:float(np.mean([s[k] for s in scenes if s[k] is not None])) if any(s[k] is not None for s in scenes) else None for k in ['unsafe_given_safe','unsafe_all','regret']}
    return dict(mean=means,scenes=scenes,records=records)

def load(root,split,kind):
    if split=='confirmation' and not (root/'selection_frozen.json').exists():raise RuntimeError('Freeze before confirmation access')
    rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']==split];xx=[];yy=[];ss=[]
    for r in rows:
        p=root/'data'/r['name'];assert sr.sha(p/'inputs.npz')==r['input_sha256'] and sr.sha(p/'labels.npz')==r['labels_sha256']
        with np.load(p/'inputs.npz') as z:
            x=invariant_features(z['initial_state'],z['goals'],z['actions']) if kind=='invariant' else sr.features(z['initial_state'],z['goals'],z['actions'])[:,None]
        with np.load(p/'labels.npz') as z:yy.append(z['true_cost']);ss.append(z['min_separation'])
        xx.append(x)
    return rows,np.stack(xx),np.stack(yy),np.stack(ss)

def infer(net,x,mean,std):
    with torch.no_grad():return net(torch.from_numpy((x-mean)/std)).mean(-2).numpy()

def fit(root,kind):
    rows,x,y,s=load(root,'train',kind);vr,vx,vy,vs=load(root,'validation',kind)
    mean=x.reshape(-1,44).mean(0);std=np.maximum(x.reshape(-1,44).std(0),.01);cm=float(y.mean());cs=float(max(y.std(),.01))
    X=torch.from_numpy((x-mean)/std);U=torch.from_numpy((s<.30).astype(np.float32));Y=torch.from_numpy(((y-cm)/cs).astype(np.float32));S=torch.from_numpy((s/.30).astype(np.float32))
    torch.manual_seed(0);net=sr.Scorer();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    pw=(U.numel()-U.sum())/U.sum();best=None;history=[];wait=0
    for epoch in range(1,201):
        for b in torch.randperm(len(X)).split(8):
            raw=net(X[b]).mean(-2);loss=F.binary_cross_entropy_with_logits(raw[...,0],U[b],pos_weight=pw)+.1*F.mse_loss(raw[...,1],Y[b])+F.mse_loss(raw[...,2],S[b])
            opt.zero_grad();loss.backward();opt.step()
        raw=infer(net,vx,mean,std);candidates=[]
        for threshold in [.1,.25,.5]:
            for margin in [0.,.03,.06]:
                idx,_=choose(raw,cm,cs,threshold,margin);m=metric(vr,idx,vy,vs)['mean']
                candidates.append(((m['unsafe_given_safe'],m['regret'],epoch,threshold,margin),threshold,margin))
        key,threshold,margin=min(candidates);history.append(dict(epoch=epoch,key=list(key)))
        if best is None or key<best:
            best=key;state=copy.deepcopy(net.state_dict());policy=dict(threshold=threshold,margin=margin);wait=0
        else:wait+=1
        if wait>=30:break
    net.load_state_dict(state);checkpoint=dict(state_dict=state,mean=mean,std=std,cost_mean=cm,cost_std=cs,epoch=best[2],policy=policy,kind=kind)
    torch.save(checkpoint,root/f'{kind}.pt');sr.write(root/f'{kind}_history.json',history)
    portable=dict(mean=mean,std=std,cost_mean=np.array(cm),cost_std=np.array(cs))
    portable.update({k.replace('.','_'):v.numpy() for k,v in state.items()});np.savez(root/f'{kind}_portable.npz',**portable)
    return checkpoint

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Preserve previous run')
    torch.set_num_threads(1);start=time.perf_counter();geos=geometries()
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sr.sha(Path(__file__)),
        kind='Direct current-state/action scorer repair; NOT LeWM or closed-loop',geometries=geos,
        split='24 train /8 validation /12 confirmation whole new geometries; 6 states per geometry',
        prior_evaluation='Neither consumed old test nor phase26 stress data read for fitting or selection',
        candidates='Original64 unchanged, append4 fixed open-loop zero/brake-both/brake0/brake1; H8',
        native='VMAS1.5.2 dt.1 substeps2 radius.1 drag.25; safe separation.30 incl initial and native substeps',
        models=['fresh raw44 MLP control','centered/aligned44 MLP with two agent-order predictions averaged'],
        training='seed0 CPU AdamW lr.001 wd.01 max200 patience30 group batch8; train-only moments; BCE class-weight+.1costMSE+sepMSE',
        selection='Validation geometry-mean conditional unsafe, then regret, epoch, threshold, margin; threshold[.1,.25,.5], margin[0,.03,.06]',
        policy='Risk<threshold AND predicted minsep>=.30+margin; else lowest risk then highest predicted minsep, cost, index. No guarantee.',
        gate='Invariant repair original64 vs frozen original baseline original64: >=10pp conditional unsafe reduction and >=8/12 geometry wins; descriptive only. No closed-loop in this block even if passed.',
        limitations=['Single training seed','Synthetic route distributions shared across splits','Proposal and selector ablations descriptive, not causal isolation of every component'])
    sr.write(out/'protocol.json',protocol);manifest=[];maxerror=0.
    for gi,geo in enumerate(geos):
        ns=cr.namespace('S1_symmetric',geo['goals']);scene=dict(starts=geo['starts'],goals=geo['goals'])
        for di,dist in enumerate([.34,.45]):
            original=ns['generate_conflict_actions'](scene,64,8,2026120000+gi*2+di).cpu().numpy()
            for condition in sg.CONDITIONS:
                initial=sg.initial_state(geo,condition,dist);actions=proposals(initial,original);at=torch.tensor(actions,device='cuda')
                states,sub=rollout(ns,initial,at);again,subagain=rollout(ns,initial,at)
                err=max(float((states-again).abs().max()),float((sub-subagain).abs().max()));maxerror=max(maxerror,err);assert err==0
                cost=ns['calculate_ranking_labels'](states,at,np.array(geo['goals'],np.float32))['total_cost']
                allsub=torch.cat([states[:,:1],sub],1);minimum=torch.linalg.vector_norm(allsub[:,:,0,:2]-allsub[:,:,1,:2],dim=-1).min(1).values.cpu().numpy()
                name=f"{geo['base_id']}_{condition}_d{di}";p=out/'data'/name;p.mkdir(parents=True)
                np.savez_compressed(p/'inputs.npz',initial_state=initial,goals=np.array(geo['goals'],np.float32),actions=actions)
                np.savez_compressed(p/'labels.npz',state_path=states.cpu().numpy(),substep_state_path=sub.cpu().numpy(),true_cost=cost,min_separation=minimum)
                manifest.append(dict(name=name,base_id=geo['base_id'],split=geo['split'],condition=condition,input_sha256=sr.sha(p/'inputs.npz'),labels_sha256=sr.sha(p/'labels.npz')))
        print('GEOMETRY',gi+1,'/44',flush=True)
    sr.write(out/'manifest.json',manifest)
    checkpoints={kind:fit(out,kind) for kind in ['raw','invariant']}
    baselinepath=a.baseline/'state_action_scorer.pt';frozen=json.loads((a.baseline/'selection_frozen.json').read_text());assert sr.sha(baselinepath)==frozen['checkpoint_sha256']
    sr.write(out/'selection_frozen.json',dict(status='frozen',protocol_sha256=sr.sha(out/'protocol.json'),baseline_sha256=sr.sha(baselinepath),
        models={k:dict(checkpoint_sha256=sr.sha(out/f'{k}.pt'),epoch=c['epoch'],policy=c['policy']) for k,c in checkpoints.items()}))
    rows,rx,y,s=load(out,'confirmation','raw');_,ix,_,_=load(out,'confirmation','invariant')
    old=torch.load(baselinepath,map_location='cpu',weights_only=False);net=sr.Scorer();net.load_state_dict(old['state_dict'])
    rawold=infer(net,rx,old['mean'].numpy(),old['std'].numpy());methods={};prediction=dict(old=rawold)
    for count in [64,68]:
        idx=sr.select(rawold[:,:count],old['cost_mean'],old['cost_std'],old['threshold'])[0]
        methods[f'old_safety_{count}']=metric(rows,idx,y[:,:count],s[:,:count])
    for kind,x in [('raw',rx),('invariant',ix)]:
        c=checkpoints[kind];net.load_state_dict(c['state_dict']);raw=infer(net,x,c['mean'],c['std']);prediction[kind]=raw
        for count in [64,68]:
            idx,fb=choose(raw[:,:count],c['cost_mean'],c['cost_std'],**c['policy']);m=metric(rows,idx,y[:,:count],s[:,:count]);m['fallback_groups']=int(fb.sum());methods[f'{kind}_consistent_{count}']=m
        idx=sr.select(raw[:,:64],c['cost_mean'],c['cost_std'],c['policy']['threshold'])[0]
        methods[f'{kind}_legacy_selector_64']=metric(rows,idx,y[:,:64],s[:,:64])
    np.savez_compressed(out/'confirmation_predictions.npz',**prediction)
    control=methods['old_safety_64'];repair=methods['invariant_consistent_64']
    wins=sum(b['unsafe_given_safe']<a['unsafe_given_safe'] for a,b in zip(control['scenes'],repair['scenes']) if a['unsafe_given_safe'] is not None and b['unsafe_given_safe'] is not None)
    reduction=control['mean']['unsafe_given_safe']-repair['mean']['unsafe_given_safe']
    result=dict(methods=methods,geometry_wins=wins,unsafe_reduction=reduction,gate_passed=bool(reduction>=.10 and wins>=8),
        groups=len(rows),candidates_total=len(manifest)*68,max_native_replay_error=maxerror,wall_seconds=time.perf_counter()-start,
        confirmation_consumed=True,no_closed_loop=True,no_world_model=True)
    sr.write(out/'summary.json',result);sr.write(out/'CONFIRMATION_CONSUMED.json',dict(reason='One frozen repair evaluation; never tune on these geometries',selection_sha256=sr.sha(out/'selection_frozen.json')))
    print('COMPLETE',json.dumps({k:v['mean'] for k,v in methods.items()}),flush=True)
if __name__=='__main__':main()
