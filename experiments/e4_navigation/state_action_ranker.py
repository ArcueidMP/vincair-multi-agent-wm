"""Direct observed-state/action safety scoring baseline, NOT a LeWM adapter."""
import argparse,copy,datetime,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def features(initial_state,goals,actions):
    s,g,a=map(np.asarray,(initial_state,goals,actions))
    assert s.shape==(2,4) and g.shape==(2,2) and a.ndim==4 and a.shape[1:]==(8,2,2)
    assert np.isfinite(s).all() and np.isfinite(g).all() and np.isfinite(a).all() and abs(a).max()<=1.000001
    #8 current physical components +4 relative goals +32 proposed controls. No future input.
    x=np.concatenate([np.repeat(s.reshape(1,8),len(a),0),np.repeat((g-s[:,:2]).reshape(1,4),len(a),0),a.reshape(len(a),32)],1)
    assert x.shape==(len(a),44);return x.astype(np.float32)
class Scorer(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(44,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,x):return self.net(x)
def load(data,split,access,freeze=None):
    assert split in ('train','validation','test')
    if split=='test' and (freeze is None or not freeze.exists()):raise RuntimeError('Test access requires frozen checkpoint and policy')
    if split=='test':
        frozen=json.loads(freeze.read_text());assert frozen['status']=='frozen'
        assert sha(freeze.parent/'state_action_scorer.pt')==frozen['checkpoint_sha256']
        assert sha(freeze.parent/'protocol.json')==frozen['protocol_sha256']
    rows=[q for q in json.loads((data/'manifest.json').read_text()) if q['split']==split]
    xx=[];cost=[];minimum=[]
    for row in rows:
        p=data/split/row['name'];assert sha(p/'inputs.npz')==row['input_sha256'] and sha(p/'labels.npz')==row['labels_sha256']
        with np.load(p/'inputs.npz') as z:xx.append(features(z['initial_state'],z['goals'],z['actions']))
        with np.load(p/'labels.npz') as z:cost.append(z['total_cost']);minimum.append(z['substep_minimum_separation'])
        access.append(dict(split=split,name=row['name'],feature_keys=['initial_state','goals','actions'],supervision_keys=['total_cost','substep_minimum_separation']))
    return rows,np.stack(xx),np.stack(cost),np.stack(minimum)
def select(raw,cmean,cstd,threshold):
    risk=1/(1+np.exp(-np.clip(raw[...,0],-50,50)));cost=raw[...,1]*cstd+cmean;sep=raw[...,2]*.30
    indices=[];fallback=[]
    for p,c,s in zip(risk,cost,sep):
        valid=np.flatnonzero(p<threshold)
        if len(valid):indices.append(int(valid[c[valid].argmin()]));fallback.append(False)
        else:indices.append(int(np.lexsort((np.arange(len(c)),c,-s))[0]));fallback.append(True)
    return np.array(indices),np.array(fallback),cost,risk,sep
def evaluate(rows,raw,truth,minimum,cmean,cstd,threshold):
    selected,fallback,cost,risk,sep=select(raw,cmean,cstd,threshold)
    choices=dict(learned_cost_only=cost.argmin(1),learned_safety=selected,
        true_soft_oracle=truth.argmin(1),true_safety_oracle=np.array([np.lexsort((np.arange(len(c)),c,np.where(s<.30,-s,0),(s<.30).astype(int)))[0] for c,s in zip(truth,minimum)]))
    methods={};records=[]
    for method,idxs in choices.items():
        rec=[]
        for i,(row,idx) in enumerate(zip(rows,idxs)):
            safe=minimum[i]>=.30
            rec.append(dict(geometry_id=row['geometry_id'],name=row['name'],selected=int(idx),unsafe=bool(not safe[idx]),
                safe_available=bool(safe.any()),regret=float(truth[i,idx]-truth[i].min()),min_separation=float(minimum[i,idx])))
        scenes=[]
        for gid in sorted({q['geometry_id'] for q in rec}):
            g=[q for q in rec if q['geometry_id']==gid];eligible=[q for q in g if q['safe_available']]
            scenes.append(dict(geometry_id=gid,unsafe_given_safe=float(np.mean([q['unsafe'] for q in eligible])) if eligible else None,
                unsafe_all=float(np.mean([q['unsafe'] for q in g])),regret=float(np.mean([q['regret'] for q in g]))))
        methods[method]=dict(mean={k:float(np.mean([q[k] for q in scenes if q[k] is not None])) for k in ['unsafe_given_safe','unsafe_all','regret']},scenes=scenes)
        records.extend(dict(method=method,**q) for q in rec)
    expected=minimum<.30;predicted=risk>=threshold
    calibration=dict(unsafe_candidates=int(expected.sum()),false_safe_candidates=int((expected&~predicted).sum()),
        false_unsafe_candidates=int((~expected&predicted).sum()),predicted_no_safe_groups=int(fallback.sum()),groups=len(rows))
    return dict(methods=methods,calibration=calibration,groups=records)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists(),'Preserve previous experiment'
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);started=time.perf_counter();access=[]
    protocol=dict(status='frozen',kind='direct current-state/action MLP baseline; not LeWM, no WM checkpoint',
        architecture=[44,64,32,3],heads=['unsafe_logit','scaled_soft_cost','scaled_min_substep_separation'],parameter_count=5059,
        data_manifest_sha256=sha(args.data/'manifest.json'),source_sha256=sha(Path(__file__)),
        features='current state8 + relative goals4 + action sequence32. No observed future, true labels, oracle safety or rollout features.',
        normalization='Train-only mean/std floor.01; soft-cost mean/std train-only; separation/.30',
        seed=0,device='cpu',learning_rate=.001,weight_decay=.01,epochs_max=200,patience=30,batch_groups=8,
        loss='train-only class-balanced BCE +0.1 scaled soft-cost MSE +scaled min-separation MSE',
        risk_thresholds=[.1,.25,.5],policy='Predicted-safe candidates by predicted cost; if none, predicted minsep descending, cost ascending, stable index.',
        selection='Validation scene-mean unsafe-given-safe primary, soft regret secondary, earlier epoch, lower threshold ties. Same network cost head is a scoring-rule ablation, not independently selected competitor.',
        test_policy='Freeze checkpoint+threshold before one test activation; no test tuning/refit or closed-loop in this block.',
        predeclared_success='Test safety selection reduces conditional unsafe vs same-network cost-only by>=10 percentage points and in>=2/3 geometries. Descriptive evidence only.',
        limitations=['Three route families shared across splits.','Finite H8 dataset performance only; no LeWM transfer or closed-loop guarantee.'])
    write(out/'protocol.json',protocol)
    tr,x,y,s=load(args.data,'train',access);va,vx,vy,vs=load(args.data,'validation',access)
    mean=x.reshape(-1,44).mean(0);std=np.maximum(x.reshape(-1,44).std(0),.01);cmean=float(y.mean());cstd=float(max(y.std(),.01))
    X=torch.from_numpy((x-mean)/std);V=torch.from_numpy((vx-mean)/std)
    unsafe=torch.from_numpy((s<.30).astype(np.float32));targetcost=torch.from_numpy(((y-cmean)/cstd).astype(np.float32));targetsep=torch.from_numpy((s/.30).astype(np.float32))
    positive=float(unsafe.sum());negative=float(unsafe.numel()-positive);assert positive>0 and negative>0
    net=Scorer();assert sum(p.numel() for p in net.parameters())==5059
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);posweight=torch.tensor(negative/positive)
    best=None;beststate=None;bestthreshold=None;wait=0;history=[]
    for epoch in range(1,201):
        net.train();losses=[]
        for batch in torch.randperm(len(X)).split(8):
            raw=net(X[batch]);risk=F.binary_cross_entropy_with_logits(raw[...,0],unsafe[batch],pos_weight=posweight)
            costloss=F.mse_loss(raw[...,1],targetcost[batch]);seploss=F.mse_loss(raw[...,2],targetsep[batch]);loss=risk+.1*costloss+seploss
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():pred=net(V).numpy()
        choices=[]
        for threshold in [.1,.25,.5]:
            result=evaluate(va,pred,vy,vs,cmean,cstd,threshold);m=result['methods']['learned_safety']['mean'];choices.append((m['unsafe_given_safe'],m['regret'],threshold,result))
        winner=min(choices,key=lambda q:q[:3]);key=winner[:2]
        history.append(dict(epoch=epoch,mean_loss=float(np.mean(losses)),threshold=winner[2],validation_unsafe=key[0],validation_regret=key[1]))
        better=best is None or key[0]<best[0]-1e-10 or (abs(key[0]-best[0])<=1e-10 and key[1]<best[1]-1e-8)
        if better:best=key;beststate=copy.deepcopy(net.state_dict());bestthreshold=winner[2];bestepoch=epoch;bestvalidation=winner[3];wait=0
        else:wait+=1
        if epoch%20==0:print('EPOCH',epoch,key,flush=True)
        if wait>=30:break
    assert beststate is not None;net.load_state_dict(beststate);net.eval()
    checkpoint=dict(state_dict=beststate,mean=torch.from_numpy(mean),std=torch.from_numpy(std),cost_mean=cmean,cost_std=cstd,threshold=bestthreshold,epoch=bestepoch,
        model_kind='direct_state_action_baseline',protocol_sha256=sha(out/'protocol.json'))
    torch.save(checkpoint,out/'state_action_scorer.pt')
    arrays=dict(mean=mean,std=std,cost_mean=cmean,cost_std=cstd,threshold=bestthreshold)
    arrays.update({k.replace('.','_'):v.numpy() for k,v in beststate.items()});np.savez_compressed(out/'portable_scorer.npz',**arrays)
    write(out/'training_history.json',history);write(out/'validation_summary.json',bestvalidation)
    frozen=dict(status='frozen',selected_epoch=bestepoch,threshold=bestthreshold,checkpoint_sha256=sha(out/'state_action_scorer.pt'),
        portable_sha256=sha(out/'portable_scorer.npz'),protocol_sha256=sha(out/'protocol.json'),test_candidate_groups_opened=0,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_wall_seconds=time.perf_counter()-started)
    write(out/'selection_frozen.json',frozen);assert not any(q['split']=='test' for q in access)
    write(out/'pretest_access_log.json',access)
    # One post-freeze test activation. No training or policy changes below this point.
    te,tx,ty,ts=load(args.data,'test',access,out/'selection_frozen.json')
    with torch.no_grad():raw=net(torch.from_numpy((tx-mean)/std)).numpy()
    test=evaluate(te,raw,ty,ts,cmean,cstd,bestthreshold)
    a=test['methods']['learned_safety'];b=test['methods']['learned_cost_only'];d=[q['unsafe_given_safe']-r['unsafe_given_safe'] for q,r in zip(a['scenes'],b['scenes'])]
    improvement=b['mean']['unsafe_given_safe']-a['mean']['unsafe_given_safe'];better=sum(v<0 for v in d)
    success=improvement>=.1 and better>=2
    test.update(criterion_met=success,conditional_unsafe_reduction=improvement,better_geometries=better,
        test_consumed=True,checkpoint_sha256=sha(out/'state_action_scorer.pt'),refit_after_test=False)
    assert sha(out/'state_action_scorer.pt')==frozen['checkpoint_sha256']
    write(out/'test_summary.json',test);write(out/'access_log.json',access)
    np.savez_compressed(out/'test_predictions.npz',raw=raw,truth=ty,min_separation=ts,features=tx)
    print('COMPLETE',json.dumps(dict(epoch=bestepoch,threshold=bestthreshold,criterion_met=success,
        validation={k:v['mean'] for k,v in bestvalidation['methods'].items()},test={k:v['mean'] for k,v in test['methods'].items()})),flush=True)
if __name__=='__main__':main()
