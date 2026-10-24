"""Training-scene-only small scorer selection and frozen final refit."""
import time,datetime,copy,shutil
import numpy as np
import torch
from torch.nn import functional as F
from phase22_common import *
torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
protocol=json.loads((OUT/'protocol.json').read_text());assert not (OUT/'selection_frozen.json').exists()
assert not (OUT/'confirmation_data/manifest.json').exists(),'Confirmation must not exist during model selection'
data=OUT/'train_data';assert json.loads((data/'metadata.json').read_text())['status']=='passed'
trainmanifest=json.loads((data/'manifest.json').read_text());rows=[];xx=[];yy=[];rr=[];ty=[];ts=[];access=[]
for scene in trainmanifest:
    assert scene['seed'] in protocol['train_seeds']
    for g in scene['groups']:
        directory=data/g['name']
        assert sha(directory/'inputs.npz')==g['input_sha256'] and sha(directory/'labels.npz')==g['labels_sha256']
        with np.load(directory/'inputs.npz') as z:
            xx.append(z['features']);rr.append(z['predicted_cost']);assert z['features'].shape==(64,21)
        with np.load(directory/'labels.npz') as z:
            yy.append(z['true_cost']);ty.append(z['true_T_cost']);ts.append(float(z['true_T_cost_start']))
        rows.append(dict(scene=scene['scene'],seed=scene['seed'],name=g['name'],t=g['t']))
        access.append(dict(split='train',group=g['name'],feature_keys=['features','predicted_cost'],target_keys=['true_cost','true_T_cost','true_T_cost_start']))
x=np.stack(xx);y=np.stack(yy);r=np.stack(rr);T=np.stack(ty);Tstart=np.array(ts)
assert x.shape==(48,64,21) and np.isfinite(x).all() and np.isfinite(y).all()
fit_ids=np.array([q['scene']<18 for q in rows]);inner_ids=~fit_ids
assert len({q['seed'] for q,b in zip(rows,fit_ids) if b})==18
assert len({q['seed'] for q,b in zip(rows,inner_ids) if b})==6
def normalization(x):return x.reshape(-1,21).mean(0),np.maximum(x.reshape(-1,21).std(0),.01)
def metrics(scores,ids):
    ids=np.flatnonzero(ids);records=[]
    for k,s in zip(ids,scores):
        truth=y[k];idx=int(s.argmin());i,j=np.triu_indices(64,1);d=truth[j]-truth[i];mask=abs(d)>1
        direction=np.sign(s[j]-s[i])*np.sign(d)
        records.append(dict(seed=rows[k]['seed'],name=rows[k]['name'],regret=float(truth[idx]-truth.min()),
            PRA=float(np.where(direction[mask]>0,1,np.where(direction[mask]==0,.5,0)).mean()) if mask.any() else None,
            T_gain=float(Tstart[k]-T[k,idx]),selected_index=idx))
    scenes=[]
    for seed in sorted({q['seed'] for q in records}):
        g=[q for q in records if q['seed']==seed]
        scenes.append(dict(seed=seed,**{key:float(np.mean([q[key] for q in g if q[key] is not None])) for key in ('regret','PRA','T_gain')}))
    return dict(mean={key:float(np.mean([q[key] for q in scenes])) for key in ('regret','PRA','T_gain')},scenes=scenes,groups=records)
def ridge(nx,target,base,lam):
    xc=nx-nx.mean(1,keepdims=True);res=(target-base)/100;res-=res.mean(1,keepdims=True)
    a=xc.reshape(-1,21).astype(np.float64);b=res.reshape(-1)
    return np.linalg.solve(a.T@a/len(a)+lam*np.eye(21),a.T@b/len(a))
mean,std=normalization(x[fit_ids]);nx=(x-mean)/std
candidates=[dict(name='zero_correction',kind='zero',priority=0,metrics=metrics(r[inner_ids]/100,inner_ids))]
fit_models={}
for lam,priority in [(1.,1),(.01,2)]:
    beta=ridge(nx[fit_ids],y[fit_ids],r[fit_ids],lam);score=r[inner_ids]/100+np.clip(nx[inner_ids]@beta,-.25,.25)
    name=f'ridge_lambda_{lam:g}';fit_models[name]=dict(beta=beta,mean=mean,std=std)
    candidates.append(dict(name=name,kind='ridge',lam=lam,priority=priority,metrics=metrics(score,inner_ids)))
net=ResidualMLP();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
X=torch.from_numpy(nx[fit_ids]);Y=torch.from_numpy((y[fit_ids]/100).astype(np.float32));R=torch.from_numpy((r[fit_ids]/100).astype(np.float32))
V=torch.from_numpy(nx[inner_ids]);i,j=torch.triu_indices(64,64,1);history=[];best=float('inf');best_state=None;wait=0
started=time.perf_counter()
for epoch in range(1,101):
    net.train();losses=[];pairs=[];mses=[]
    for batch in torch.randperm(len(X)).split(8):
        score=R[batch]+net(X[batch]);target=Y[batch];diff=target[:,j]-target[:,i];mask=diff.abs()>.01
        pair=F.softplus(diff.sign()*(score[:,i]-score[:,j]))
        pairloss=pair[mask].mean() if mask.any() else score.sum()*0
        mse=F.mse_loss(score,target);loss=pairloss+.1*mse
        opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()));pairs.append(float(pairloss.detach()));mses.append(float(mse.detach()))
    net.eval()
    with torch.no_grad():valscore=r[inner_ids]/100+net(V).numpy()
    val=metrics(valscore,inner_ids);regret=val['mean']['regret']
    history.append(dict(epoch=epoch,loss=float(np.mean(losses)),pairwise_softplus=float(np.mean(pairs)),scaled_MSE=float(np.mean(mses)),inner=val['mean']))
    if regret<best-1e-8:best=regret;best_state=copy.deepcopy(net.state_dict());best_epoch=epoch;best_metrics=val;wait=0
    else:wait+=1
    if epoch%10==0:print('EPOCH',epoch,regret,flush=True)
    if wait>=20:break
assert best_state is not None
fit_models['residual_mlp_21_32_1']=dict(state_dict=best_state,mean=mean,std=std)
candidates.append(dict(name='residual_mlp_21_32_1',kind='mlp',epochs=best_epoch,priority=3,metrics=best_metrics))
baseline=candidates[0]['metrics']['mean']
for q in candidates:
    q['inner_relative_improvement']=float((baseline['regret']-q['metrics']['mean']['regret'])/max(baseline['regret'],1e-8))
    q['eligible']=q['kind']=='zero' or (q['inner_relative_improvement']>=.05 and q['metrics']['mean']['T_gain']>=baseline['T_gain']-1)
chosen=candidates[0]
for q in candidates[1:]:
    if q['eligible'] and q['metrics']['mean']['regret']<chosen['metrics']['mean']['regret']-1e-8:chosen=q
torch.save(fit_models,OUT/'inner_fitted_models.pt')
# Refit with all24 training scenes; no confirmation feature or target access.
mean,std=normalization(x);nx=(x-mean)/std
ck=dict(kind=chosen['kind'],name=chosen['name'],mean=torch.from_numpy(mean),std=torch.from_numpy(std),feature_names=FEATURE_NAMES,bound=.25,
    protocol_sha256=sha(OUT/'protocol.json'))
if chosen['kind']=='ridge':ck.update(beta=torch.from_numpy(ridge(nx,y,r,chosen['lam'])),lam=chosen['lam'])
elif chosen['kind']=='mlp':
    torch.manual_seed(0);finalnet=ResidualMLP();optimizer=torch.optim.AdamW(finalnet.parameters(),lr=.001,weight_decay=.01)
    XX=torch.from_numpy(nx);YY=torch.from_numpy((y/100).astype(np.float32));RR=torch.from_numpy((r/100).astype(np.float32))
    for epoch in range(chosen['epochs']):
        for batch in torch.randperm(len(XX)).split(8):
            score=RR[batch]+finalnet(XX[batch]);target=YY[batch];d=target[:,j]-target[:,i];mask=d.abs()>.01
            terms=F.softplus(d.sign()*(score[:,i]-score[:,j]));pair=terms[mask].mean() if mask.any() else score.sum()*0
            loss=pair+.1*F.mse_loss(score,target);optimizer.zero_grad();loss.backward();optimizer.step()
    ck.update(state_dict=finalnet.state_dict(),epochs=chosen['epochs'])
torch.save(ck,OUT/'selected_scorer.pt')
write(OUT/'training_history.json',history);write(OUT/'train_access_log.json',access)
frozen=dict(status='frozen',selected=chosen['name'],kind=chosen['kind'],candidates=candidates,
    checkpoint_sha256=sha(OUT/'selected_scorer.pt'),protocol_sha256=sha(OUT/'protocol.json'),
    confirmation_files_opened=0,old_test_candidate_npz_opened=0,training_wall_seconds=time.perf_counter()-started,
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(OUT/'selection_frozen.json',frozen);shutil.copy2(Path(__file__),ROOT/'scripts/phase22_v1'/Path(__file__).name)
print('SELECTION',json.dumps(dict(selected=chosen['name'],candidates=[dict(name=q['name'],eligible=q['eligible'],metrics=q['metrics']['mean']) for q in candidates])),flush=True)
