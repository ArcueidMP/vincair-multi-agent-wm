"""Shared, planning-time-only features for the E1 pilot ranker."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT=Path('/mnt/d/资料库/03_开发项目/air traffic/experiments/lewm')
sys.path.insert(0,str(ROOT/'results/readout_intake_20261023/upstream_source'))
from readout import load_probe

SCALE=100.0
POSE_SCALE=np.array([512,512,512,512,1,1],dtype=np.float32)
def physical_cost(p,g):
    p,g=np.asarray(p,dtype=np.float64),np.asarray(g,dtype=np.float64)
    angle=np.arctan2(p[...,4],p[...,5])-np.arctan2(g[...,4],g[...,5])
    return np.linalg.norm(p[...,:4]-g[...,:4],axis=-1)+20/np.radians(20)*abs(np.arctan2(np.sin(angle),np.cos(angle)))

@torch.no_grad()
def planning_features(latents,goal_latent,actions,probe):
    """Only predicted latents, candidate actions, and encoded goal enter here."""
    if not torch.is_tensor(latents): latents=torch.as_tensor(latents,device='cuda')
    if not torch.is_tensor(goal_latent): goal_latent=torch.as_tensor(goal_latent,device='cuda')
    n=len(latents)
    goal_latent=goal_latent.reshape(1,-1)
    pp=probe(latents[:,-1].float()).cpu().numpy()
    pg=probe(goal_latent.float()).cpu().numpy()[0]
    readout=physical_cost(pp,pg)
    latent=(latents[:,-1].float()-goal_latent.float()).square().sum(-1).cpu().numpy()
    x=np.concatenate([latents.float().cpu().numpy().reshape(n,-1),
        np.broadcast_to(goal_latent.float().cpu().numpy(),(n,goal_latent.shape[-1])),
        np.asarray(actions).reshape(n,-1),pp/POSE_SCALE,
        np.broadcast_to(pg/POSE_SCALE,(n,6)),readout[:,None]/SCALE],axis=1).astype(np.float32)
    assert np.isfinite(x).all()
    return x,readout,latent

class Ranker(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def rank_metrics(score,truth):
    from scipy.stats import spearmanr
    score,truth=np.asarray(score),np.asarray(truth)
    i,j=np.triu_indices(len(score),1);diff=truth[j]-truth[i];mask=abs(diff)>1
    direction=np.sign(score[j]-score[i])*np.sign(diff)
    return dict(PRA=float(np.where(direction[mask]>0,1,np.where(direction[mask]==0,.5,0)).mean()) if mask.any() else None,
        spearman=float(spearmanr(score,truth).statistic) if np.ptp(score)>1e-10 and np.ptp(truth)>1e-10 else None,
        regret=float(truth[np.argmin(score)]-truth.min()),valid_pairs=int(mask.sum()))

def per_scene(rows,score,truth):
    records=[]
    for row,s,t in zip(rows,score,truth):records.append(dict(scene=row['seed'],name=row['name'],**rank_metrics(s,t)))
    result=[]
    for seed in sorted({r['scene'] for r in records}):
        group=[r for r in records if r['scene']==seed]
        q=dict(seed=seed,n_states=len(group))
        for key in ('PRA','spearman','regret'):
            values=[r[key] for r in group if r[key] is not None]
            q[key]=float(np.mean(values)) if values else None
        result.append(q)
    return result,records

def describe(values):
    values=np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(values.mean()),sd=float(values.std(ddof=1)) if len(values)>1 else None,
        median=float(np.median(values)),min=float(values.min()),max=float(values.max()))
