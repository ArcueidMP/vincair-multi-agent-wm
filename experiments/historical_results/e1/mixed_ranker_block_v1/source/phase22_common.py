"""Frozen MultiPushT mixed-pool feature contract and GPU collector engine."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from torch import nn
from e1_ranker_common import ROOT,physical_cost,load_probe,planning_features
CODE=Path('/home/dev/projects/lewm-experiments');sys.path.insert(0,str(CODE))
os.environ['STABLEWM_HOME']=str(ROOT/'data');os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
OUT=ROOT/'results/e1/mixed_ranker_block_v1'
ENV=dict(n_agents=2,others='distinct',agent_force=1e5,solver_iterations=50,jpeg_quality=95)
ARMS=['random_step','random_block','readout_CEM','privileged_contact_block']
FEATURE_NAMES=['agent_dx/512','agent_dy/512','T_dx/512','T_dy/512','goal_angle_sin','goal_angle_cos',
 'readout_cost/100','joint_position_error/100','goal_abs_angle/pi']+[f'action_mean_{i}' for i in range(4)]+[f'action_std_{i}' for i in range(4)]+['pred_T_dx/64','pred_T_dy/64','pred_rotation_sin','pred_rotation_cos']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def body(core):return np.array([[*b.position,*b.velocity,b.angle,b.angular_velocity] for b in [*core.agents,*core.drives,core.block]])
def pose(s):return np.concatenate([s[...,:2],s[...,4:6],np.sin(s[...,6,None]),np.cos(s[...,6,None])],axis=-1)
def tcost(p,g):
    a=np.arctan2(p[...,4],p[...,5])-np.arctan2(g[...,4],g[...,5])
    return np.linalg.norm(p[...,2:4]-g[...,2:4],axis=-1)+20/np.radians(20)*abs(np.arctan2(np.sin(a),np.cos(a)))
def low_features(pp,pg,p0,actions):
    # Contract accepts prediction/goal/action only. No state, contact or future label argument.
    pp,pg,p0=np.asarray(pp,np.float64),np.asarray(pg,np.float64),np.asarray(p0,np.float64)
    d=pp[:,:4]-pg[:4];a=np.arctan2(pp[:,4],pp[:,5])-np.arctan2(pg[4],pg[5]);a=np.arctan2(np.sin(a),np.cos(a))
    rot=np.arctan2(pp[:,4],pp[:,5])-np.arctan2(p0[4],p0[5])
    act=actions.reshape(len(actions),10,4).astype(np.float64)
    x=np.column_stack([d/512,np.sin(a),np.cos(a),physical_cost(pp,pg)/100,np.linalg.norm(d,axis=1)/100,abs(a)/np.pi,
        act.mean(1),act.std(1),(pp[:,2:4]-p0[2:4])/64,np.sin(rot),np.cos(rot)])
    assert x.shape==(len(actions),21) and np.isfinite(x).all()
    return x.astype(np.float32)
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(21,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return .25*torch.tanh(self.net(x).squeeze(-1))
class Engine:
    def __init__(self):
        os.chdir(CODE)
        import stable_worldmodel as swm
        from utils import pusht_stats
        self.model=swm.wm.utils.load_pretrained('lewm_coop2_joint').cuda().eval().requires_grad_(False)
        self.probe,_=load_probe(ROOT/'data/checkpoints/lewm_coop2_joint/readout/cost_probe.pt','cuda')
        mu,sd=pusht_stats('action',2);self.mu=np.asarray(mu).reshape(2,2);self.sd=np.asarray(sd).reshape(2,2)
        self.lo=(-1-self.mu)/self.sd;self.hi=(1-self.mu)/self.sd
        self.im=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.istd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    @torch.no_grad()
    def encode(self,images):
        out=[]
        for i in range(0,len(images),16):
            x=torch.from_numpy(np.stack(images[i:i+16])).cuda().permute(0,3,1,2).float()/255
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out.append(self.model.projector(self.model.encoder((x-self.im)/self.istd,interpolate_pos_encoding=True).last_hidden_state[:,0]).float())
        return torch.cat(out)
    @torch.no_grad()
    def predict(self,z0,actions):
        n=len(actions);a=torch.from_numpy(((actions-self.mu)/self.sd).reshape(n,2,20).astype(np.float32)).cuda()
        emb=z0.expand(n,-1)[:,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for t in range(2):
                ae=self.model.action_encoder(a[:,:t+1]);emb=torch.cat([emb,self.model.predict(emb[:,-3:],ae[:,-3:])[:,-1:].float()],1)
        return emb
    @torch.no_grad()
    def decode(self,z):return self.probe(z.float()).cpu().numpy()
    def pools(self,z0,decoded_goal,seed,t,direction):
        pools=[]
        for arm,chunks in enumerate([10,2],1):
            rng=np.random.default_rng([seed,t,arm]);a=np.clip(rng.standard_normal((16,chunks,2,2))*self.sd+self.mu,-1,1)
            pools.append(np.repeat(a,10//chunks,axis=1).astype(np.float32))
        rng=np.random.default_rng([seed,t,3]);mean=np.zeros((10,2,2));std=np.ones_like(mean);trace=[]
        for it in range(3):
            c=np.clip(mean[None]+std[None]*rng.standard_normal((16,10,2,2)),self.lo,self.hi);c[0]=mean
            a=(c*self.sd+self.mu).astype(np.float32);z=self.predict(z0,a)
            score=physical_cost(self.decode(z[:,-1]),decoded_goal)
            elite=c[np.argsort(score)[:4]];mean=elite.mean(0);std=np.maximum(elite.std(0),.05)
            trace.append(dict(actions=a,latents=z.cpu().numpy(),score=score))
        pools.append(a)
        rng=np.random.default_rng([seed,t,4]);speed=rng.uniform(0,.3,(16,2,2));angle=rng.uniform(-np.arctan(.25),np.arctan(.25),(16,2,2))
        perp=np.stack([-direction[:,1],direction[:,0]],-1)
        pools.append(np.repeat(speed[...,None]*(np.cos(angle)[...,None]*direction+np.sin(angle)[...,None]*perp),5,axis=1).astype(np.float32))
        return np.concatenate(pools),trace
