"""Shared per-agent numerical tokens; no T object or agent identity embeddings."""
import torch
from torch import nn
from module import SIGReg

class NavigationJEPA(nn.Module):
    def __init__(self,history=3,dim=64,latent=32,enc_layers=1,pred_layers=2,heads=4,sigreg_projections=64):
        super().__init__();self.history=history
        self.embed=nn.Sequential(nn.Linear(4,dim),nn.GELU(),nn.Linear(dim,dim))
        layer=lambda:nn.TransformerEncoderLayer(dim,heads,4*dim,dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer(),enc_layers,enable_nested_tensor=False)
        self.proj=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,latent));self.unproj=nn.Linear(latent,dim)
        self.act=nn.Sequential(nn.Linear(2,dim),nn.GELU(),nn.Linear(dim,dim));self.time=nn.Parameter(torch.randn(history,dim)*.02)
        self.predictor=nn.TransformerEncoder(layer(),pred_layers,enable_nested_tensor=False)
        self.next=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,latent));self.state_head=nn.Sequential(nn.Linear(latent,64),nn.GELU(),nn.Linear(64,4))
        self.regularizer=SIGReg(num_proj=sigreg_projections)

    def encode(self,state):
        if state.shape[-1]!=4:raise ValueError('Expected per-agent [x,y,vx,vy]')
        lead=state.shape[:-2];n=state.shape[-2];x=self.embed(state.reshape(-1,n,4));return self.proj(self.encoder(x)).reshape(*lead,n,-1)

    def predict(self,history,actions):
        b,h,n,_=history.shape
        if h>self.history or actions.shape!=(b,h,n,2):raise ValueError('History/action shape mismatch')
        x=self.unproj(history)+self.act(actions)+self.time[-h:,None]
        frame=torch.arange(h,device=x.device).repeat_interleave(n);mask=frame[None,:]>frame[:,None]
        out=self.predictor(x.reshape(b,h*n,-1),mask=mask).reshape(b,h,n,-1)
        return self.next(out[:,-1])

    def rollout(self,z_history,past_actions,future_actions):
        z=z_history;a=past_actions;out=[]
        for t in range(future_actions.shape[1]):
            a=torch.cat([a,future_actions[:,t:t+1]],1);pred=self.predict(z[:,-self.history:],a[:,-self.history:])
            out.append(pred);z=torch.cat([z,pred[:,None]],1)
        return torch.stack(out,1)

    def loss(self,states,actions,delta_std):
        h=self.history;k=states.shape[1]-h;z=self.encode(states)
        pred=self.rollout(z[:,:h],actions[:,:h-1],actions[:,h-1:h-1+k]);target=z[:,h:]
        latent=(pred-target).square().mean();regularizer=self.regularizer(z.reshape(1,-1,z.shape[-1]))
        encoded=self.state_head(z);decoded=self.state_head(pred)
        reconstruction=(encoded-states).square().mean();physical=(decoded-states[:,h:]).square().mean()
        previous=torch.cat([encoded[:,h-1:h],decoded[:,:-1]],1)
        delta=((decoded-previous-(states[:,h:]-states[:,h-1:-1]))/delta_std).square().mean()
        total=latent+.09*regularizer+reconstruction+physical+.1*delta
        return total,dict(latent=float(latent.detach()),sigreg=float(regularizer.detach()),encoded_head=float(reconstruction.detach()),predicted_head=float(physical.detach()),delta=float(delta.detach()))
