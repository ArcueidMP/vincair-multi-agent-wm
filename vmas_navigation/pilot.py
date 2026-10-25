"""Fixed Navigation data/train/validation pilot; no claim of teammate protocol parity."""
import argparse,datetime,hashlib,json,platform,time,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from vmas_navigation import data
from vmas_navigation.model import NavigationJEPA

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def seed_training(seed):
    # VMAS CPU environments can reset global CUDA RNG. Isolate the trainer AFTER collection.
    torch.manual_seed(seed);np.random.seed(seed)
    return dict(cpu_initial_seed=torch.initial_seed(),cuda_initial_seed=torch.cuda.initial_seed())

def normalization(states,train_episodes):
    values=states[train_episodes].reshape(-1,4);mean=values.mean(0);std=values.std(0).clamp(min=.01)
    normalized=(states[train_episodes]-mean)/std;delta=(normalized[:,1:]-normalized[:,:-1]).reshape(-1,4).std(0).clamp(min=.02)
    return mean,std,delta

@torch.no_grad()
def evaluate(model,states,actions,meta,mean,std,device):
    h=model.history;k=8;ss=[];aa=[];rows=[]
    for i,r in enumerate(meta):
        if r['split']!='validation':continue
        for t in [0,20,50,80,110]:
            ss.append(states[i,t:t+h+k]);aa.append(actions[i,t:t+h+k-1]);rows.append(dict(episode_id=r['episode_id'],geometry_id=r['geometry_id'],policy=r['policy'],start=t))
    raw=torch.stack(ss).to(device);a=torch.stack(aa).to(device);s=(raw-mean)/std;model.eval();z=model.encode(s)
    predicted=model.rollout(z[:,:h],a[:,:h-1],a[:,h-1:]);decoded=model.state_head(predicted)*std+mean;truth=raw[:,h:]
    last=raw[:,h-1];persist=last[:,None].expand(-1,k,-1,-1)
    times=torch.arange(1,k+1,device=device)[None,:,None,None]*.1
    constant=torch.cat([last[:,None,:,:2]+last[:,None,:,2:]*times,last[:,None,:,2:].expand(-1,k,-1,-1)],-1)
    analytic=data.free_rollout(last,a[:,h-1:]);encoded=model.state_head(z[:,h:])*std+mean
    metrics={}
    for name,pred in [('jepa',decoded),('persistence',persist),('constant_velocity',constant),('free_analytic',analytic),('encoded_readout',encoded)]:
        pos=torch.linalg.vector_norm(pred[...,:2]-truth[...,:2],dim=-1);vel=torch.linalg.vector_norm(pred[...,2:]-truth[...,2:],dim=-1)
        actualsep=torch.linalg.vector_norm(truth[:,:,0,:2]-truth[:,:,1,:2],dim=-1);predsep=torch.linalg.vector_norm(pred[:,:,0,:2]-pred[:,:,1,:2],dim=-1)
        # Include observed initial state, matching prior initial+native-substep safety semantics at available frames.
        # This evaluates only model-step endpoints, not unobserved internal substeps.
        initialsep=torch.linalg.vector_norm(last[:,0,:2]-last[:,1,:2],dim=-1)
        actualmin=torch.minimum(initialsep,actualsep.min(1).values);predmin=torch.minimum(initialsep,predsep.min(1).values)
        metrics[name]=dict(position_L2_by_k=pos.mean((0,2)).cpu().tolist(),velocity_L2_by_k=vel.mean((0,2)).cpu().tolist(),
            endpoint_minsep_MAE=float((actualmin-predmin).abs().mean()),false_safe_endpoint_windows=int(((predmin>=.3)&(actualmin<.3)).sum()),
            unsafe_endpoint_windows=int((actualmin<.3).sum()))
    result=dict(windows=len(rows),validation_geometries=4,metrics=metrics,
        latent_prediction_MSE=float((predicted-z[:,h:]).square().mean()),latent_persistence_MSE=float((z[:,h-1:h]-z[:,h:]).square().mean()),
        encoded_latent_mean_std=float(z.reshape(-1,z.shape[-1]).std(0).mean()),head_trained_on_encoded_and_predicted_latents=True,
        scope='Fixed validation windows only; no untouched test, no planner or closed-loop claim; endpoint safety is not native-substep safety')
    arrays=dict(state_windows=raw.cpu().numpy(),actions=a.cpu().numpy(),predicted_state=decoded.cpu().numpy(),predicted_latent=predicted.cpu().numpy(),encoded_future_latent=z[:,h:].cpu().numpy(),encoded_state=encoded.cpu().numpy())
    model.train();return result,arrays,rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
    if out.exists():raise RuntimeError('Use a NEW output directory; preserve previous attempt')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);started=time.perf_counter();geos=data.layouts()
    config=dict(history=3,dim=64,latent=32,enc_layers=1,pred_layers=2,heads=4,sigreg_projections=64)
    sources={p.name:sha(p) for p in [Path(__file__),Path(data.__file__),Path(__file__).with_name('model.py')]}
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_id='LOCAL_NAV_NUMERIC_PILOT_V1',
        parent_commit='55c634c1d10ca99bad46d88cdb598f470444ed62',source_sha256=sources,geometry_specs=geos,
        teammate_alignment='Pending: collection/config/architecture commit not supplied. Local pilot, NOT identical-protocol reproduction.',
        original_plan_preserved=True,input='Centralized set of each agents own observed [x,y,vx,vy], first4 of native18D observation; full18D saved; goals metadata only, not model input',
        targets='Same physical4D observations at future frames, separated from planner inputs; shared agent head on encoded and predicted latents',
        physics='Native VMAS1.5.2 dt.1 substeps2 radius.1 mass1 world_drag.25, component action bounds1; no maxspeed, no reset at success',
        episodes=64,episode_steps=128,policies=data.MODES,split='Whole geometry12 train/4 validation;48 train/16 validation episodes. No test set in this pilot.',
        model_config=config,frameskip=1,model_dt=.1,train_rollout_k=3,validation_rollout_k=8,
        trainer=dict(seed=0,device='cuda',batch=128,steps=500,optimizer='AdamW',lr=.0003,weight_decay=.001,gradient_clip=1.),
        loss='latentMSE+.09*SIGReg64+encoded_stateMSE+predicted_stateMSE+.1*normalized_deltaMSE; both encoder/predictor targets receive gradients as existing JEPA recipe',
        selection='Fixed500-step last checkpoint; validation at step0,100,...500 for diagnostics only. No hyperparameter/model selection or untouched test claim.',
        baselines=['persistence','constant_velocity','known_free_analytic','encoded_readout'],
        pilot_gate='Pipeline: finite data/loss/gradient, legal actions, exact replay, train-only moments, end-to-end save/reload. Qualification separately requires positionL2 at k1 AND k8 < persistence.',
        limitations=['Small local architecture differs from team defaults:dim64 vs PushT128 and SIGReg64 vs1024','Numerical agent encoder is a new adaptation, not transferred image weights','No confirmed native MARL collection parity','Dynamic agent shapes/equivariance unit tests do not establish three-agent generalization'],
        budget='CollectionCPU +500 short GPU updates; wall cap900s checked per training step after collection. No full training or planner in this block.')
    write(out/'protocol.json',protocol);collection_wall=time.perf_counter();collection_cpu=time.process_time()
    dataset,meta=data.collect(geos,128);collection=dict(wall_seconds=time.perf_counter()-collection_wall,cpu_process_seconds=time.process_time()-collection_cpu,
        native_state_and_substep_replay_error=0,episodes=64,transitions=8192,native_observation_dim=18)
    torch.save(dataset,out/'dataset.pt');write(out/'episode_manifest.json',meta);write(out/'collection.json',collection)
    train=torch.tensor([i for i,r in enumerate(meta) if r['split']=='train']);mean,std,delta=normalization(dataset['state'],train)
    mean,std,delta=mean.cuda(),std.cuda(),delta.cuda();states=((dataset['state']-mean.cpu())/std.cpu()).cuda();actions=dataset['action'].cuda()
    training_seeds=seed_training(0);assert training_seeds==dict(cpu_initial_seed=0,cuda_initial_seed=0)
    model=NavigationJEPA(**config).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    generator=torch.Generator().manual_seed(0);history=[];losses=[];train_started=time.perf_counter();torch.cuda.reset_peak_memory_stats();model_initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial,_,_=evaluate(model,dataset['state'],dataset['action'],meta,mean,std,'cuda');write(out/'validation_step0.json',initial)
    for step in range(1,501):
        if time.perf_counter()-started>900:raise RuntimeError('Pilot wall budget reached; preserve outputs')
        s,a_batch,_,_=data.sample(states,actions,train,128,3,3,generator);model.train();loss,parts=model.loss(s,a_batch,delta)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
        opt.step();losses.append(float(loss.detach()))
        if step%100==0:
            metrics,arrays,windows=evaluate(model,dataset['state'],dataset['action'],meta,mean,std,'cuda');history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),**parts,validation=metrics))
            write(out/'training_history.json',history);print('TRAIN',step,'/500',float(loss.detach()),metrics['metrics']['jepa']['position_L2_by_k'][-1],flush=True)
    torch.cuda.synchronize();training_seconds=time.perf_counter()-train_started
    torch.save(dict(model=model.state_dict(),model_config=config,mean=mean.cpu(),std=std.cpu(),delta_std=delta.cpu(),protocol_sha256=sha(out/'protocol.json'),
        optimizer=opt.state_dict(),sampler_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),training_seed_records=training_seeds,step=500),out/'navigation_jepa_pilot.pt')
    metrics,arrays,windows=evaluate(model,dataset['state'],dataset['action'],meta,mean,std,'cuda');np.savez_compressed(out/'validation_predictions.npz',**arrays);write(out/'validation_windows.json',windows);write(out/'validation.json',metrics)
    fresh=NavigationJEPA(**config).cuda();fresh.load_state_dict(torch.load(out/'navigation_jepa_pilot.pt',map_location='cuda',weights_only=False)['model']);reloaded,again,_=evaluate(fresh,dataset['state'],dataset['action'],meta,mean,std,'cuda')
    reload_error=float(abs(arrays['predicted_state']-again['predicted_state']).max());assert reload_error==0
    changed=any(not torch.equal(model_initial[k],v.detach().cpu()) for k,v in model.state_dict().items());assert changed
    qualified=all(metrics['metrics']['jepa']['position_L2_by_k'][i]<metrics['metrics']['persistence']['position_L2_by_k'][i] for i in [0,7])
    summary=dict(pipeline_passed=True,model_position_gate_passed=qualified,collection=collection,training_wall_seconds=training_seconds,
        seconds_per_training_step_including_validation=training_seconds/500,peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
        parameters=sum(p.numel() for p in model.parameters()),steps=500,first50_loss_mean=float(np.mean(losses[:50])),last50_loss_mean=float(np.mean(losses[-50:])),
        checkpoint_reload_prediction_error=reload_error,model_weights_changed=True,training_seed_records=training_seeds,validation=metrics,dataset_sha256=sha(out/'dataset.pt'),checkpoint_sha256=sha(out/'navigation_jepa_pilot.pt'),
        wall_seconds=time.perf_counter()-started,formal_training_started=False,teammate_protocol_parity=False,planner_evaluated=False)
    write(out/'summary.json',summary);write(out/'environment.json',dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','vmas','numpy']},device=torch.cuda.get_device_name(0)))
    print('COMPLETE',json.dumps({k:summary[k] for k in ['pipeline_passed','model_position_gate_passed','training_wall_seconds','peak_cuda_allocated_MiB','parameters']}),flush=True)
if __name__=='__main__':main()
