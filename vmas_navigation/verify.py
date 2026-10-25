"""Independent NumPy physical metrics, no-future prediction and full native replay audit."""
import argparse,types,json
from pathlib import Path
import numpy as np
import torch
from vmas_navigation import data,pilot
from vmas_navigation.model import NavigationJEPA

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();root=a.run;torch.set_num_threads(1)
    protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text());meta=json.loads((root/'episode_manifest.json').read_text());metrics=json.loads((root/'validation.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert pilot.sha(Path(__file__).with_name(name))==digest
    assert pilot.sha(root/'dataset.pt')==summary['dataset_sha256'] and pilot.sha(root/'navigation_jepa_pilot.pt')==summary['checkpoint_sha256']
    d=torch.load(root/'dataset.pt',weights_only=False);checkpoint=torch.load(root/'navigation_jepa_pilot.pt',map_location='cpu',weights_only=False)
    assert checkpoint['step']==500 and 'optimizer' in checkpoint and 'sampler_rng' in checkpoint and checkpoint['protocol_sha256']==pilot.sha(root/'protocol.json')
    assert checkpoint['training_seed_records']==summary['training_seed_records']==dict(cpu_initial_seed=0,cuda_initial_seed=0)
    s=d['state'].numpy();actions=d['action'].numpy();obs=d['native_observation'].numpy();sub=d['substep_state'].numpy();goals=d['goals'].numpy()
    assert s.shape==(64,129,2,4) and actions.shape==(64,128,2,2) and obs.shape==(64,129,2,18) and sub.shape==(64,256,2,4)
    assert np.isfinite(s).all() and np.isfinite(obs).all() and abs(actions).max()<=1
    np.testing.assert_array_equal(obs[...,:4],s);np.testing.assert_array_equal(sub[:,1::2],s[:,1:])
    training=[i for i,r in enumerate(meta) if r['split']=='train'];validation=[i for i,r in enumerate(meta) if r['split']=='validation']
    assert len(training)==48 and len(validation)==16 and not ({meta[i]['geometry_id'] for i in training}&{meta[i]['geometry_id'] for i in validation})
    mean=s[training].reshape(-1,4).mean(0);std=np.maximum(s[training].reshape(-1,4).std(0,ddof=1),.01)
    np.testing.assert_allclose(mean,checkpoint['mean'].numpy(),atol=2e-6,rtol=1e-5);np.testing.assert_allclose(std,checkpoint['std'].numpy(),atol=2e-6,rtol=1e-5)
    normalized=(s[training]-checkpoint['mean'].numpy())/checkpoint['std'].numpy();delta=np.maximum(np.diff(normalized,axis=1).reshape(-1,4).std(0,ddof=1),.02)
    np.testing.assert_allclose(delta,checkpoint['delta_std'].numpy(),atol=2e-6,rtol=1e-5)
    # Independently replay all64 episodes in16 batched environments, with fresh native substep capture.
    maxreplay=0.
    for gi,geo in enumerate(protocol['geometry_specs']):
        ns=data.native.context('S1_symmetric','cpu',129);ns['GOALS']=np.array(geo['goals'],np.float32);ns['STARTS']=np.array(geo['starts'],np.float32)
        env=ns['create_scenario'](4);initial=s[gi*4,0];ns['set_cem_state'](env,initial[:,:2],initial[:,2:]);w=env.scenario.world;original=w._integrate_state;trace=[]
        def integrate(world,entity,substep):
            original(entity,substep)
            if entity is world.agents[-1]:trace.append(data.native.state(env).detach().numpy().copy())
        w._integrate_state=types.MethodType(integrate,w)
        for t in range(128):
            at=torch.from_numpy(actions[gi*4:gi*4+4,t]);env.step([at[:,i] for i in range(2)])
            error=float(abs(data.native.state(env).numpy()-s[gi*4:gi*4+4,t+1]).max());maxreplay=max(maxreplay,error);assert error==0
        np.testing.assert_array_equal(np.stack(trace,1),sub[gi*4:gi*4+4]);np.testing.assert_array_equal(goals[gi*4:gi*4+4],np.repeat(np.array(geo['goals'],np.float32)[None],4,0))
    windows=json.loads((root/'validation_windows.json').read_text())
    with np.load(root/'validation_predictions.npz') as z:pred={k:z[k] for k in z.files}
    assert len(windows)==80
    byid={r['episode_id']:i for i,r in enumerate(meta)}
    native_min=[]
    for j,r in enumerate(windows):
        i=byid[r['episode_id']];assert i in validation;t=r['start'];np.testing.assert_array_equal(pred['state_windows'][j],s[i,t:t+11]);np.testing.assert_array_equal(pred['actions'][j],actions[i,t:t+10])
        future_sub=sub[i,2*(t+2):2*(t+10)];initial=pred['state_windows'][j,2];dist=np.linalg.norm(future_sub[:,0,:2]-future_sub[:,1,:2],axis=-1)
        native_min.append(min(float(dist.min()),float(np.linalg.norm(initial[0,:2]-initial[1,:2]))))
    raw=pred['state_windows'];act=pred['actions'];truth=raw[:,3:];last=raw[:,2];persist=np.repeat(last[:,None],8,1)
    constant=np.concatenate([last[:,None,:,:2]+last[:,None,:,2:]*np.arange(1,9,dtype=np.float32)[None,:,None,None]*.1,np.repeat(last[:,None,:,2:],8,1)],-1)
    pos=last[...,:2].copy();vel=last[...,2:].copy();free=[]
    for t in range(8):
        vel*=.75
        for _ in range(2):vel+=act[:,t+2]*.05;pos+=vel*.05
        free.append(np.concatenate([pos,vel],-1).copy())
    versions=dict(jepa=pred['predicted_state'],persistence=persist,constant_velocity=constant,free_analytic=np.stack(free,1),encoded_readout=pred['encoded_state'])
    maxmetric=0.;native_diagnostic={}
    for name,version in versions.items():
        pe=np.linalg.norm(version[...,:2]-truth[...,:2],axis=-1).mean((0,2));ve=np.linalg.norm(version[...,2:]-truth[...,2:],axis=-1).mean((0,2));ref=metrics['metrics'][name]
        maxmetric=max(maxmetric,float(abs(pe-ref['position_L2_by_k']).max()),float(abs(ve-ref['velocity_L2_by_k']).max()))
        np.testing.assert_allclose(pe,ref['position_L2_by_k'],atol=2e-7,rtol=1e-5);np.testing.assert_allclose(ve,ref['velocity_L2_by_k'],atol=2e-7,rtol=1e-5)
        initial=np.linalg.norm(last[:,0,:2]-last[:,1,:2],axis=-1);predmin=np.minimum(initial,np.linalg.norm(version[:,:,0,:2]-version[:,:,1,:2],axis=-1).min(-1));truemin=np.minimum(initial,np.linalg.norm(truth[:,:,0,:2]-truth[:,:,1,:2],axis=-1).min(-1))
        assert abs(float(abs(predmin-truemin).mean())-ref['endpoint_minsep_MAE'])<2e-7
        assert int(((predmin>=.3)&(truemin<.3)).sum())==ref['false_safe_endpoint_windows']
        native_minimum=np.array(native_min);native_diagnostic[name]=dict(predicted_unsafe_windows=int((predmin<.3).sum()),true_native_unsafe_windows=int((native_minimum<.3).sum()),
            false_safe_vs_native=int(((predmin>=.3)&(native_minimum<.3)).sum()),false_unsafe_vs_native=int(((predmin<.3)&(native_minimum>=.3)).sum()),
            note='Post-hoc endpoint-prediction vs true native-substep diagnostic; not a certified safe rollout')
    # Predict using ONLY initial context frames, never encoding future target states.
    model=NavigationJEPA(**checkpoint['model_config']).eval();model.load_state_dict(checkpoint['model'])
    normalized=torch.from_numpy((raw[:,:3]-checkpoint['mean'].numpy())/checkpoint['std'].numpy());ta=torch.from_numpy(act)
    with torch.no_grad():
        z=model.encode(normalized);roll=model.rollout(z,ta[:,:2],ta[:,2:]);physical=(model.state_head(roll)*checkpoint['std']+checkpoint['mean']).numpy()
    forwarderror=float(abs(physical-pred['predicted_state']).max());np.testing.assert_allclose(physical,pred['predicted_state'],atol=3e-5,rtol=2e-5)
    assert summary['model_position_gate_passed']==all(metrics['metrics']['jepa']['position_L2_by_k'][i]<metrics['metrics']['persistence']['position_L2_by_k'][i] for i in [0,7])
    pilot.write(root/'native_safety_diagnostics.json',native_diagnostic)
    pilot.write(root/'verification.json',dict(status='passed',episodes=64,transitions=8192,whole_geometry_train_validation_disjoint=True,train_only_normalization_verified=True,
        raw_observation_state_subset_exact=True,independent_native_replay_error=maxreplay,validation_windows_exact=80,max_numpy_physical_metric_error=maxmetric,
        cpu_context_only_vs_saved_gpu_prediction_error=forwarderror,no_future_target_frames_needed_for_prediction=True,checkpoint_optimizer_rng_present=True,training_cpu_cuda_seed0_verified=True,position_gate_verified=True))
    print((root/'verification.json').read_text())
if __name__=='__main__':main()
