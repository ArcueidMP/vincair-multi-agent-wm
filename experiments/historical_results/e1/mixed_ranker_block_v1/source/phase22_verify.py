"""Independent array/label/feature checks, optionally train only before confirmation."""
import argparse,shutil,datetime
import numpy as np
from phase22_common import *
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['train','confirmation'],required=True);args=parser.parse_args()
folder=OUT/f'{args.split}_data';manifest=json.loads((folder/'manifest.json').read_text());meta=json.loads((folder/'metadata.json').read_text())
assert meta['status']=='passed';protocol=json.loads((OUT/'protocol.json').read_text());assert meta['protocol_sha256']==sha(OUT/'protocol.json')
if args.split=='confirmation':assert (OUT/'selection_frozen.json').exists()
seen=set();groups=0;maxcost=0.;maxfeat=0.;maxpose=0.
for scene in manifest:
    assert scene['seed'] not in seen;seen.add(scene['seed']);assert scene['seed'] in protocol[f'{args.split}_seeds']
    with np.load(folder/f"scene{scene['scene']:02d}_source.npz") as source:
        initial=source['state'][0].copy()
        assert np.array_equal(initial,np.array(scene['initial_state']))
        assert len(scene['groups'])==2
        for g in scene['groups']:
            p=folder/g['name'];assert sha(p/'inputs.npz')==g['input_sha256'] and sha(p/'labels.npz')==g['labels_sha256']
            with np.load(p/'inputs.npz') as z, np.load(p/'labels.npz') as labels:
                assert not {'true_cost','state_path','contact','goal','real_encoded_pose'}&set(z.files)
                x=z['features'];a=z['actions'];pp=z['predicted_pose'];pg=z['decoded_goal'];p0=z['initial_predicted_pose']
                assert x.shape==(64,21) and a.shape==(64,10,2,2) and abs(a).max()<=1.000001 and np.isfinite(x).all()
                # Separate formula implementation, no call to low_features.
                d=pp.astype(np.float64)[:,:4]-pg.astype(np.float64)[:4]
                angle=np.arctan2(pp.astype(np.float64)[:,4],pp.astype(np.float64)[:,5])-np.arctan2(float(pg[4]),float(pg[5]));angle=np.arctan2(np.sin(angle),np.cos(angle))
                rot=np.arctan2(pp.astype(np.float64)[:,4],pp.astype(np.float64)[:,5])-np.arctan2(float(p0[4]),float(p0[5]))
                act=a.reshape(64,10,4).astype(np.float64)
                pcost=np.linalg.norm(d,axis=1)+20/np.radians(20)*abs(angle)
                rebuilt=np.column_stack([d/512,np.sin(angle),np.cos(angle),pcost/100,np.linalg.norm(d,axis=1)/100,abs(angle)/np.pi,
                    act.mean(1),act.std(1),(pp.astype(np.float64)[:,2:4]-p0.astype(np.float64)[2:4])/64,np.sin(rot),np.cos(rot)]).astype(np.float32)
                maxfeat=max(maxfeat,float(abs(rebuilt-x).max()));assert np.allclose(rebuilt,x,rtol=0,atol=1e-7)
                assert np.allclose(pcost,z['predicted_cost'],rtol=0,atol=1e-8)
                states=labels['state_path'];assert states.shape==(64,11,11) and np.isfinite(states).all()
                assert np.array_equal(states[:,0],np.repeat(source['state'][g['t']][None],64,axis=0))
                assert np.array_equal(labels['prefix_actions'],source['action'][:g['t']])
                tp=pose(states[:,-1]);gp=pose(labels['goal']);expected=physical_cost(tp,gp)
                maxcost=max(maxcost,float(abs(expected-labels['true_cost']).max()));assert np.allclose(expected,labels['true_cost'],rtol=0,atol=1e-10)
                assert np.allclose(tcost(tp,gp),labels['true_T_cost'],rtol=0,atol=1e-10)
                maxpose=max(maxpose,float(abs(tp-labels['true_pose']).max()));assert np.allclose(tp,labels['true_pose'],rtol=0,atol=1e-12)
                for slice_ in (slice(16,32),slice(48,64)):
                    b=a[slice_];assert np.array_equal(b[:,:5],np.repeat(b[:,:1],5,axis=1));assert np.array_equal(b[:,5:],np.repeat(b[:,5:6],5,axis=1))
                cone=a[48:];assert np.linalg.norm(cone,axis=-1).max()<=.300001 and (cone*labels['direction']).sum(-1).min()>=-1e-8
            with np.load(p/'cem_intermediate_UNLABELED.npz') as z:assert z['actions'].shape==(2,16,10,2,2) and 'true_cost' not in z.files
            groups+=1
assert groups==2*len(manifest) and meta['max_branch_error']==0 and meta['max_second_replay_error']==0
assert all(sha(Path(p))==h for p,h in protocol['checkpoint_hashes'].items())
result=dict(status='passed',split=args.split,groups=groups,labeled_candidates=64*groups,max_cost_error=maxcost,max_feature_error=maxfeat,
    max_pose_error=maxpose,input_target_files_separate=True,feature_formula_independently_rebuilt=True,
    native_second_replay_error=meta['max_second_replay_error'],checkpoint_hashes_unchanged=True,old_test_candidate_npz_opened=0,
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(OUT/f'{args.split}_verification.json',result);shutil.copy2(Path(__file__),ROOT/'scripts/phase22_v1'/Path(__file__).name)
print(json.dumps(result),flush=True)
