"""Independent NumPy inference/physics/decision audit; no fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def raw_features(s,g,a):return np.column_stack([np.tile(s.ravel(),(len(a),1)),np.tile((g-s[:,:2]).ravel(),(len(a),1)),a.reshape(len(a),32)])
def canonical(s,g,a):
    result=[]
    for order in ([0,1],[1,0]):
        v=s[order];delta=v[1,:2]-v[0,:2];theta=np.arctan2(delta[1],delta[0]);c=np.cos(theta);t=np.sin(theta)
        def rot(x):return np.stack([x[...,0]*c+x[...,1]*t,-x[...,0]*t+x[...,1]*c],-1)
        center=v[:,:2].mean(0);state=np.concatenate([rot(v[:,:2]-center),rot(v[:,2:])],-1)
        result.append(np.column_stack([np.tile(state.ravel(),(len(a),1)),np.tile(rot(g[order]-v[:,:2]).ravel(),(len(a),1)),rot(a[:,:,order]).reshape(len(a),32)]))
    return np.stack(result,1)
def infer(x,w):
    x=(x-w['mean'])/w['std'];x=np.maximum(x@w['net_0_weight'].T+w['net_0_bias'],0)
    x=np.maximum(x@w['net_2_weight'].T+w['net_2_bias'],0);return (x@w['net_4_weight'].T+w['net_4_bias']).mean(-2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);args=ap.parse_args();root=args.run
    frozen=json.loads((root/'selection_frozen.json').read_text());assert frozen['protocol_sha256']==sha(root/'protocol.json')
    rows=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'summary.json').read_text());weights={};training={'raw':[],'invariant':[]};traincost=[]
    for kind in training:
        assert sha(root/f'{kind}.pt')==frozen['models'][kind]['checkpoint_sha256']
        with np.load(root/f'{kind}_portable.npz') as z:weights[kind]={k:z[k] for k in z.files}
    with np.load(args.baseline/'portable_scorer.npz') as z:weights['old']={k:z[k] for k in z.files}
    assert sha(args.baseline/'state_action_scorer.pt')==frozen['baseline_sha256']
    with np.load(root/'confirmation_predictions.npz') as z:pred={k:z[k] for k in z.files}
    maxcost=0.;maxforward=0.;maxinvariance=0.;ci=0;decisions=0;hashes={};pairs={}
    for row in rows:
        p=root/'data'/row['name']
        for file,key in [('inputs.npz','input_sha256'),('labels.npz','labels_sha256')]:
            assert sha(p/file)==row[key];hashes[str((p/file).relative_to(root))]=row[key]
        with np.load(p/'inputs.npz') as z,np.load(p/'labels.npz') as lab:
            s,g,a=z['initial_state'],z['goals'],z['actions'];path=lab['state_path'];sub=lab['substep_state_path'];y=lab['true_cost'];minimum=lab['min_separation']
            assert a.shape==(68,8,2,2) and path.shape==(68,9,2,4) and sub.shape==(68,16,2,4)
            assert np.array_equal(sub[:,1::2],path[:,1:]);assert np.array_equal(path[:,0],np.repeat(s[None],68,0));assert abs(a).max()<=1.000001
            distance=np.linalg.norm(path[:,1:,0,:2]-path[:,1:,1,:2],axis=-1)
            cost=5*np.linalg.norm(path[:,-1,:,:2]-g,axis=-1).sum(-1)+100*np.maximum(.3-distance,0).__pow__(2).sum(-1)+.01*(a*a).mean((1,2,3))
            maxcost=max(maxcost,float(abs(cost-y).max()));np.testing.assert_allclose(cost,y,atol=1e-5,rtol=1e-5)
            allsub=np.concatenate([path[:,:1],sub],1);m=np.linalg.norm(allsub[:,:,0,:2]-allsub[:,:,1,:2],axis=-1).min(-1)
            np.testing.assert_allclose(m,minimum,atol=1e-7,rtol=0)
            # Actions0:64 remain identical across paired initial conditions.
            key=(row['base_id'],row['name'][-2:]);value=(g,a[:64])
            if key in pairs:
                for x,z in zip(pairs[key],value):np.testing.assert_array_equal(x,z)
            else:pairs[key]=tuple(v.copy() for v in value)
            xraw=raw_features(s,g,a)[:,None];xinv=canonical(s,g,a)
            np.testing.assert_allclose(xinv[:,::-1],canonical(s[::-1],g[::-1],a[:,:,::-1]),atol=1e-6)
            if row['split']=='train':
                training['raw'].append(xraw);training['invariant'].append(xinv);traincost.append(y)
            if row['split']!='confirmation':continue
            outputs={k:infer(xinv if k=='invariant' else xraw,w) for k,w in weights.items()}
            # Verify translation invariance on actual confirmation inputs without changing labels.
            shift=np.array([.23,-.19],np.float32);shifted=s.copy();shifted[:,:2]+=shift
            shiftedraw=infer(canonical(shifted,g+shift,a),weights['invariant'])
            maxinvariance=max(maxinvariance,float(abs(shiftedraw-outputs['invariant']).max()))
            for k,raw in outputs.items():
                maxforward=max(maxforward,float(abs(raw-pred[k][ci]).max()));np.testing.assert_allclose(raw,pred[k][ci],atol=2e-5,rtol=1e-5)
            for method,result in summary['methods'].items():
                kind=method.split('_')[0];count=int(method.split('_')[-1]);raw=outputs[kind][:count];w=weights[kind]
                if kind=='old':threshold=float(w['threshold']);margin=0
                else:policy=frozen['models'][kind]['policy'];threshold=policy['threshold'];margin=policy['margin']
                risk=1/(1+np.exp(-np.clip(raw[:,0],-50,50)));cost=raw[:,1]*w['cost_std']+w['cost_mean'];sep=raw[:,2]*.3
                consistent='consistent' in method
                valid=[j for j in range(count) if risk[j]<threshold and (not consistent or sep[j]>=.3+margin)]
                if valid:idx=min(valid,key=lambda j:(cost[j],j))
                elif consistent:idx=min(range(count),key=lambda j:(risk[j],-sep[j],cost[j],j))
                else:idx=min(range(count),key=lambda j:(-sep[j],cost[j],j))
                record=result['records'][ci];assert record['name']==row['name'] and record['selected']==idx
                assert record['unsafe']==bool(minimum[idx]<.3);assert record['safe_available']==bool((minimum[:count]>=.3).any())
                assert abs(record['regret']-float(y[idx]-y[:count].min()))<1e-6;decisions+=1
            ci+=1
    for kind,xx in training.items():
        x=np.stack(xx).reshape(-1,44);w=weights[kind]
        np.testing.assert_allclose(x.mean(0),w['mean'],atol=1e-6,rtol=1e-5);np.testing.assert_allclose(np.maximum(x.std(0),.01),w['std'],atol=1e-6,rtol=1e-5)
        y=np.stack(traincost);assert abs(y.mean()-w['cost_mean'])<1e-6 and abs(y.std()-w['cost_std'])<1e-6
    for result in summary['methods'].values():
        for name in ['unsafe_given_safe','unsafe_all','regret']:
            values=[]
            for scene in result['scenes']:
                rr=[r for r in result['records'] if r['base_id']==scene['base_id'] and (name!='unsafe_given_safe' or r['safe_available'])]
                v=float(np.mean([r['regret' if name=='regret' else 'unsafe'] for r in rr])) if rr else None
                assert v==scene[name]
                if v is not None:values.append(v)
            assert abs(np.mean(values)-result['mean'][name])<1e-12
    assert ci==72 and len(rows)==264 and summary['max_native_replay_error']==0
    dump(root/'dataset_sha256.json',hashes)
    dump(root/'verification.json',dict(status='passed',groups=264,candidates=17952,confirmation_groups=ci,
        independent_selected_indices=decisions,max_cost_error=maxcost,max_numpy_forward_error=maxforward,
        translation_prediction_max_error=maxinvariance,native_replay_error=0,train_only_statistics_verified=True,
        input_label_hashes_verified=True,paired_original_actions_verified=True,all_aggregate_metrics_verified=True))
    print((root/'verification.json').read_text())
if __name__=='__main__':main()
