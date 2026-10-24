"""Independent selected-ridge refit and protocol/freeze checks; training data only."""
import shutil
import numpy as np
import torch
from phase22_common import *
frozen=json.loads((OUT/'selection_frozen.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
assert frozen['confirmation_files_opened']==0 and frozen['old_test_candidate_npz_opened']==0
assert sha(OUT/'selected_scorer.pt')==frozen['checkpoint_sha256'] and sha(OUT/'protocol.json')==frozen['protocol_sha256']
ck=torch.load(OUT/'selected_scorer.pt',map_location='cpu',weights_only=False)
manifest=json.loads((OUT/'train_data/manifest.json').read_text());xs=[];ys=[];rs=[]
for scene in manifest:
    for g in scene['groups']:
        p=OUT/'train_data'/g['name']
        with np.load(p/'inputs.npz') as z:xs.append(z['features']);rs.append(z['predicted_cost'])
        with np.load(p/'labels.npz') as z:ys.append(z['true_cost'])
x=np.stack(xs);y=np.stack(ys);r=np.stack(rs)
mean=x.reshape(-1,21).mean(0);std=np.maximum(x.reshape(-1,21).std(0),.01)
assert np.array_equal(mean,ck['mean'].numpy()) and np.array_equal(std,ck['std'].numpy())
normal_error=None;beta_error=None;correction=None
if ck['kind']=='ridge':
    nx=(x-mean)/std;xc=nx-nx.mean(1,keepdims=True);res=(y-r)/100;res-=res.mean(1,keepdims=True)
    a=xc.reshape(-1,21).astype(np.float64);b=res.reshape(-1)
    lhs=np.einsum('ni,nj->ij',a,a)/len(a)+ck['lam']*np.eye(21);rhs=np.einsum('ni,n->i',a,b)/len(a)
    beta=np.linalg.solve(lhs,rhs);beta_error=float(abs(beta-ck['beta'].numpy()).max());normal_error=float(abs(lhs@ck['beta'].numpy()-rhs).max())
    assert beta_error<1e-10 and normal_error<1e-10
    correction=np.clip(nx@ck['beta'].numpy(),-.25,.25)
    assert np.isfinite(correction).all() and abs(correction).max()<=.25
    np.savez_compressed(OUT/'selected_scorer_portable.npz',mean=mean,std=std,beta=ck['beta'].numpy(),bound=.25,lam=ck['lam'])
assert ck['feature_names']==FEATURE_NAMES and len(FEATURE_NAMES)==21
conditions=[]
baseline=frozen['candidates'][0]['metrics']['mean']
for q in frozen['candidates']:
    expected=q['kind']=='zero' or ((baseline['regret']-q['metrics']['mean']['regret'])/max(baseline['regret'],1e-8)>=.05 and q['metrics']['mean']['T_gain']>=baseline['T_gain']-1)
    assert q['eligible']==expected
    conditions.append(q['name'])
eligible=[q for q in frozen['candidates'] if q['eligible']]
best=eligible[0]
for q in eligible[1:]:
    if q['metrics']['mean']['regret']<best['metrics']['mean']['regret']-1e-8:best=q
assert best['name']==frozen['selected']
result=dict(status='passed',normalization_train_only_exact=True,selected_ridge_beta_max_error=beta_error,
    selected_ridge_normal_equation_max_error=normal_error,correction_bound_checked=True,selection_recomputed=True,
    checkpoint_hash_matches=True,model_checkpoint=frozen['checkpoint_sha256'],confirmation_files_used_for_fitting=0,
    candidates=conditions,old_test_npz_access=0)
write(OUT/'model_verification.json',result);shutil.copy2(Path(__file__),ROOT/'scripts/phase22_v1'/Path(__file__).name)
print(json.dumps(result),flush=True)
