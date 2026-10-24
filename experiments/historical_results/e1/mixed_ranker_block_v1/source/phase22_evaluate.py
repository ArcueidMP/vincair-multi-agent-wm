"""Frozen confirmation evaluation and explicit closed-loop gate, no retraining."""
import datetime,shutil
import numpy as np
import torch
from phase22_common import *
torch.set_num_threads(1)
assert not (OUT/'confirmation_summary.json').exists()
protocol=json.loads((OUT/'protocol.json').read_text());frozen=json.loads((OUT/'selection_frozen.json').read_text())
assert sha(OUT/'selected_scorer.pt')==frozen['checkpoint_sha256']
assert sha(OUT/'protocol.json')==frozen['protocol_sha256']
ck=torch.load(OUT/'selected_scorer.pt',map_location='cpu',weights_only=False)
if ck['kind']=='mlp':net=ResidualMLP().eval();net.load_state_dict(ck['state_dict'])
def apply(x,base):
    nx=(x-ck['mean'].numpy())/ck['std'].numpy()
    if ck['kind']=='zero':return base/100
    if ck['kind']=='ridge':return base/100+np.clip(nx@ck['beta'].numpy(),-.25,.25)
    with torch.no_grad():return base/100+net(torch.from_numpy(nx)).numpy()
folder=OUT/'confirmation_data';meta=json.loads((folder/'metadata.json').read_text());assert meta['status']=='passed'
manifest=json.loads((folder/'manifest.json').read_text());assert len(manifest)==8
records=[];access=[];allscore=[];alltruth=[];allT=[];allbase=[];sceneids=[]
for scene in manifest:
    assert scene['seed'] in protocol['confirmation_seeds']
    for g in scene['groups']:
        p=folder/g['name'];assert sha(p/'inputs.npz')==g['input_sha256'] and sha(p/'labels.npz')==g['labels_sha256']
        with np.load(p/'inputs.npz') as z:
            x=z['features'];base=z['predicted_cost'];score=apply(x,base)
            assert np.isfinite(score).all() and score.shape==(64,)
        with np.load(p/'labels.npz') as z:
            truth=z['true_cost'];T=z['true_T_cost'];tstart=float(z['true_T_cost_start']);gstart=float(z['true_cost_start'])
            real=z['real_encoded_cost']
        for name,s in [('readout',base/100),('selected',score),('real_encoded_diagnostic',real/100)]:
            idx=int(s.argmin());i,j=np.triu_indices(64,1);d=truth[j]-truth[i];use=abs(d)>1;direction=np.sign(s[j]-s[i])*np.sign(d)
            records.append(dict(method=name,scene=scene['scene'],seed=scene['seed'],t=g['t'],name=g['name'],
                regret=float(truth[idx]-truth.min()),PRA=float(np.where(direction[use]>0,1,np.where(direction[use]==0,.5,0)).mean()) if use.any() else None,
                T_gain=float(tstart-T[idx]),global_gain=float(gstart-truth[idx]),selected_index=idx,proposal=ARMS[idx//16],
                T_oracle_gain=float(tstart-T.min())))
        access.append(dict(scene=scene['scene'],group=g['name'],split='confirmation'))
        allscore.append(score);alltruth.append(truth);allT.append(T);allbase.append(base/100);sceneids.append(scene['scene'])
def aggregate(method):
    rec=[q for q in records if q['method']==method];scenes=[]
    for scene in range(8):
        group=[q for q in rec if q['scene']==scene]
        scenes.append(dict(scene=scene,seed=group[0]['seed'],**{k:float(np.mean([q[k] for q in group if q[k] is not None])) for k in ['regret','PRA','T_gain','global_gain','T_oracle_gain']}))
    return dict(mean={k:float(np.mean([q[k] for q in scenes])) for k in ['regret','PRA','T_gain','global_gain','T_oracle_gain']},scenes=scenes)
results={m:aggregate(m) for m in ['readout','selected','real_encoded_diagnostic']}
a=results['selected'];b=results['readout'];delta=np.array([q['regret']-r['regret'] for q,r in zip(a['scenes'],b['scenes'])])
improvement=(b['mean']['regret']-a['mean']['regret'])/max(b['mean']['regret'],1e-8)
conditions=dict(nonzero_selected=ck['kind']!='zero',relative_regret_improvement_at_least_10pct=bool(improvement>=.1),
    better_at_least_5_scenes=bool((delta<0).sum()>=5),T_gain_not_worse_by_over1=bool(a['mean']['T_gain']>=b['mean']['T_gain']-1))
gate=all(conditions.values())
summary=dict(status='passed',selected=frozen['selected'],results=results,regret_delta_by_scene=delta.tolist(),
    relative_regret_improvement=float(improvement),better_scenes=int((delta<0).sum()),worse_scenes=int((delta>0).sum()),
    closed_loop_gate=gate,gate_conditions=conditions,checkpoint_sha256=sha(OUT/'selected_scorer.pt'),
    no_refit_after_confirmation=True,old_test_candidate_npz_opened=0,
    evaluated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    limitations=['Eight confirmation scenes, one fitting seed; descriptive local confirmation only.',
    'Mixed pool contains privileged contact actions; not an image-only planner result.',
    'T-only secondary does not replace original global task/success contract.'])
np.savez_compressed(OUT/'confirmation_scores.npz',truth=np.stack(alltruth),true_T=np.stack(allT),readout=np.stack(allbase),selected=np.stack(allscore),scene=np.array(sceneids))
write(OUT/'confirmation_group_metrics.json',records);write(OUT/'confirmation_access_log.json',access);write(OUT/'confirmation_summary.json',summary)
shutil.copy2(Path(__file__),ROOT/'scripts/phase22_v1'/Path(__file__).name)
print(json.dumps(summary,indent=2),flush=True)
