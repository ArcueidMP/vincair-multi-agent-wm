"""Independent NumPy inference/selection validation, no fitting after test."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();out=args.run
    freeze=json.loads((out/'selection_frozen.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    assert sha(out/'state_action_scorer.pt')==freeze['checkpoint_sha256'] and sha(out/'portable_scorer.npz')==freeze['portable_sha256']
    assert sha(out/'protocol.json')==freeze['protocol_sha256'] and sha(args.data/'manifest.json')==protocol['data_manifest_sha256']
    pre=json.loads((out/'pretest_access_log.json').read_text());assert len(pre)==60 and not any(q['split']=='test' for q in pre)
    manifest=json.loads((args.data/'manifest.json').read_text());train=[];traincost=[]
    for row in manifest:
        if row['split']!='train':continue
        p=args.data/'train'/row['name']
        with np.load(p/'inputs.npz') as z:
            x=np.column_stack([np.tile(z['initial_state'].reshape(8),(64,1)),np.tile((z['goals']-z['initial_state'][:,:2]).reshape(4),(64,1)),z['actions'].reshape(64,32)])
            train.append(x)
        with np.load(p/'labels.npz') as z:traincost.append(z['total_cost'])
    train=np.stack(train);traincost=np.stack(traincost)
    with np.load(out/'portable_scorer.npz') as z:w={k:z[k] for k in z.files}
    assert np.array_equal(train.reshape(-1,44).mean(0),w['mean']) and np.array_equal(np.maximum(train.reshape(-1,44).std(0),.01),w['std'])
    assert float(traincost.mean())==float(w['cost_mean']) and float(traincost.std())==float(w['cost_std'])
    with np.load(out/'test_predictions.npz') as z:features=z['features'];savedraw=z['raw'];truth=z['truth'];minimum=z['min_separation']
    x=(features-w['mean'])/w['std'];x=np.maximum(x@w['net_0_weight'].T+w['net_0_bias'],0)
    x=np.maximum(x@w['net_2_weight'].T+w['net_2_bias'],0);raw=x@w['net_4_weight'].T+w['net_4_bias']
    error=float(abs(raw-savedraw).max());assert np.allclose(raw,savedraw,rtol=1e-5,atol=2e-6)
    # Use independently inferred scores to reproduce actual saved discrete decisions.
    cost=raw[...,1]*w['cost_std']+w['cost_mean'];risk=1/(1+np.exp(-np.clip(raw[...,0],-50,50)));sep=raw[...,2]*.30
    test=json.loads((out/'test_summary.json').read_text());rows=[q for q in manifest if q['split']=='test'];selected=[]
    for i,row in enumerate(rows):
        with np.load(args.data/'test'/row['name']/'inputs.npz') as z:
            rebuild=np.column_stack([np.tile(z['initial_state'].reshape(8),(64,1)),np.tile((z['goals']-z['initial_state'][:,:2]).reshape(4),(64,1)),z['actions'].reshape(64,32)])
            assert np.array_equal(rebuild,features[i])
        valid=[k for k in range(64) if risk[i,k]<w['threshold']]
        idx=min(valid,key=lambda k:(cost[i,k],k)) if valid else min(range(64),key=lambda k:(-sep[i,k],cost[i,k],k))
        for name,picked in [('learned_safety',idx),('learned_cost_only',int(cost[i].argmin()))]:
            r=next(q for q in test['groups'] if q['name']==row['name'] and q['method']==name)
            assert picked==r['selected'] and bool(minimum[i,picked]<.30)==r['unsafe']
            assert abs(float(truth[i,picked]-truth[i].min())-r['regret'])<1e-6
        selected.append(idx)
    a=test['methods']['learned_safety'];b=test['methods']['learned_cost_only']
    improvement=b['mean']['unsafe_given_safe']-a['mean']['unsafe_given_safe']
    better=sum(q['unsafe_given_safe']<r['unsafe_given_safe'] for q,r in zip(a['scenes'],b['scenes']))
    assert test['criterion_met']==bool(improvement>=.1 and better>=2)
    result=dict(status='passed',train_only_normalization_exact=True,input_features_independently_rebuilt=True,
        numpy_prediction_max_abs_error=error,all_test_choices_and_regrets_reproduced=True,pretest_candidate_groups=60,test_candidate_groups=12,
        checkpoint_unchanged=True,refit_after_test=False,test_now_consumed=True,criterion_met=test['criterion_met'])
    write(out/'verification.json',result)
    dest=Path(__file__).resolve().parent/'state_action_results/v1';dest.mkdir(parents=True,exist_ok=True)
    # Access logs contain group IDs, not raw future arrays. Checkpoints remain external.
    for name in ['protocol.json','selection_frozen.json','training_history.json','validation_summary.json','test_summary.json','verification.json','pretest_access_log.json']:
        (dest/name).write_bytes((out/name).read_bytes())
    results=[]
    for split in ['validation','test']:
        s=json.loads((out/f'{split}_summary.json').read_text())
        for method,m in s['methods'].items():results.append(f"| {split} | {method} | {100*m['mean']['unsafe_given_safe']:.2f}% | {100*m['mean']['unsafe_all']:.2f}% | {m['mean']['regret']:.4f} |")
    doc=f'''# Direct state-action safety scorer baseline v1

This is an actually trained5059-parameter MLP, **not LeWM, not a Navigation world model or readout integration**. User confirmed the dedicated Navigation weights have not arrived. It receives only current observed state, relative goals and candidate action sequences (44 features). No simulator future or safety label enters inference.

## Frozen protocol and results

Train12 geometries/48 groups/3072 candidates; validation3 geometries/12 groups/768 candidates. Select epoch and threshold on validation; freeze weights and policy; activate test3 geometries/12 groups/768 candidates once. Three transformed route families remain shared; not held-out-family generalization.

Model44→64→32→3 has heads unsafe logit, scaled original soft cost and scaled minimum substep separation. CPU seed0, AdamW lr.001 wd.01, class-balanced BCE+0.1 cost MSE+separation MSE. Feature and target statistics use train only. Validation selection minimizes scene-average unsafe selection when a safe candidate exists, then original soft regret. Maximum200 epochs, patience30, thresholds0.1/0.25/0.5. Selected epoch{freeze['selected_epoch']}, threshold{freeze['threshold']}.

| Split | Scoring rule | Unsafe when safe available* | Unsafe over all states* | Original soft regret* |
|---|---|---:|---:|---:|
{chr(10).join(results)}

*Average within geometry over eligible offsets, then equally across geometries. `learned_cost_only` uses the cost head of the SAME safety-selected network: a scoring-rule ablation, not an independently selected cost-only competitor. True soft/safety oracles use future supervision and are diagnostic only. A soft oracle can prefer unsafe actions because the original soft objective does not enforce safety.

Risk policy: among predicted-safe candidates choose lowest predicted cost; if all predicted unsafe, choose largest predicted minimum separation then lowest cost. This fallback is not guaranteed safe. Test includes an all-unsafe true candidate pool, so conditional and unconditional unsafe rates differ; true-safety oracle unconditional risk is not zero.

Test conditional unsafe reduction is{100*improvement:.2f} percentage points in{better}/3 geometries. Predeclared success required>=10 points AND>=2/3 geometries: **not met**. Validation0% does not transfer perfectly; retain this negative/limited result. No closed-loop episode, Navigation checkpoint claim, LeWM improvement, or post-test refit is reported.

## Validation and storage

Sixteen targeted unit tests pass: prior physics checks plus feature contract, future-label argument rejection, pre-IO test guard, checkpoint-hash guard, fallback, safe selection and parameter count. Independent NumPy reconstruction verifies train-only statistics, all test input features, forward predictions (maximum absolute difference{error:.2e}), every selected candidate/regret and the success criterion. Weight hash remains unchanged after test. Raw model/portable checkpoint and predictions remain external; frozen hashes identify them.

The previously sealed test split has now been consumed by this frozen experiment. It cannot be repeatedly used to select thresholds, models or features. Earlier conflict-data reports refer to its status at creation; this report and current PROJECT_STATE supersede that status. A later change needs new confirmation geometries.

## Reproduce

```bash
python -m unittest discover -s tests -v
python experiments/e4_navigation/state_action_ranker.py --data outputs/conflict_ranking_v1 --output outputs/state_action_ranker_v1
python experiments/e4_navigation/verify_state_action_ranker.py --data outputs/conflict_ranking_v1 --run outputs/state_action_ranker_v1
```

Required next material: Navigation-specific checkpoint, load API/config/normalization, agent/history/action contract and compatible state head. This direct scorer remains a separately named baseline when those arrive. Do not plug PushT weights into the Navigation world-model adapter.
'''
    (dest/'README.md').write_text(doc,encoding='utf-8');(out/'STATE_ACTION_RANKER_REPORT.md').write_text(doc,encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
