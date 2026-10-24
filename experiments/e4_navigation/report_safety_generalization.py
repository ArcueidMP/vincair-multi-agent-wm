"""Independent numerical audit, complete geometry plots and deterministic case figures."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def infer(x,w):
    x=(x-w['mean'])/w['std'];x=np.maximum(x@w['net_0_weight'].T+w['net_0_bias'],0)
    x=np.maximum(x@w['net_2_weight'].T+w['net_2_bias'],0);return x@w['net_4_weight'].T+w['net_4_bias']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--old-data',type=Path,required=True);args=ap.parse_args()
    out=args.run;protocol=json.loads((out/'protocol.json').read_text());summary=json.loads((out/'summary.json').read_text());old=json.loads((out/'old_failure_records.json').read_text());rows=json.loads((out/'new_failure_records.json').read_text())
    assert sha(args.model/'state_action_scorer.pt')==protocol['checkpoint_sha256']
    previous_test=json.loads((args.model/'test_summary.json').read_text())
    for r in [q for q in old if q['split']=='test']:
        previous=next(q for q in previous_test['groups'] if q['method']=='learned_safety' and q['name']==r['name'])
        assert previous['selected']==r['selected'] and previous['unsafe']==r['methods']['learned_safety']['unsafe']
    with np.load(args.model/'portable_scorer.npz') as z:w={k:z[k] for k in z.files}
    maxcost=0.;maxraw=0.;decisions=0;matched={};hashes={};allrecords=old+rows
    for r in rows:
        p=out/'new_data'/r['name']
        for name,key in [('inputs.npz','input_sha256'),('labels.npz','labels_sha256'),('predictions.npz','prediction_sha256')]:
            assert sha(p/name)==r[key];hashes[str((p/name).relative_to(out))]=r[key]
        with np.load(p/'inputs.npz') as z,np.load(p/'labels.npz') as lab,np.load(p/'predictions.npz') as pred:
            initial=z['initial_state'];a=z['actions'];goals=z['goals'];s=lab['state_path'];sub=lab['substep_state_path']
            r['actual_initial_separation']=float(np.linalg.norm(initial[0,:2]-initial[1,:2]))
            assert a.shape==(64,8,2,2) and s.shape==(64,9,2,4) and sub.shape==(64,16,2,4)
            assert np.isfinite(s).all() and abs(a).max()<=1.000001 and np.array_equal(sub[:,1::2],s[:,1:])
            assert np.array_equal(s[:,0],np.repeat(initial[None],64,axis=0))
            x=np.column_stack([np.tile(initial.reshape(8),(64,1)),np.tile((goals-initial[:,:2]).reshape(4),(64,1)),a.reshape(64,32)])
            assert np.array_equal(x,pred['features']);raw=pred['raw'];independent=infer(x,w)
            maxraw=max(maxraw,float(abs(raw-independent).max()));assert np.allclose(raw,independent,rtol=1e-5,atol=2e-6)
            ds=np.linalg.norm(s[:,1:,0,:2]-s[:,1:,1,:2],axis=-1);goal=np.linalg.norm(s[:,-1,:,:2]-goals[None],axis=-1).sum(1)
            true=5*goal+100*(np.maximum(.3-ds,0)**2).sum(1)+.01*(a*a).mean((1,2,3))
            maxcost=max(maxcost,float(abs(true-lab['true_cost']).max()));assert np.allclose(true,lab['true_cost'],rtol=1e-5,atol=1e-5)
            allsub=np.concatenate([s[:,:1],sub],1);minimum=np.linalg.norm(allsub[:,:,0,:2]-allsub[:,:,1,:2],axis=-1).min(1)
            assert np.allclose(minimum,lab['min_separation'],rtol=0,atol=1e-7)
            risk=1/(1+np.exp(-np.clip(raw[:,0],-50,50)));cost=raw[:,1]*w['cost_std']+w['cost_mean'];sep=raw[:,2]*.3
            accepted=[k for k in range(64) if risk[k]<protocol['threshold']]
            idx=min(accepted,key=lambda k:(cost[k],k)) if accepted else min(range(64),key=lambda k:(-sep[k],cost[k],k))
            assert idx==r['selected'] and (not bool(accepted))==r['fallback']
            safe=lab['min_separation']>=.3;assert bool(safe.any())==r['safe_available']
            assert bool(not safe[idx])==r['methods']['learned_safety']['unsafe'];decisions+=1
            best=int(np.flatnonzero(safe)[lab['true_cost'][safe].argmin()]) if safe.any() else None
            if best is None:expected_category='no_safe_candidate'
            elif not safe[idx]:expected_category='risk_false_accept' if accepted else 'fallback_miss'
            elif lab['true_cost'][idx]-lab['true_cost'][best]<=1e-4:expected_category='safe_optimal'
            elif risk[best]>=protocol['threshold']:expected_category='safe_risk_exclusion'
            else:expected_category='safe_cost_ranking'
            assert r['category']==expected_category
            key=(r['base_id'],r['distance'])
            if key in matched:
                baseline=matched[key];assert np.array_equal(a,baseline['actions']) and np.array_equal(goals,baseline['goals'])
            else:matched[key]=dict(actions=a.copy(),goals=goals.copy())
    assert len(rows)==72 and len(matched)==24 and summary['max_native_replay_error']==0
    # Rebuild aggregate decision rates, independent of producer aggregate().
    for condition,stats in summary['conditions'].items():
        rr=[q for q in rows if q['condition']==condition]
        for method in stats['methods']:
            vals=[]
            for gid in sorted({q['base_id'] for q in rr}):
                g=[q for q in rr if q['base_id']==gid and q['safe_available']]
                if g:vals.append(np.mean([q['methods'][method]['unsafe'] for q in g]))
            assert abs(float(np.mean(vals))-stats['methods'][method]['mean']['unsafe_given_safe'])<1e-12
        assert sum(stats['category_counts'].values())==24
    verification=dict(status='passed',new_groups=72,new_candidates=4608,paired_action_goal_groups=24,
        independent_decisions_checked=decisions,max_cost_formula_error=maxcost,max_numpy_prediction_error=maxraw,
        native_second_replay_error=0,checkpoint_unchanged=True,no_refit=True,no_threshold_search=True,
        aggregate_rates_reproduced=True,input_feature_formula_exact=True)
    write(out/'verification.json',verification);write(out/'dataset_sha256.json',hashes)
    figures=out/'figures';figures.mkdir(exist_ok=True);colors=['#0072B2','#D55E00']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout='constrained')
    titles=['New angle / orientation','Asymmetric initial velocity','Arrival offset']
    for ax,(condition,stats),title in zip(axes,summary['conditions'].items(),titles):
        a=stats['methods']['learned_cost_only']['scenes'];b=stats['methods']['learned_safety']['scenes']
        for k,(x,y) in enumerate(zip(a,b)):
            jitter=(k-5.5)*.012;values=[100*x['unsafe_given_safe'],100*y['unsafe_given_safe']]
            ax.plot([jitter,1+jitter],values,color='#aaaaaa',alpha=.6,lw=.7,zorder=1)
            ax.scatter(jitter,values[0],color=colors[0],marker='o',s=22,zorder=2)
            ax.scatter(1+jitter,values[1],color=colors[1],marker='s',s=22,zorder=2)
        for i,m in enumerate(['learned_cost_only','learned_safety']):
            value=100*stats['methods'][m]['mean']['unsafe_given_safe'];ax.plot([i-.15,i+.15],[value,value],color='black',lw=2.5,zorder=3)
        ax.set(xticks=[0,1],xticklabels=['Cost head only','Safety rule'],ylim=(-5,105),xlim=(-.3,1.3),title=title)
        ax.grid(axis='y',alpha=.2);ax.set_xlabel('12 paired base geometries; black = mean')
    axes[0].set_ylabel('Unsafe selection when safe option exists (%)')
    fig.savefig(figures/'generalization_pairs.png',dpi=160);fig.savefig(figures/'generalization_pairs.svg');plt.close(fig)
    # Deterministic illustrative selection; this figure is not the full sample distribution.
    test=sorted([q for q in old if q['split']=='test'],key=lambda q:q['name']);examples=[]
    for predicate in [lambda q:q['safe_available'] and q['methods']['learned_safety']['unsafe'],lambda q:not q['safe_available'],lambda q:not q['methods']['learned_safety']['unsafe']]:
        chosen=next((q for q in test if predicate(q) and q not in examples),None)
        if chosen:examples.append(chosen)
    fig,axes=plt.subplots(2,len(examples),figsize=(12,7),layout='constrained',squeeze=False);caseprovenance=[];limits=[];loaded=[]
    for r in examples:
        with np.load(args.old_data/'test'/r['name']/'labels.npz') as lab:paths=lab['substep_state_path'];initial=lab['state_path'][:,0]
        paths=np.concatenate([initial[:,None],paths],1);chosen=r['selected'];alternative=r['best_true_safe_index']
        idxs=[chosen]+([alternative] if alternative is not None and alternative!=chosen else [])
        limits.append(paths[idxs,:,:,:2].reshape(-1,2));loaded.append((r,paths,idxs))
    allxy=np.concatenate(limits);xmin,ymin=allxy.min(0)-.05;xmax,ymax=allxy.max(0)+.05
    allseps=np.concatenate([np.linalg.norm(paths[idxs,:,0,:2]-paths[idxs,:,1,:2],axis=-1).reshape(-1) for _,paths,idxs in loaded])
    sepmin=min(.30,float(allseps.min()))-.005;sepmax=max(.30,float(allseps.max()))+.005
    for col,(r,paths,idxs) in enumerate(loaded):
        ax=axes[0,col];sepax=axes[1,col]
        for j,idx in enumerate(idxs):
            linestyle='-' if j==0 else '--'
            for agent in range(2):
                p=paths[idx,:,agent,:2];ax.plot(p[:,0],p[:,1],color=colors[agent],ls=linestyle,lw=1.8)
                ax.scatter(*p[0],color=colors[agent],marker='o',s=20);ax.scatter(*p[-1],color=colors[agent],marker='x',s=30)
            sep=np.linalg.norm(paths[idx,:,0,:2]-paths[idx,:,1,:2],axis=-1)
            sepax.plot(np.arange(len(sep))*.05,sep,color='#0072B2' if j==0 else '#666666',ls=linestyle,label='Selected' if j==0 else 'True-safe best (oracle)')
        ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='x (VMAS units)',ylabel='y (VMAS units)',title=r['category']+'\n'+r['name']);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        sepax.axhline(.30,color='black',ls=':',label='Safety threshold');sepax.set(xlabel='Time (s), native substeps',ylabel='Agent separation',ylim=(sepmin,sepmax));sepax.grid(alpha=.2);sepax.legend(fontsize=8)
        caseprovenance.append(dict(name=r['name'],category=r['category'],selected=r['selected'],oracle_reference=r['best_true_safe_index']))
    axes[0,0].legend(handles=[Line2D([0],[0],color=colors[0],label='Agent0'),Line2D([0],[0],color=colors[1],label='Agent1')],fontsize=8)
    fig.savefig(figures/'old_test_cases.png',dpi=160);fig.savefig(figures/'old_test_cases.svg');plt.close(fig)
    originals=out/'raw_svg_exports';originals.mkdir(exist_ok=True)
    for p in figures.glob('*.svg'):
        (originals/p.name).write_bytes(p.read_bytes())
        p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    write(out/'figure_provenance.json',dict(destination='Provisional research/group-meeting figures; publisher requirements not specified',
        distribution='All12 paired base geometry rates for every condition; no smoothing/inferential intervals.',
        cases=caseprovenance,case_rule=protocol['case_selection'],scope='Local H8 trajectories, not closed-loop routes; oracle reference uses future labels.',
        svg_processing='Remove trailing line spaces only for Git hygiene; original SVG exports preserved separately, coordinates unchanged.',
        exports={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in figures.iterdir()}))
    dest=Path(__file__).resolve().parent/'safety_generalization_results/v1';dest.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','summary.json','old_failure_summary.json','verification.json','figure_provenance.json','dataset_sha256.json']:(dest/name).write_bytes((out/name).read_bytes())
    for name in ['generalization_pairs.svg','old_test_cases.svg']:(dest/name).write_bytes((figures/name).read_bytes())
    fields=['name','base_id','split','condition','actual_initial_separation','category','safe_available','safe_candidates','predicted_safe_candidates','selected','fallback','selected_risk','selected_predicted_minimum','selected_true_minimum','best_true_safe_index','safe_cost_gap','false_safe_candidates','false_unsafe_candidates']
    for filename,rr in [('old_failure_cases.csv',old),('new_failure_cases.csv',rows)]:
        with (dest/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rr)
    lines=['| New condition | Cost-only unsafe* | Safety-rule unsafe* | No safe pool | False-accept selections | Unsafe fallback selections |','|---|---:|---:|---:|---:|---:|']
    for condition,v in summary['conditions'].items():
        lines.append(f"| {condition} | {100*v['methods']['learned_cost_only']['mean']['unsafe_given_safe']:.2f}% | {100*v['methods']['learned_safety']['mean']['unsafe_given_safe']:.2f}% | {v['no_safe_groups']} | {v['category_counts']['risk_false_accept']} | {v['category_counts']['fallback_miss']} |")
    oldtest=json.loads((out/'old_failure_summary.json').read_text())['test'];counts=summary['all_conditions']['category_counts']
    fallback_failures=[q for q in rows if q['category']=='fallback_miss']
    optimistic=sum(q['selected_predicted_minimum']>=.3 for q in fallback_failures)
    mean_sep_error=float(np.mean([q['selected_predicted_minimum']-q['selected_true_minimum'] for q in fallback_failures]))
    doc=f'''# Frozen scorer: failure diagnosis and new-scenario generalization

## Completed blocks and conclusions

1. Post-hoc diagnosis of the already-consumed validation/test data: old test has12 states, including one all-unsafe pool and one avoidable unsafe selection. The avoidable failure is a false risk acceptance; seven other states select a safe action while excluding the lowest-cost true-safe action. Categories follow a fixed hierarchy and may involve interacting mechanisms, not unique causal explanations.
2. Frozen new-scenario evaluation:12 base geometries x3 conditions x2 distances x64 candidates =4608 new labeled candidates. Existing checkpoint epoch126/threshold0.25 is unchanged, no optimizer, retraining or threshold search. New geometry specifications were saved before old-case inspection. Both post-hoc diagnostics and new results are distinct from model-selection data.

{chr(10).join(lines)}

*First average two states within each base geometry, then equally across12 base geometries. All72 states contain at least one safe candidate. In all three conditions combined there are{counts['risk_false_accept']} false-accept unsafe selections and{counts['fallback_miss']} unsafe fallback selections. Thus current new-scenario unsafe decisions are not explained by missing safe proposals. Baseline8.33% from the earlier test does not transfer: new-condition rates37.5/37.5/45.83%. Scenario distributions differ, so these are stress/generalization comparisons, not a randomized estimate of one causal factor.

Within a base geometry and distance-grid index, candidate actions and goals are EXACTLY shared across the three conditions. Velocity condition changes only initial velocity; arrival condition redistributes approach offsets while preserving velocity. The latter also changes instantaneous initial separation, so it is not a timing intervention at fixed separation; actual separation is exported in the new-case CSV. Across base geometries, angles45/75/105/135deg and full-circle orientations vary. All initial separations exceed0.30. These are offline H8 tests, not closed-loop flights or three-agent/3D validation.

## Mechanism checks

`true_risk_predicted_cost` substitutes future true safety for the risk filter; all conditional unsafe rates become0 by construction. `predicted_risk_true_cost` substitutes true soft cost but keeps the predicted risk filter/fallback; unsafe rates remain high and can worsen because the soft objective itself favors some unsafe actions. These are explicitly oracle counterfactual diagnostics, not deployable methods or additive causal attribution.

The model can both reject real safe candidates and overestimate fallback quality.19/29 new unsafe decisions occur after predicting no candidate safe, even though a safe option exists. In{optimistic}/19 such failures, the selected predicted minimum separation is nevertheless>=0.30, with mean optimistic error{mean_sep_error:.4f} VMAS length units. This is direct evidence of unreliable fallback estimates, not a guaranteed-safe rejection mechanism. Purely increasing soft-cost accuracy or adding a safety penalty is not an established repair. Coordinate/agent-order sensitivity and narrow training coverage are hypotheses suggested by raw44D input and changed distributions; no rotational-invariance intervention was performed, so they are NOT proven causes.

Recommended next repair block: keep readout/true-simulator controls, predefine invariant/relative state-action representations and explicit safe-fallback coverage, select using training-only splits, and obtain NEW confirmation scenarios. Do not tune the current checkpoint or threshold on these now-consumed stress scenarios. Navigation-specific WM materials remain missing; this is a direct scorer baseline, not LeWM transfer.

## Validation and evidence

All4608 native VMAS candidate trajectories/substeps were run twice from fresh environments, error0. Independent NumPy checks recompute feature44D, cost and safety labels, forward inference (max error{maxraw:.2e}), selected decisions, paired actions/goals and scene-mean rates; original checkpoint hash unchanged. Twenty unit tests pass. Cost formula max CPU/GPU reduction difference{maxcost:.2e}; tolerance1e-5. No rows were excluded.

`generalization_pairs.svg` shows every geometry pair, with black mean lines and fixed0–100% context. `old_test_cases.svg` uses the predeclared first-name selection rule (avoidable unsafe, all-unsafe, safe), not favorable case selection. Native-substep paths are H8 snippets, not complete goal trajectories; true-safe alternatives are oracle references. Underlying case CSVs, source protocol, figure hashes and full raw external artifacts are preserved. Raw trajectories, predictions and weights are not committed to Git.

## Reproduce

```bash
python -m unittest discover -s tests -v
python experiments/e4_navigation/safety_generalization.py --data outputs/conflict_ranking_v1 --model outputs/state_action_ranker_v1 --output outputs/safety_generalization_v1
python experiments/e4_navigation/report_safety_generalization.py --run outputs/safety_generalization_v1 --model outputs/state_action_ranker_v1 --old-data outputs/conflict_ranking_v1
```

CUDA VMAS labels and CPU scoring use the unchanged prior environment. The prior checkpoint and its portable NPZ are external prerequisites; a new output directory is required. This block consumes the new evaluation scenarios, not a new training set. No closed-loop experiments were launched.
'''
    (dest/'README.md').write_text(doc,encoding='utf-8');(out/'SAFETY_GENERALIZATION_REPORT.md').write_text(doc,encoding='utf-8');print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
