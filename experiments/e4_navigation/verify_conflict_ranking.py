"""Independent NumPy labels/preferences audit. Test has integrity checks only."""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();root=args.run
    manifest=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'summary.json').read_text());assert summary['status']=='passed'
    groups={};paircounts={'train':0,'validation':0};maxcost=0.;maxsep=0.;hashes={};allgroups=0
    for row in manifest:
        gid=row['geometry_id'];groups.setdefault(gid,set()).add(row['split']);directory=root/row['split']/row['name']
        for name,key in [('inputs.npz','input_sha256'),('labels.npz','labels_sha256')]:
            h=sha(directory/name);assert h==row[key];hashes[str((directory/name).relative_to(root))]=h
        with np.load(directory/'inputs.npz',allow_pickle=False) as inp,np.load(directory/'labels.npz',allow_pickle=False) as lab:
            assert set(inp.files)=={'initial_state','goals','actions','candidate_mode'}
            initial=inp['initial_state'];actions=inp['actions'];goals=inp['goals'];state=lab['state_path'];sub=lab['substep_state_path']
            assert actions.shape==(64,8,2,2) and state.shape==(64,9,2,4) and sub.shape==(64,16,2,4)
            assert np.isfinite(state).all() and np.isfinite(actions).all() and abs(actions).max()<=1.000001
            assert np.array_equal(state[:,0],np.repeat(initial[None],64,axis=0)) and np.array_equal(sub[:,1::2],state[:,1:])
            assert np.linalg.norm(initial[0,:2]-initial[1,:2])>.30
            dist=np.linalg.norm(state[:,1:,0,:2]-state[:,1:,1,:2],axis=-1)
            goal=np.linalg.norm(state[:,-1,:,:2]-goals[None],axis=-1).sum(1)
            safety=np.maximum(.30-dist,0)**2;act=(actions**2).mean((1,2,3));cost=5*goal+100*safety.sum(1)+.01*act
            maxcost=max(maxcost,float(abs(cost-lab['total_cost']).max()));assert np.allclose(cost,lab['total_cost'],rtol=1e-5,atol=1e-5)
            assert np.array_equal((dist<.30).sum(1),lab['violation_steps'])
            suball=np.concatenate([state[:,:1],sub],axis=1);ds=np.linalg.norm(suball[:,:,0,:2]-suball[:,:,1,:2],axis=-1)
            minimum=ds.min(1);exposure=(np.maximum(.30-ds,0)**2).sum(1)
            maxsep=max(maxsep,float(abs(minimum-lab['substep_minimum_separation']).max()))
            assert np.allclose(minimum,lab['substep_minimum_separation'],rtol=0,atol=1e-7)
            assert np.allclose(exposure,lab['substep_squared_violation'],rtol=1e-5,atol=1e-7)
            # Verify ordering by a Python tuple comparator, independent of producer's lexsort.
            savedcost=lab['total_cost'];sm=lab['substep_minimum_separation'];se=lab['substep_squared_violation']
            def key(k):return (0,float(savedcost[k]),k) if sm[k]>=.30 else (1,-float(sm[k]),float(se[k]),float(savedcost[k]),k)
            order=np.array(sorted(range(64),key=key));assert np.array_equal(order,lab['safety_order'])
            rank=np.argsort(order);assert np.array_equal(rank,lab['safety_rank'])
            if row['split'] in paircounts:
                with np.load(directory/'preferences.npz',allow_pickle=False) as z:
                    p=z['preferred_pairs'];k=z['kind'];assert p.ndim==2 and p.shape[1]==2 and len(k)==len(p)
                    assert (rank[p[:,0]]<rank[p[:,1]]).all() and (p>=0).all() and (p<64).all()
                    safe=sm>=.30;cross=safe[p[:,0]]!=safe[p[:,1]]
                    assert np.array_equal(cross,k==0)
                    assert safe[p[k==0,0]].all() and (~safe[p[k==0,1]]).all()
                    assert (safe[p[k==1]]).all() and (~safe[p[k==2]]).all()
                    paircounts[row['split']]+=len(p)
                hashes[str((directory/'preferences.npz').relative_to(root))]=sha(directory/'preferences.npz')
            else:assert not (directory/'preferences.npz').exists()
        allgroups+=1
    assert all(len(s)==1 for s in groups.values()) and allgroups==84
    assert paircounts==summary['preference_pair_counts']
    assert len([g for g,s in groups.items() if s=={'train'}])==12 and len([g for g,s in groups.items() if s=={'validation'}])==3 and len([g for g,s in groups.items() if s=={'test'}])==3
    report=dict(status='passed',groups=84,candidates=5376,max_cost_formula_abs_error=maxcost,max_substep_separation_abs_error=maxsep,
        scene_split_disjoint=True,inputs_targets_separate=True,preference_pairs=paircounts,preference_order_independent_comparator=True,
        native_fresh_replay_error=summary['max_native_replay_error'],test_method_evaluations=0,test_preferences_generated=False)
    write(root/'verification.json',report);write(root/'dataset_sha256.json',hashes)
    dest=Path(__file__).resolve().parent/'conflict_results/v1';dest.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','summary.json','verification.json','pilot_comparison.json','manifest.json','dataset_sha256.json']:
        (dest/name).write_bytes((root/name).read_bytes())
    diagnostics=json.loads((root/'group_diagnostics.json').read_text())
    write(dest/'group_diagnostics.json',diagnostics)
    lines=['| Split | Geometries | States | Mixed states | No safe candidate | Soft unsafe selection* | Safety-first oracle* |','|---|---:|---:|---:|---:|---:|---:|']
    for split,v in summary['splits'].items():lines.append(f"| {split} | {v['geometries']} | {v['groups']} | {v['mixed_groups']} | {v['no_safe_groups']} | {100*v['soft_unsafe_given_safe_available_scene_mean']:.2f}% | {100*v['safety_unsafe_given_safe_available_scene_mean']:.2f}% |")
    doc=f'''# Conflict ranking dataset and oracle audit v1

Reconstructed from teammate notebook04 at commit a4ff326. No original private NPZ was downloaded: these are newly generated, provenance-recorded artifacts. The12 public pilot states match cached unsafe candidate counts, minimum step separation and both best indices. Total5376 candidates: public pilot768 and new geometry-grouped data4608.

{chr(10).join(lines)}

*Conditional on at least one safe candidate; first average within geometry over eligible offsets, then equally over eligible geometries. Zero safety-first oracle error follows from the priority definition; it is not learned-model performance. No test method metrics are computed. Train has48 groups/3072 candidates, validation12/768, sealed test12/768. Geometry rotations/translations/scales differ across splits; all four offsets and64 candidates stay within one geometry group. Route families remain shared: this is not held-out-family generalization.

## Targets

Preserve original goal/safety/action weighted soft cost5/100/.01 using environment-step states. New safety labels include the initial state and all native integration substeps at separation threshold0.30. Safe candidates precede unsafe candidates; safe candidates sort by soft cost. When all are unsafe, maximize minimum separation, then minimize squared violation exposure, soft cost, and stable index. No feasible action is invented. Never describe an all-unsafe fallback as safe.

Training/validation preference kinds:0 safe-vs-unsafe,1 safe cost,2 unsafe robust fallback. Exactly tied candidates do not produce preference pairs. Pairs:train{paircounts['train']}, validation{paircounts['validation']}; group-correlated, not independent sample counts. No test preferences were generated.

## Validation and storage

All5376 candidates were rolled out twice in fresh native VMAS environments, state/substep replay error0. Independent NumPy verification checks separated inputs, bounds, native substep endpoints, cost/separation labels, tuple-based ordering, every preference direction and geometry splits. Maximum soft-cost formula difference{maxcost:.2e}; tolerance1e-5 absolute/relative accounts for GPU/CPU floating reductions. Original data were not altered. Nine targeted unit tests (including the prior4 reproduction tests) pass.

The labels are true-simulator/oracle supervision. No Navigation weights were trained or evaluated; no closed-loop claim. Finite H8 safe candidates do not guarantee safety beyond the window. Raw data remain external with SHA256 manifest; committed files are source, contracts and audit tables. `inputs.npz` contains only initial state/goals/actions/mode, and `labels.npz` contains future state/substep trajectories and supervision. Never use labels as online model features.

## Reproduce

```bash
python -m unittest discover -s tests -v
python experiments/e4_navigation/conflict_ranking.py --output outputs/conflict_ranking_v1
python experiments/e4_navigation/verify_conflict_ranking.py --run outputs/conflict_ranking_v1
```

Current entry point uses CUDA with VMAS1.5.2 to match the public pilot. The protocol is saved before collection, all original source functions come from the checked-in notebook, and existing outputs are preserved/rejected. The test split is generated and integrity-checked only, not used for target tuning, model selection or method metrics. A Navigation checkpoint, compatible state readout/normalization and actual learned adapter remain the next material dependency.
'''
    (dest/'README.md').write_text(doc,encoding='utf-8');(root/'CONFLICT_RANKING_REPORT.md').write_text(doc,encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
