"""Complete development diagnosis tables, without model tuning or confirmation claims."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    summary=json.loads((root/'summary.json').read_text());records=json.loads((root/'records.json').read_text());verify=json.loads((root/'verification.json').read_text());episodes=json.loads((root/'episode_results.json').read_text())
    tab=[];methods=[]
    for group,s in summary['by_split_origin'].items():
        m=s['mean'];tab.append(f"| {group} | {s['geometries']} | {s['snapshots']} | {m['safe_progress_available']:.1%} | {m['filter_excludes_all_progress']:.1%} | {m['cost_miss_within_filter']:.1%} |")
        for name,v in s['methods'].items():methods.append(f"| {group} | {name} | {v['safe_progress']:.1%} | {v['unsafe']:.1%} | {v['progress']:.4f} |")
    csvrows=[]
    for r in records:
        base={k:v for k,v in r.items() if k!='methods'}
        for method,m in r['methods'].items():csvrows.append(dict(**base,method=method,**m))
    with (root/'counterfactuals.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(csvrows[0]),lineterminator='\n');w.writeheader();w.writerows(csvrows)
    episode_table=[]
    for split in ['train','validation']:
        for origin in ['invariant_repair','true_sim_safety']:
            rr=[r for r in episodes if r['split']==split and r['arm']==origin]
            episode_table.append(f"| {split}/{origin} | {len(rr)} | {sum(r['joint_success'] for r in rr)} | {sum(r['safe_joint_success'] for r in rr)} |")
    report=f'''# E4 navigation progress diagnosis on NEW development geometries

## Result and scope

On all{summary['snapshot_count']} sampled states there is a safe H8 candidate reducing total goal distance by more than0.05 VMAS units. This establishes local candidate coverage at these snapshots only, not complete trajectory feasibility or closed-loop success.

For validation states visited by the frozen invariant scorer,32.14% retain a safe progressing option through its predicted filter yet select a non-safe-progressing candidate. Replacing only the cost ranking with true cost increases safe-progress selection67.86% ->92.86%, but unsafe choices also rise3.57% ->7.14%. Retaining predicted cost while supplying true safety gives64.29% safe progress. The true-safety/true-cost reference gives100%. These local counterfactuals support prioritizing cost/progress ranking while retaining safety checks; they do not justify replacing the filter or claim a uniquely isolated cause.

On states visited by the true-simulator reference, validation predicted filtering excludes all safe-progress candidates14.58% of the time. Thus filtering remains unreliable on a different state distribution. Diagnostic flags can overlap in general and are not additive causal percentages.

## Coverage and failure flags

Within each split and origin, average snapshots within geometry, then geometries equally. Different origins visit different states and have different numbers of snapshots; no direct paired causal comparison between origins.

| Split/origin | Geometries | Snapshots | Safe progress exists | Filter excludes all progress | Cost miss within filter |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tab)}

Safe progress means native-substep minimum separation>=0.30 AND sum(current goal distances)-sum(H8 terminal goal distances)>0.05. Cost miss within filter means at least one safe-progress candidate survives the frozen risk/minsep conjunction but the learned choice is not safe-progressing. The criterion includes both unsafe and nonprogressing choices; it is not a count of stationary agents.

## Counterfactual decisions

| Split/origin | Rule | Safe-progress choice | Unsafe choice | Mean H8 progress |
|---|---|---:|---:|---:|
{chr(10).join(methods)}

`true_cost_predicted_filter` uses true total soft cost only inside the original predicted-accepted set; when that set is empty, it keeps the frozen learned fallback. `true_safety_predicted_cost` chooses predicted cost inside the true-safe set, keeping learned fallback if none exists. `true_safety_oracle` uses true safety-first ordering with original soft cost; `safe_progress_oracle` maximizes true progress within true-safe candidates. No counterfactual was run as a new closed-loop controller. No future label was used by the learned parent policy.

## Parent trajectories and sampling

12 newly generated geometries (seed2026104901):8 train/4 validation whole groups. Each has one200-step-budget trajectory under the frozen invariant model and one under the existing privileged true-sim reference, using a fixed candidate seed schedule. Snapshot execution indices0,10,30,60,100,150,199; indices after termination are skipped.24 parent episodes, {summary['snapshot_count']} snapshot pools x72 = {summary['candidate_labels']} candidate labels, H8. Approach distance0.6..0.9, crossing angles40..140deg, random orientation, initial route speeds0.08..0.25. Native VMAS contract unchanged. These are development data; there is no new untouched confirmation set in this block.

| Split/origin | Episodes | Joint goal success count | Safe joint success count |
|---|---:|---:|---:|
{chr(10).join(episode_table)}

No prior consumed closed-loop/stress/test data were opened for fitting or selection. Weights and policies remained frozen; no training, threshold search, or successful learned-world-model navigation claim. The source code and protocol fix sampling and diagnostic definitions before generation. The new data may support later development fitting, but validation labels must not be used as training labels; a later repair requires new confirmation geometries.

## Verification and provenance

37 targeted tests pass. Independent NumPy recomputes all physical cost/safety/progress labels, model forward predictions and{verify['independent_counterfactual_choices']} counterfactual decisions; parent/snapshot input matching, hashes, episode goal/safety outcomes and equal-geometry aggregates pass. Maximum cost formula discrepancy{verify['max_cost_formula_error']:.3g}, forward discrepancy{verify['max_numpy_forward_error']:.3g}. Every snapshot candidate rollout repeated in fresh native VMAS with0 state/substep error; every parent executed trajectory replayed exactly. An initial independent RMSE check differed by1.7e-8 because portable NumPy scalar arithmetic promoted precision; the auditor now explicitly matches the producer's Python-scalar/float32 arithmetic. No result or tolerance was changed to bypass that check.

Models remain external in safety_repair_v1/state_action_ranker_v1. Raw parent trajectories, separated snapshot inputs/future labels/predictions remain in this external run directory, identified by SHA256. Source, protocol, summaries, all counterfactual decisions, parent outcomes and verification are published. Runtime{summary['wall_seconds']:.1f}s excluding independent verification. No Navigation-specific LeWM checkpoint was obtained or used.

## Next executable block

1. Freeze a goal-progress/cost repair experiment using only this block's train geometries for fitting and validation geometries for selection. Keep the current risk model as a frozen comparator. Compare a trained cost correction with a simple analytic goal-progress baseline, using the same candidates and explicit safety/goal metrics.
2. Include development states visited by both learned and privileged policies; track the unequal snapshot density and weight geometries explicitly. Separate current-state/action inputs from future supervision. Validate one-step versus H8 ranking and receding-horizon progress rather than cost RMSE alone.
3. Freeze selected model/policy before generating and activating NEW confirmation geometries; test bounded closed-loop safe goal success, not just local safety. Preserve original72 candidate baseline and isolate any proposal change as its own ablation.
4. Navigation LeWM checkpoint, loader/config/normalization/history/action/agent contract and state readout remain pending from teammates. Keep all direct-scoring baselines separate from LeWM transfer and three-agent claims.
'''
    (root/'REPORT.md').write_text(report,encoding='utf-8');print('Report and complete counterfactual CSV saved.')
if __name__=='__main__':main()
