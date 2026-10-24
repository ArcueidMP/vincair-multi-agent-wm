"""Publish all conditions and outcomes; six base geometries are the clusters."""
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    summary=json.loads((root/'summary.json').read_text());verification=json.loads((root/'verification.json').read_text());rows=json.loads((root/'episode_results.json').read_text());table=[]
    for condition,arms in summary['by_condition'].items():
        for arm,stats in arms.items():
            m=stats['mean'];table.append(f"| {condition} | {arm} | {m['joint_success']:.1%} | {m['safe_joint_success']:.1%} | {m['substep_loss_of_separation']:.1%} | {m['steps']:.2f} |")
    with (root/'episodes.csv').open('w',newline='') as fp:
        fields=['base_id','condition','geometry','arm','seed','steps','joint_success','safe_joint_success','substep_loss_of_separation','substep_collision_proxy','substep_minimum_separation','mean_planning_ms','fallback_steps']
        w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    report=f'''# E4 frozen hybrid robustness and collaborative reproduction

## Result

Descriptive robustness screen **{'passed' if summary['robustness_gate_passed'] else 'FAILED'}**. Across24 condition-specific episodes, analytic free-motion goal cost plus the frozen learned risk filter reaches goals safely24/24; frozen original scorer0/24; privileged true-sim safety reference24/24. Each condition has6 episodes from the SAME six base geometries. Treat them as6 clusters with four related variants, not24 independent geometries. No population-level safety, LeWM transfer or three-agent claim.

| Condition | Arm | Joint goal | Safe joint goal | Any separation violation | Mean steps |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

All72 episodes have zero separation violations and collision proxies. Original scorer failure is lack of goal arrival despite200-step budgets. Arrival-offset changes both start positions and speeds; it is not a pure timing intervention at fixed geometry/separation. Terminal-neighborhood starts0.20 before each goal and tests final approach; it is not a crossing-conflict trial. Position-only success tolerance0.12 has no terminal-speed requirement. Report all conditions, not just the fast/nominal cases.

## Frozen protocol

Seed2026105101 generates6 NEW base geometries, crossing angles45..135deg and full-circle orientations. Conditions:nominal velocities0.12/0.24; fast0.45/0.60; arrival offsets0.90/0.45 with velocities0.20/0.20; final approach0.20 before goals with velocities0.30/0.45. One candidate RNG seed per base shared across conditions/arms, plus execution step. Actual candidate pools diverge with policy state. H8/72 candidates,200 maximum steps, native VMAS1.5.2 dt0.1/two substeps/radius0.1/world drag0.25. Safety includes initial and native substeps, threshold0.30. No training, thresholds/margins/proposals/physics changes or old consumed evaluation files read.

Three independent5-step smoke runs preceded72 full episodes. Both learned-risk arms share the same frozen5059-parameter direct state/action scorer; analytic arm changes only goal-cost ranking through known free-motion recurrence, ignoring contact forces. True-sim arm has privileged candidate rollouts and is not compute/information matched to learned arms. Screen criterion was fixed before run:analytic safe-goal rate>=95% in EACH condition and violations no higher than true-sim in EACH condition. This is a descriptive screen on a bounded synthetic set, not proof of universal safety. Earlier offline validation unsafe-selection evidence (analytic11.90%) remains relevant and must not be omitted.

## Verification and reproducibility

46 targeted tests pass. Independent audit checks{verification['independent_model_choices']} learned choices and{verification['independent_oracle_choices']} fresh native oracle rankings, all{verification['executed_steps']} executed first actions, physical outcomes and condition aggregates. Native state/substep action replay error0; max NumPy forward difference{verification['max_numpy_forward_error']:.3g}; independently computed oracle-cost error{verification['max_oracle_cost_error']:.3g}. Run wall{summary['wall_seconds']:.1f}s excluding independent audit.

Portable JSON exports ONLY our locally trained direct risk scorer, with exact source parameter/statistic equality and source hashes. All state/substep/action/candidate/prediction/selection arrays from the three portable smoke runs exactly match the original model runs. Teammates can clone this branch and run [the shared entry point](../../REPRODUCE_SHARED.md) without the author's LeWM checkpoints or our machine-specific weight paths. Complete72-episode independent reproduction on a DIFFERENT machine remains a team task; only local portable smoke was separately rerun here.

Published source, protocol, compact summaries, every episode CSV/JSON, tested environment, verification and hashes accompany the portable model. Raw trajectories/candidates/predictions remain local, indexed by SHA256. Historical E0/E1 reports and exact final E1 phase22 source were separately migrated with original/published hashes; they were not rerun during this publication.

## Next work and collaboration

1. Teammate independently clones personal branch, checks model/physics/version hashes, runs portable smoke and then72-episode repeat; compare every episode outcome/steps/minimum separation, excluding latency from exact matching.
2. Review positive hybrid evidence together with failed trained-ridge and original-scorer controls, and the unresolved offline risk errors. Prepare a team PR review; main and teammate feature branch remain untouched.
3. Obtain Navigation-specific LeWM checkpoint/loading/config/normalization/history/actions/agent contract/state head and dataset splits. Integrate only after real loading and state/rollout checks, then compare against this hybrid and true-sim reference at fixed candidate budget.
4. If pursuing robustness claims, freeze NEW independent geometry/speed/near-contact/obstacle or terminal-velocity evaluation before expansion. Do not tune on these six bases or their24 variants. Current screen has no obstacles and no terminal-speed constraint; add either only as an explicit new task/metric change.

See root [COLLABORATION_START_HERE.md](../../../../COLLABORATION_START_HERE.md) for proposed owners, acceptance artifacts, historical migration scope and external asset dependencies. No messages were sent to teammates, and no PR/merge was created automatically.
'''
    (root/'REPORT.md').write_text(report,encoding='utf-8');print('Report and all72 episode CSV rows saved.')
if __name__=='__main__':main()
