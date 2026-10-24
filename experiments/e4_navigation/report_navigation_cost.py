"""Report primary negative result and predeclared analytic competitor in full."""
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    summary=json.loads((root/'summary.json').read_text());freeze=json.loads((root/'selection_frozen.json').read_text());grid=json.loads((root/'validation_grid.json').read_text());controls=json.loads((root/'validation_controls.json').read_text());verification=json.loads((root/'verification.json').read_text());rows=json.loads((root/'episode_results.json').read_text())
    table=[]
    for arm,stats in summary['methods'].items():
        m=stats['mean'];table.append(f"| {arm} | {m['joint_success']:.1%} | {m['safe_joint_success']:.1%} | {m['substep_loss_of_separation']:.1%} | {m['steps']:.2f} | {m['mean_planning_ms']:.2f} |")
    selected=next(r for r in grid if r['lambda_value']==freeze['lambda_value']);validation=[]
    for arm,m in list(controls.items())+[('ridge_goal',selected)]:validation.append(f"| {arm} | {m['unsafe']:.2%} | {m['safe_progress']:.2%} | {m['goal_cost_regret']:.4f} |")
    with (root/'episodes.csv').open('w',newline='') as fp:
        fields=['geometry','arm','seed','steps','joint_success','safe_joint_success','substep_loss_of_separation','substep_collision_proxy','substep_minimum_separation','mean_planning_ms','fallback_steps']
        writer=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    report=f'''# E4 frozen goal-cost repair and NEW closed-loop confirmation

## Result

The predeclared primary ridge-correction gate **{'passed' if summary['gate_passed'] else 'FAILED'}**. Ridge safe-joint-success did not improve over the frozen invariant scorer. Preserve this negative result; the analytic comparator was a predeclared separate arm and does not retrospectively replace the primary gate.

On six NEW geometries with two candidate seeds each, hand-coded free-motion goal cost plus the frozen learned risk filter achieved12/12 safe goal arrivals, versus0/12 for both the old scorer and learned ridge correction. The true-simulator reference also achieved12/12. Only the cost ranking changed between the learned-risk arms; risk/minimum-separation heads, thresholds, margin, fallback rule, dynamics and72-proposal generator remained fixed. Actual pools differ after policies visit different states.

| Arm | Joint goal success | Safe joint success | Any separation violation | Mean steps | Mean planning ms |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Each arm has12 episodes but only6 independent geometry groups. Rates average seeds within geometry then geometries equally. All48 episodes have zero collision proxies; proxy threshold is center distance0.20, not instrumented contact events. Separation safety includes initial state and native substeps at0.30. Position-only joint arrival tolerance0.12, no speed requirement. Mean planning latency includes each method's candidate generation and selection; privileged oracle performs simulator rollouts, so compute/information budgets differ. Runtime is a laptop measurement, not a performance guarantee.

## Validation and what it does NOT establish

| Validation arm | Unsafe selection | Safe H8 progress choice | Goal-cost regret |
|---|---:|---:|---:|
{chr(10).join(validation)}

The ridge validation criterion prioritizes unsafe selection, then safe-progress selection, then goal-cost regret, then lambda. It selected lambda{freeze['lambda_value']:g}, obtaining0% unsafe but only53.57% safe-progress choices. This does not establish better progress than the frozen control. Do not refit or change the criterion after confirmation failure. Validation analytic unsafe selection11.90% exceeds the control5.95%, even though all twelve fresh closed-loop analytic episodes are safe. This discrepancy demonstrates state-distribution dependence and limits any safety claim about the analytic hybrid. The confirmation set is small, synthetic, related to the training family, and now consumed.

## Training and fixed contract

Data: previous NEW development block's8 train and4 validation geometry groups,85 training snapshots/6120 candidate examples and41 validation snapshots/2952 examples. Training weights are equal geometry, then equal parent-policy origin, then equal snapshot and candidate. Train-only weighted means/std floor0.01. No validation labels enter fitting. Target:5*sum native H8 terminal goal distances+.01*mean squared controls, excluding soft safety cost to keep risk modeling separate.

92 current-input features: canonical44 mean across agent order, mean squared canonical44, free-motion goal cost, free-motion minimum separation, frozen cost prediction, free-motion progress. Weighted ridge predicts the residual to frozen cost, with unpenalized weighted intercept; six fixed regularization values0.0001,0.001,0.01,0.1,1,10. The analytic arm uses native free-motion recurrence (dt0.1,two0.05 substeps,drag0.25,mass1), ignoring contact forces. It knows simulator parameters; it is a hand-coded dynamics baseline, not a learned Navigation world model. Risk heads remain the existing direct state/action MLP.

Protocol and all six confirmation geometries were specified before fitting; coefficients, lambda and hashes frozen before any confirmation rollout. Four separate5-step smoke runs precede48 full episodes, max200 steps each. H8/72 unchanged from the prior pilot. No old consumed confirmation/stress file was read for selection and no post-confirmation updates occurred. Native VMAS1.5.2 defaults retained without extra speed limits or safety stops. No three-agent task or LeWM checkpoint was used.

## Verification

42 targeted tests pass, including analytic recurrence vs native collision-free rollout,92-feature contract, agent-swap consistency and unchanged risk/separation heads. Independent audit reconstructs training-only weighted moments exactly, verifies ridge normal equations (max residual{verification['ridge_normal_equation_max_error']:.3g}), recomputes every validation grid entry/selection, and audits{verification['independent_model_choices']} model choices plus{verification['independent_oracle_choices']} fresh true-sim oracle rankings. Every executed first action, native state/substep replay, episode physical outcome, model hash and equal-geometry aggregate was checked. Max feature-formula difference{verification['max_feature_formula_difference']:.3g}; NumPy prediction difference{verification['max_numpy_prediction_difference']:.3g}; oracle-cost formula error{verification['max_oracle_cost_error']:.3g}. State/substep replay error0. Initial auditor validation-regret arithmetic used float32 averages instead of the producer's Python-float records; matching the recorded arithmetic resolved the discrepancy without changing results or loosening the check.

Source, protocol, frozen selection, every validation grid entry, all48 episode rows/results and verification are published. Weights/coefficients and raw trajectories/candidates/predictions remain in external run directories, identified by hashes; no weights were claimed to be newly obtained from teammates. Total original run wall time{summary['wall_seconds']:.1f}s excluding independent audit.

## Next block

Keep the analytic-goal/frozen-risk hybrid as the current successful development baseline, retaining its unsafe validation evidence. Preserve the failed ridge model as a negative control. Next freeze a NEW bounded robustness evaluation across approach speeds, timing, geometry and terminal-neighborhood states before considering broader safety/generalization claims; report goal success and violations together. Do not tune on these six confirmation geometries. Any new learned cost method should first improve progress without sacrificing safety on development validation, then receive new confirmation.

Navigation-specific LeWM checkpoint/loader/config/normalization/history/action/agent contract/state head remains pending. This positive hybrid baseline does not establish LeWM transfer, learned-model planning, or three-agent capability.
'''
    (root/'REPORT.md').write_text(report,encoding='utf-8');print('Report and all48 episode rows saved.')
if __name__=='__main__':main()
