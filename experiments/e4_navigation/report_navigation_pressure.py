"""Pressure report preserving the failure of the previously successful hybrid."""
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    summary=json.loads((root/'summary.json').read_text());verification=json.loads((root/'verification.json').read_text());rows=json.loads((root/'episode_results.json').read_text());diag=json.loads((root/'initial_pool_diagnostics.json').read_text())
    table=[]
    for condition,arms in summary['by_condition'].items():
        for arm,stats in arms.items():
            m=stats['mean'];table.append(f"| {condition} | {arm} | {m['joint_success']:.1%} | {m['safe_joint_success']:.1%} | {m['substep_loss_of_separation']:.1%} |")
    failure='\n'.join(f"| {r['geometry']} | {r['initial_pool_safe_candidates']} | {r['selected_initial_H8_true_minimum']:.4f} | {r['selected_initial_predicted_risk']:.4f} | {r['selected_initial_predicted_minimum']:.4f} | {r['first_violation_execution_step']} |" for r in diag if r['condition']=='high_inertia')
    with (root/'episodes.csv').open('w',newline='') as fp:
        fields=['base_id','condition','geometry','arm','seed','steps','joint_success','safe_joint_success','substep_loss_of_separation','substep_collision_proxy','substep_minimum_separation','mean_planning_ms','initial_pool_safe_candidates','initial_pool_best_minimum']
        writer=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    report=f'''# Frozen Navigation pressure screen: high-inertia failures

## Result

Predeclared stress screen **{'passed' if summary['stress_gate_passed'] else 'FAILED'}**. The analytic-goal/frozen-risk hybrid is safe6/6 in the near-conflict condition but safe0/6 in high inertia, despite reaching goals6/6. True-sim safety reference is safe6/6 in both. The previous four-condition24/24 success cannot be generalized to this harder state distribution.

| Condition | Arm | Joint arrival | Safe joint arrival | Any separation violation |
|---|---|---:|---:|---:|
{chr(10).join(table)}

All collision proxies remain0. A separation violation is distance<0.30 at initial/native-substep states; collision proxy is center distance<0.20. Neither zero collisions nor successful arrival cancels a separation violation. Position-only joint goal tolerance0.12, no terminal-speed constraint.

## Protocol

6 NEW independent base geometries seed2026105201, angles75..135deg/full orientations, two correlated variants each, one candidate seed per base:near conflict initial separation0.42/speeds0.10,0.20; high inertia separation0.36/speeds0.35,0.50.12 variants x3 arms=36 episodes, max200 steps,H8/72. Native VMAS1.5.2 dt0.1/two substeps/radius0.1/drag0.25 unchanged. Both variants initially satisfy separation>=0.30. Conditions jointly change speed and separation, so this is not an isolated velocity causal estimate. Only6 independent clusters, not12.

No fitting/threshold/physics/proposal changes or old consumed confirmation data used. Only published portable direct scorer JSON was loaded; no author or Navigation WM checkpoint. Same initial candidate pools across all3 arms independently verified. All six bases and their variants are now consumed evaluation data and must not tune a later model.

## Post-hoc initial-pool diagnostics

Each high-inertia initial pool has1..4 truly safe H8 candidates, while the analytic policy initially selects a predicted-accepted but truly unsafe H8 sequence. Predicted risk remains below the frozen0.5 threshold and predicted minimum separation exceeds0.30. Actual episode violations begin at zero-based execution step1 or2. These results expose unsafe acceptance/optimistic separation under high inertia. The local H8 label alone does not prove an executed first action must violate; full trajectories supply the actual outcome. Finite candidate coverage is not proof of global physical feasibility or optimality.

| High-inertia geometry | True-safe options initially | Chosen true H8 minsep | Predicted risk | Predicted minsep | First violation step (0-based) |
|---|---:|---:|---:|---:|---:|
{failure}

Do not remove the safety filter, tune a threshold on these six cases, or report prior successes without this negative screen. New risk calibration/training needs separate NEW development geometry and then a fresh frozen confirmation set.

## Verification

51 targeted tests pass after upstream readout synchronization. Independent NumPy audits{verification['independent_model_choices']} model choices, fresh native rollout audits{verification['independent_oracle_choices']} oracle choices, and all{verification['executed_steps']} executed first actions/physical outcomes/condition aggregates. Max NumPy forward error{verification['max_numpy_forward_error']:.3g}, oracle-cost formula error{verification['max_oracle_cost_error']:.3g}. State and native-substep action replay error0; paired initial pools exact; portable model hash unchanged. Original run wall{summary['wall_seconds']:.1f}s excluding independent audit.

Raw arrays remain external; per-episode tables, protocol, diagnostics, hashes and verification are published. No learning-world-model, three-agent or universal-safety claim.
'''
    (root/'REPORT.md').write_text(report,encoding='utf-8');print('Pressure report and36 episode rows saved.')
if __name__=='__main__':main()
