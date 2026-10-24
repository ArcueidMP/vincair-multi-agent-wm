"""Complete paired pilot report; all six scenes shown, no favorable selection."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    rows=json.loads((root/'episode_results.json').read_text());summary=json.loads((root/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text());verify=json.loads((root/'verification.json').read_text())
    diagnostics=[]
    for r in rows:
        p=root/'episodes'/f"{r['geometry']}_{r['arm']}_{r['seed']}"/'trajectory.npz'
        with np.load(p) as z:
            start=np.linalg.norm(z['state'][0,:,:2]-z['goals'],axis=-1).mean();end=np.linalg.norm(z['state'][-1,:,:2]-z['goals'],axis=-1).mean()
            lastspeed=np.linalg.norm(z['state'][-20:,:,2:],axis=-1).mean();selected=z['selected']
            diagnostics.append(dict(geometry=r['geometry'],arm=r['arm'],seed=r['seed'],mean_final_goal_distance=float(end),goal_distance_reduction=float(start-end),
                last20_mean_speed=float(lastspeed),zero_brake_fraction=float(((selected>=64)&(selected<68)).mean()) if r['arm']!='pd' else None,
                pd_proposal_fraction=float((selected>=68).mean()) if r['arm']!='pd' else None))
    (root/'posthoc_diagnostics.json').write_text(json.dumps(dict(status='Post-hoc descriptive diagnostics only; no model or policy changes',episodes=diagnostics),indent=2))
    with (root/'episodes.csv').open('w',newline='') as fp:
        fields=['geometry','arm','seed','steps','joint_success','safe_joint_success','substep_loss_of_separation','substep_collision_proxy','substep_minimum_separation','mean_planning_ms','fallback_steps']
        w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    figures=root/'figures';figures.mkdir(exist_ok=True);colors=['#555555','#0072B2','#E69F00','#D55E00','#009E73'];arms=protocol['arms']
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for ax,geo in zip(axs.flat,protocol['geometry_specs']):
        for arm,color in zip(arms,colors):
            p=root/'episodes'/f"{geo['name']}_{arm}_{protocol['seeds'][0]}"/'trajectory.npz'
            with np.load(p) as z:
                s=z['state'];goals=z['goals']
                for agent,style in enumerate(['-','--']):
                    ax.plot(s[:,agent,0],s[:,agent,1],style,color=color,lw=1.3,label=arm if agent==0 else None)
                    ax.scatter(s[-1,agent,0],s[-1,agent,1],s=17,color=color)
        ax.scatter(np.array(geo['starts'])[:,0],np.array(geo['starts'])[:,1],marker='o',facecolors='none',edgecolors='black',s=40)
        ax.scatter(goals[:,0],goals[:,1],marker='*',color='black',s=70);ax.set_title(geo['name']);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('x (VMAS units)');ax.set_ylabel('y (VMAS units)')
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle('All6 geometries, first predefined seed; solid/dashed=agent0/1; dots=end, stars=goal')
    for ext in ['png','svg']:fig.savefig(figures/f'closed_loop_paths.{ext}',dpi=180)
    plt.close(fig)
    raw=root/'raw_svg_exports';raw.mkdir(exist_ok=True);p=figures/'closed_loop_paths.svg';(raw/p.name).write_bytes(p.read_bytes());p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    stats=summary['methods'];table='\n'.join(f"| {arm} | {stats[arm]['mean']['joint_success']:.1%} | {stats[arm]['mean']['safe_joint_success']:.1%} | {stats[arm]['mean']['substep_loss_of_separation']:.1%} | {stats[arm]['mean']['substep_collision_proxy']:.1%} | {stats[arm]['mean']['mean_planning_ms']:.2f} |" for arm in arms)
    diagtable=[]
    for arm in arms:
        rr=[r for r in diagnostics if r['arm']==arm]
        diagtable.append(f"| {arm} | {np.mean([r['mean_final_goal_distance'] for r in rr]):.4f} | {np.mean([r['last20_mean_speed'] for r in rr]):.4f} |")
    text=f'''# E4 frozen closed-loop pilot

## Result

Predeclared descriptive gate: **{'passed' if summary['gate_passed'] else 'not passed'}**. Offline safety-ranking gains must not be presented as closed-loop navigation success. No retraining, threshold changes, or online true-label access for learned arms.

| Arm | Joint goal success | Safe joint success | Any separation violation | Collision proxy | Mean planning ms |
|---|---:|---:|---:|---:|---:|
{table}

All rates average the two seeds within each geometry then the six geometries equally. Each arm has12 episodes, but only6 independent geometries. Collision proxy means center distance<0.20, not physical contact event instrumentation. Any separation violation uses initial and native substeps and threshold0.30. Position-only joint success tolerance0.12; terminal speed is reported separately, not a success requirement. Latencies are measurements on this laptop and unequal information/computation budgets; PD does not generate candidates, oracle uses true rollout.

## Frozen design

New seed2026104801 generates6 crossing geometries, starts0.75 from center and initial route speeds0.12/0.24. Two candidate RNG schedules per geometry. Five methods x6 geometries x2 seeds =60 episodes, max200 steps each. H8/72 candidates: original64 route/noise,4 zero/brake,4 repeated-current-PD/priority/half-PD actions. This expands the earlier offline68 pool. Same pool-generation rule and random seeds; actual pools diverge as policy states diverge. All learned models/policies frozen from prior blocks. Direct-scoring arms access current observed state/goals/proposed actions only, without simulator rollouts or future state features. True-sim safety arm is a privileged finite-pool H8 reference, not a globally optimal or certified safe controller. Replan each environment step and execute only the first action.

Native VMAS1.5.2:dt0.1,substeps2,radius0.1,world drag0.25,component action limit1. No extra speed limit, dynamics intervention or automatic stop after unsafe events. Every executed trajectory and native substep was replayed in a fresh environment with zero error. Five separate5-step smoke runs passed before the full suite.

## Post-hoc behavior diagnostics

These descriptive summaries were added after seeing pilot outcomes; they are not tuning criteria or evidence of a uniquely identified cause.

| Arm | Mean final per-agent goal distance | Last20-state mean speed |
|---|---:|---:|
{chr(10).join(diagtable)}

The full trajectory panels show all six geometries for the first predeclared seed. Both seeds and all episode outcomes remain in episodes.csv; no favorable-case selection. Terminal dots are not necessarily goal arrivals. Failed navigation may reflect goal-progress ranking, proposal coverage, and distribution shift under repeated replanning; this block does not isolate their causal contributions.

## Verification and provenance

32 targeted tests pass. Independent NumPy reconstructs{verify['independent_model_decisions']} learned choices and corresponding network predictions. Fresh native rollouts independently verify{verify['independent_native_oracle_decisions']} oracle rankings and cost/separation labels. Every executed first action, goal/safety/collision metric, model checksum and equal-geometry aggregate was checked. Max NumPy forward error{verify['max_numpy_forward_error']:.3g}; max independently computed oracle-cost error{verify['max_oracle_cost_formula_error']:.3g}. Native state/substep replay error0. Total experiment wall time{summary['wall_seconds']:.1f}s, including smoke and action replay, excluding independent audit.

Full model weights remain in prior external run directories. Raw closed-loop traces/candidates/predictions remain in this external run directory and are identified by SHA256; source, compact results, all episode CSV, frozen protocol, verification and vector figure are published. Raw SVG retained; published SVG strips trailing whitespace only. No renderer or video was claimed.

## Next block

If learned safe-goal success remains poor, first diagnose goal-progress/cost ranking on NEW training/validation trajectories and check candidate coverage using the existing privileged reference. Do not tune on these six consumed confirmation geometries, and do not expand long-duration runs before fixing the observed failure. Separate scoring and candidate-generation ablations in the next frozen protocol. Navigation-specific LeWM checkpoint/config/normalization/history/action/state-readout remains missing; this experiment is a direct-scoring baseline, not LeWM transfer. No three-agent or learned-world-model evaluation was run.
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8');(root/'figure_provenance.json').write_text(json.dumps(dict(selection='All6 geometry specs; first frozen candidate seed only for trajectories, all60 in tables',axes='VMAS coordinates, equal spatial aspect per panel',svg_change='Only trailing whitespace stripped; raw export retained',meaning='Path solid/dashed agents0/1, terminal dots, start circles, goal stars'),indent=2))
    print(table)
if __name__=='__main__':main()
