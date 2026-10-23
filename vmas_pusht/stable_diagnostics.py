"""Post-registration diagnostics; no weights, primary controllers or goals change."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import time

import torch

from vmas_pusht.abstraction_composition import checkpoint_digest
from vmas_pusht.abstraction_models import predict_checkpoint
from vmas_pusht.abstraction_transfer import ANGLE_TOLERANCE, TRANSLATION_TOLERANCE, load_checkpoints, _support_commands
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.stable_composition import candidates_for_bank, merge_snapshot_records, snapshot_from_record, _plain
from vmas_pusht.stable_local_wm import predict_entry
from vmas_pusht.snapshot import restore
from vmas_pusht.scenario import wrap

ROOT = Path('data/novelty')
PREFIXES = ['original_matched', 'selected_matched', 'original_cross', 'selected_cross']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, subset in sorted(groups.items()):
        entry = dict(zip(keys, key)); entry['rows'] = len(subset)
        for field, kind in fields.items():
            values = [row[field] for row in subset]
            entry[field + '_' + kind] = (sum(values) if kind == 'count' else
                (sum(values) / len(values) if kind == 'mean' else max(values)))
        output.append(entry)
    return output


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    artifacts = {}
    files = [ROOT / ('stable_composition_' + name + '.pt') for name in PREFIXES]
    model_paths = {'original': ROOT / 'stable_local_original_models.pt',
                   'dry_light': ROOT / 'stable_local_selected_models.pt'}
    sources = [Path(__file__), Path('vmas_pusht/stable_composition.py'),
        Path('vmas_pusht/stable_local_wm.py'), Path('vmas_pusht/physics_profiles.py'),
        Path('vmas_pusht/rebrace.py'), Path('vmas_pusht/snapshot.py')]
    provenance = {str(path): sha(path) for path in files + list(model_paths.values()) + sources}
    for name, path in zip(PREFIXES, files):
        artifacts[name] = torch.load(path, map_location='cpu')
    model_packs = {name: torch.load(path, map_location='cpu')['checkpoints'] for name, path in model_paths.items()}
    legacy, legacy_provenance, _ = load_checkpoints(ROOT / 'abstraction_models.pt', ROOT / 'abstraction_models_preload.pt')
    for path, metadata in legacy_provenance.items(): provenance[path] = metadata['sha256']
    before_models = checkpoint_digest([model_packs, legacy])

    handoffs = []
    for name, artifact in artifacts.items():
        for group in artifact['groups']:
            for i, row in enumerate(group['rows']):
                initial = group['initial_pose'][i]
                ends = row['phase_end_steps']
                boundary = {}
                for phase in ('rotate_first', 'retreat', 'recovery', 'rotate_final'):
                    physical = group['trajectory'][i, ends[phase]]
                    boundary[phase] = dict(angle_rad=float(wrap(physical[16] - initial[2])),
                        translation_units=float((physical[12:14] - initial[:2]).norm()))
                theta = boundary['recovery']['angle_rad']
                target = row['relative_goal_rad']
                handoffs.append(dict(source=name, physics_profile=row['physics_profile'],
                    model_training_profile=row['model_training_profile'], substeps=row['substeps'],
                    support=row['support'], planner=row['planner'], goals=row['goal_pair'],
                    first_goal_rad=row['first_goal_rad'], relative_goal_rad=target,
                    initial_angle_deg=row['initial_angle_deg'], nuisance_id=row['nuisance_id'],
                    first_gate=row['first_goal_success_at_phase_boundary'], recovery_gate=row['contact_reacquired'],
                    ordered_goal_success=row['ordered_goal_success'],
                    handoff_angle_rad=theta, handoff_error_rad=theta-target,
                    overshoot=theta > target + ANGLE_TOLERANCE,
                    below=theta < target - ANGLE_TOLERANCE,
                    within=abs(theta-target) <= ANGLE_TOLERANCE,
                    translation_valid=boundary['recovery']['translation_units'] < TRANSLATION_TOLERANCE,
                    retreat_dtheta_rad=boundary['retreat']['angle_rad'] - boundary['rotate_first']['angle_rad'],
                    recovery_dtheta_rad=theta - boundary['retreat']['angle_rad'], phase_boundaries=boundary))

    common_errors = defaultdict(list)
    common_response = defaultdict(list)
    # The union uses all primary model policies' sampled states with equal row
    # weight. Predictions of every model are compared on this SAME state/label set.
    for name in ('original_matched', 'selected_matched'):
        artifact = artifacts[name]
        for group, counterfactual in zip(artifact['groups'], artifact['counterfactuals']):
            for block in counterfactual['blocks']:
                rows = block['rows']
                c, f, h, y = [block[key] for key in ('coarse', 'full', 'history', 'response')]
                for support in ('passive', 'world_anchor', 'preloaded_anchor'):
                    for stage in ('post_prelude', 'post_recovery'):
                        idx = torch.tensor([i for i, row in enumerate(rows)
                            if row['support'] == support and row['snapshot_stage'] == stage], dtype=torch.long)
                        if not len(idx): continue
                        actual_profile = group['physics_profile']['name']; ss = rows[int(idx[0])]['substeps']
                        for training_profile, pack in model_packs.items():
                            for entry_key, entry in pack.items():
                                mode, model, seed_name = entry_key.split('/')
                                if mode != support: continue
                                prediction = predict_entry(entry, c[idx], full=f[idx], history=h[idx])
                                key = (actual_profile, ss, training_profile, model, seed_name, support, stage)
                                common_errors[key].append(prediction - y[idx]); common_response[key].append(y[idx])
                        coarse = c[idx].clone(); coarse[:,15] = 20
                        prediction = predict_checkpoint(legacy[support], coarse)
                        key = (actual_profile, ss, 'historical_original', 'legacy_wm', 'seed0', support, stage)
                        common_errors[key].append(prediction - y[idx]); common_response[key].append(y[idx])
    common_summary = []
    keys = ('actual_physics', 'substeps', 'training_physics', 'model', 'seed', 'support', 'snapshot_stage')
    for key in sorted(common_errors):
        e = torch.cat(common_errors[key]); y = torch.cat(common_response[key])
        common_summary.append(dict(zip(keys,key), rows=len(e),
            rmse_xy_units_rad=e.square().mean(0).sqrt().tolist(),
            translation_rmse_units=float(e[:,:2].square().sum(1).mean().sqrt()),
            angular_rmse_rad=float(e[:,2].square().mean().sqrt()),
            zero_response_rmse_xy_units_rad=y.square().mean(0).sqrt().tolist(),
            joint_within_one_third_task_tolerance_count=int(((e[:,:2].norm(dim=1)<TRANSLATION_TOLERANCE/3)
                & (e[:,2].abs()<ANGLE_TOLERANCE/3)).sum())))

    # True one-macro oracle diagnostic. This never supplies actions to or reruns
    # the primary controller. It is a bounded local action-set comparison only.
    oracle_records = []
    selected_groups = [group for group in artifacts['selected_matched']['groups'] if group['rows'][0]['substeps']==160]
    for group in selected_groups:
        for record in group['counterfactual_snapshots']:
            if record['rows'][0]['snapshot_stage'] != 'post_recovery': continue
            copied = dict(record, rows=[])
            for row in record['rows']:
                i=row['composition_row']; extra=dict(row)
                extra['initial_pose'] = group['initial_pose'][i].tolist()
                extra['primary_first_gate'] = group['rows'][i]['first_goal_success_at_phase_boundary']
                step=group['rows'][i]['phase_end_steps']['recovery']
                chosen=-1
                for selection in group['selected_macros']:
                    if selection['step']==step and selection['rows'][i]:
                        chosen=int(selection['chosen_index'][i]); break
                extra['primary_model_chosen_index'] = chosen
                copied['rows'].append(extra)
            oracle_records.append(copied)
    merged = merge_snapshot_records(oracle_records)
    original_rows = merged['rows']; n=len(original_rows); k=13
    index=torch.arange(n).repeat_interleave(k)
    env=make_profile(selected_groups[0]['physics_profile'],num_envs=n*k,substeps=160,seed=101)
    restore(env,snapshot_from_record(merged),index)
    rows=[row for row in original_rows for _ in range(k)]
    anchor=merged['anchor'][index]
    current=env.scenario.t_pose().reshape(n,k,3)[:,0].clone()
    candidates=candidates_for_bank(current[:,2], 'bank13')
    actions=candidates.reshape(-1,2)
    for _ in range(5):env.step([actions,_support_commands(env,anchor,rows)])
    endpoints=env.scenario.t_pose().reshape(n,k,3)
    initial=torch.tensor([row['initial_pose'] for row in original_rows])
    targets=initial[:,2]+torch.tensor([row['relative_goal_rad'] for row in original_rows])
    def costs(pose):
        err=wrap(pose[:,:,2]-targets[:,None]); shift=(pose[:,:,:2]-initial[:,None,:2]).norm(dim=-1)
        return (err/ANGLE_TOLERANCE).square()+(shift/TRANSLATION_TOLERANCE).square(),err,shift
    final_cost,error,shift=costs(endpoints); start_cost,start_error,start_shift=costs(current[:,None])
    start_cost=start_cost[:,0]; start_error=start_error[:,0]; start_shift=start_shift[:,0]
    oracle=[]
    for i,row in enumerate(original_rows):
        best=int(final_cost[i].argmin()); chosen=row['primary_model_chosen_index']
        oracle.append(dict(physics_profile='dry_light',substeps=160,support=row['support'],planner=row['planner'],
            first_goal_rad=row['first_goal_rad'],relative_goal_rad=row['relative_goal_rad'],
            initial_angle_deg=row['initial_angle_deg'],nuisance_id=row['nuisance_id'],
            first_gate=row['primary_first_gate'], handoff_angle_error_rad=float(start_error[i]),
            handoff_translation_units=float(start_shift[i]), overshoot=bool(start_error[i]>ANGLE_TOLERANCE),
            initial_cost=float(start_cost[i]),zero_action_cost=float(final_cost[i,0]),
            best_cost=float(final_cost[i,best]),best_index=best,
            best_cost_improvement=float(start_cost[i]-final_cost[i,best]),
            any_cost_improvement=bool(final_cost[i].min()<start_cost[i]-1e-6),
            best_angle_improvement_rad=float(start_error[i].abs()-error[i].abs().min()),
            any_action_reaches_final_pose=bool(((error[i].abs()<ANGLE_TOLERANCE)&(shift[i]<TRANSLATION_TOLERANCE)).any()),
            model_chosen_index=chosen,model_chosen_cost=float(final_cost[i,chosen]) if chosen>=0 else None,
            all_costs=final_cost[i].tolist(),all_angle_errors_rad=error[i].tolist()))
    if checkpoint_digest([model_packs,legacy]) != before_models: raise AssertionError('Frozen model mutated')
    for path,digest in provenance.items():
        if sha(path)!=digest:raise AssertionError('Source artifact changed: '+path)
    report=dict(schema_version=1,source_sha256=provenance,elapsed_seconds=time.perf_counter()-started,
        handoff_rows=handoffs,handoff_summary=summarize(handoffs,
            ('source','physics_profile','model_training_profile','substeps','support','planner','first_goal_rad','relative_goal_rad'),
            {'first_gate':'count','recovery_gate':'count','overshoot':'count','below':'count','within':'count',
             'handoff_error_rad':'mean','retreat_dtheta_rad':'mean','recovery_dtheta_rad':'mean'}),
        same_state_counterfactual_summary=common_summary,
        one_macro_oracle_rows=oracle,one_macro_oracle_summary=summarize(oracle,
            ('support','planner','first_goal_rad','relative_goal_rad'),
            {'first_gate':'count','overshoot':'count','any_cost_improvement':'count','any_action_reaches_final_pose':'count',
             'handoff_angle_error_rad':'mean','best_cost_improvement':'mean','best_angle_improvement_rad':'mean'}),
        limitations=[
            'Post-hoc diagnosis only; no primary physics, model, controller, goal or success criterion was changed.',
            'Union scoring holds physical test states and labels equal for all models, but support modes have different state distributions; cross-support model-error differences are not isolated physical causal effects.',
            'Every frozen model seed is evaluated; composition primary results remain preregistered seed0. Historical600-update reset-only checkpoint is not an equal-budget learning control.',
            'Oracle considers thirteen current five-step constant commands, not all future policies. Absence of a one-step improvement does not prove global unreachability.',
            'Local training contains constant-push prefixes after zero-primary preparation. It does not cover stop-after-strong-push transitions, goal-conditioned action switches, retreat/rebrace or changed anchors. Prediction failure cannot uniquely identify missing-information effects.',
            'Expanded oracle replay batches preserve stored states but can differ numerically from original batch trajectories. Oracle values are simulator labels, never task-planner inputs.',
        ])
    out=ROOT/'stable_composition_diagnostics'
    torch.save(dict(one_macro_oracle=dict(rows=original_rows,candidates=candidates,initial_pose=initial,
        handoff_pose=current,endpoint=endpoints,cost=final_cost,initial_cost=start_cost),
        model_weights_sha256=before_models,source_sha256=provenance),out.with_suffix('.pt'))
    report['raw_sha256']=sha(out.with_suffix('.pt'))
    out.with_suffix('.json').write_text(json.dumps(_plain(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),handoff_cases=len(handoffs),same_state_model_groups=len(common_summary),
        oracle_cases=len(oracle),oracle_branches=len(oracle)*13,elapsed_seconds=report['elapsed_seconds'])))

if __name__=='__main__':main()
