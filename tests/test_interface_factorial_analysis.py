import copy
import json

import pytest
import torch

from vmas_pusht import interface_factorial_analysis as audit


def synthetic_group(arm="wm_full_direct"):
    n, horizon = 3, 22
    initial = torch.zeros(n, 24)
    initial[:, 0] = torch.tensor([.1, .2, .3])
    states = initial[:, None].expand(n, horizon+1, 24).clone()
    actions = torch.zeros(n, horizon, 2, 2)
    rows = []
    for i in range(n):
        rows.append(dict(planner=arm, model_seed=0, support="passive", initial_angle_deg=0.,
            nuisance_id=i, first_goal_rad=.04, relative_goal_rad=.08, physics_profile="dry_light", substeps=160,
            phase_end_steps=dict(rotate_first=5, retreat=7, recovery=12, rotate_final=22),
            first_goal_success_at_phase_boundary=i > 0, final_goal_dwell_success=i != 1,
            terminal_success=True, finite=True, wall=False, ordered_goal_success=i == 2,
            interface_completion_success=i == 2, total_steps=32, option_queries=1))
    def snapshot(stage, step):
        snapshot_rows = [dict(row, snapshot_stage=stage, composition_row=i) for i, row in enumerate(rows)]
        return dict(rows=snapshot_rows, anchor=torch.zeros(n, 2), history=torch.zeros(n, 2, 17),
            snapshot=dict(batch_size=n, physics=("fixed", 160),
                entities=[dict(name="agent0", state=dict(pos=states[:, step, :2].clone()),
                               action=dict(u=torch.zeros(n, 2)))],
                scenario=dict(reward=torch.zeros(n)), steps=torch.full((n,), 10+step)))
    return dict(rows=rows, trajectory=states, actions=actions, initial_pose=torch.zeros(n, 3),
        prelude_trajectory=initial[:, None].expand(n, 11, 24).clone(),
        prelude_actions=torch.zeros(n, 10, 2, 2),
        counterfactual_snapshots=[snapshot("post_prelude", 0), snapshot("post_recovery", 12)])


def reorder(group, order):
    result = copy.deepcopy(group)
    index = torch.tensor(order)
    for field in ("trajectory", "actions", "initial_pose", "prelude_trajectory", "prelude_actions"):
        result[field] = result[field][index]
    result["rows"] = [result["rows"][i] for i in order]
    for record in result["counterfactual_snapshots"]:
        record["rows"] = [record["rows"][i] for i in order]
        for i, row in enumerate(record["rows"]): row["composition_row"] = i
        record["anchor"] = record["anchor"][index]
        record["history"] = record["history"][index]
        snap = record["snapshot"]
        for entity in snap["entities"]:
            for category in ("state", "action"):
                entity[category] = {key: value[index] for key, value in entity[category].items()}
        snap["scenario"] = {key: value[index] for key, value in snap["scenario"].items()}
        snap["steps"] = snap["steps"][index]
    return result


def test_pairing_uses_identity_and_keeps_failed_prefixes_after_row_reordering():
    direct = synthetic_group()
    expanded = synthetic_group("wm_full")
    expanded["rows"][1].update(final_goal_dwell_success=True, ordered_goal_success=True)
    expanded = reorder(expanded, [2, 0, 1])
    paired = audit.compare_groups(direct, expanded)
    result = audit.summarize_pairs(paired)
    assert result["cases"] == 3 and result["shared_first_goal_success"] == 2
    assert result["direct_success"] == 1 and result["compared_success"] == 2
    assert result["improved"] == 1 and result["worsened"] == 0
    assert result["both_failure"] == 1 and result["prefix_matches"]
    assert paired[0]["nuisance_id"] == 0 and not paired[0]["compared_success"]
    assert paired[0]["compared_row"] == 1


def test_duplicate_or_missing_physical_cases_cannot_change_the_denominator():
    direct = synthetic_group()
    with pytest.raises(ValueError, match="Duplicate paired"):
        audit.row_index(direct["rows"] + [copy.deepcopy(direct["rows"][0])])
    expanded = synthetic_group("wm_full")
    expanded["rows"].pop()
    with pytest.raises(ValueError, match="exactly the same"):
        audit.compare_groups(direct, expanded)


def test_later_success_cannot_repair_failed_first_goal_in_stored_labels():
    group = synthetic_group()
    group["rows"][0]["ordered_goal_success"] = True
    with pytest.raises(ValueError, match="retain failed first"):
        audit.row_index(group["rows"])


def test_prefix_audit_uses_all_24_coordinates_and_original_discrete_clocks():
    direct, expanded = synthetic_group(), synthetic_group("wm_full")
    expanded["trajectory"][0, 8, 23] += 2e-6
    expanded["rows"][1]["phase_end_steps"]["retreat"] += 1
    paired = audit.compare_groups(direct, expanded)
    assert not paired[0]["prefix_matches"] and paired[0]["max_state_difference"] > 1e-6
    assert paired[1]["state_bitwise"] and not paired[1]["discrete_prefix_clocks_match"]
    assert paired[2]["prefix_matches"]


def test_complete_snapshot_action_buffers_are_checked_separately_from_pose_trace():
    direct, expanded = synthetic_group(), synthetic_group("wm_full")
    expanded["counterfactual_snapshots"][0]["snapshot"]["entities"][0]["action"]["u"][1, 0] = .2
    paired = audit.compare_groups(direct, expanded)
    assert paired[1]["state_bitwise"]
    assert not paired[1]["initial_complete_snapshot"]["matches"]
    assert not paired[1]["prefix_matches"]


def test_history_replication_checks_final_trajectory_but_pairing_stops_at_handoff():
    direct, expanded = synthetic_group(), synthetic_group("wm_full")
    expanded["trajectory"][0, 17, 16] = .2
    assert audit.summarize_pairs(audit.compare_groups(direct, expanded))["prefix_matches"]
    assert not audit.summarize_pairs(audit.compare_groups(direct, expanded, full_task=True))["prefix_matches"]


def budget_group():
    group = synthetic_group("wm_full_budget39_s0")
    n = len(group["rows"])
    for row in group["rows"]: row["candidate_policy"] = "budget39_s0"
    table = audit.expected_allocation(0)[None].expand(n, -1, -1).clone()
    group["option_records"] = [dict(row=i, start_step=12, start_contact_mode=3) for i in range(n)]
    group["queries"] = [dict(step=12, query_mask=torch.ones(n, dtype=torch.bool),
        phase=torch.full((n,), 3), allow_routes=torch.ones(n, dtype=torch.bool),
        diagnostics=dict(candidate_table=table, candidate_budget=torch.full((n,), 39),
            evaluated_candidates=torch.full((n,), 39), candidate_count=torch.full((n,), 39),
            simulated_candidate_steps=torch.zeros(n, dtype=torch.long), queried_mask=torch.ones(n, dtype=torch.bool),
            active_simulated_candidate_steps=torch.zeros(n, dtype=torch.long),
            padded_simulator_steps=torch.zeros(n, dtype=torch.long),
            padded_simulator_substeps=torch.zeros(n, dtype=torch.long), oracle_calls=0,
            query_wall_seconds=.25, budget_fallback=torch.zeros(n, dtype=torch.bool),
            source=["world_model"]*n))]
    return group


def test_budget_audit_preserves_marginals_and_counts_batched_wall_time_once():
    group = budget_group()
    result = audit.audit_queries(group)
    assert not result["violations"]
    assert result["batched_selector_wall_seconds"] == .25
    assert len(result["decisions"]) == 3
    assert sum(r["evaluated_candidates"] for r in result["per_episode"]) == 117
    for shift in range(3):
        table = audit.expected_allocation(shift)
        assert [(table[:, 0] == route).sum().item() for route in range(3)] == [15, 12, 12]
        assert audit._multiset(table[:, 1:]) == audit._multiset(audit.canonical_direct_table()[:, 1:])


def test_budget_audit_detects_marginal_change_duplicates_and_undeclared_routes():
    group = budget_group()
    table = group["queries"][0]["diagnostics"]["candidate_table"]
    table[0, 1, 2] = .7
    table[1, 2] = table[1, 1]
    group["option_records"][2]["start_contact_mode"] = 0
    violations = audit.audit_queries(group)["violations"]
    assert len(violations) == 3
    assert any("marginal" in text for text in violations[0]["reasons"])
    assert any("duplicate" in text for text in violations[1]["reasons"])
    assert any("another face" in text for text in violations[2]["reasons"])


def test_equal_budget_is_a_cap_for_actual_oracle_queries_not_equal_simulation_steps():
    group = budget_group()
    for row in group["rows"]: row["planner"] = "true_option_budget39_s0"
    diag = group["queries"][0]["diagnostics"]
    diag["source"] = ["true_simulator"]*3
    diag["evaluated_candidates"] = torch.tensor([39, 26, 13])
    diag["candidate_count"] = torch.tensor([39, 26, 13])
    diag["simulated_candidate_steps"] = torch.tensor([1100, 600, 65])
    diag["active_simulated_candidate_steps"] = diag["simulated_candidate_steps"].clone()
    diag["padded_simulator_steps"] = torch.tensor([1560, 1560, 1560])
    diag["padded_simulator_substeps"] = diag["padded_simulator_steps"]*160
    diag["oracle_calls"] = 1
    result = audit.audit_queries(group)
    assert not result["violations"]
    assert {r["candidate_budget"] for r in result["decisions"]} == {39}
    assert sum(r["simulated_candidate_steps"] for r in result["per_episode"]) == 1765
    assert sum(r["padded_simulator_steps"] for r in result["per_episode"]) == 4680
    assert result["oracle_calls"] == 1
    diag["evaluated_candidates"][0] = 40
    assert audit.audit_queries(group)["violations"]


def test_manifest_verifies_artifacts_before_pairing(tmp_path):
    raw, report, manifest = [tmp_path/name for name in ("group.pt", "group.json", "manifest.json")]
    group = synthetic_group()
    torch.save(dict(groups=[group]), raw)
    report.write_text("{}\n")
    source = dict(groups=[dict(planner="wm_full_direct", model_seed=0, goal="standard",
        raw_path=str(raw), raw_sha256=audit.sha(raw), report_path=str(report), report_sha256=audit.sha(report))])
    manifest.write_text(json.dumps(source))
    _, index, _ = audit.load_manifest(manifest)
    assert ("wm_full_direct", 0, "standard") in index
    report.write_text('{"changed":true}\n')
    with pytest.raises(AssertionError, match="digest mismatch"):
        audit.load_manifest(manifest)
