from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip("mujoco")

from uav_transport import mission_analysis as audit
from uav_transport.control import TransportController
from uav_transport.local_model import FEATURE_NAMES
from uav_transport.mission import MissionConfig, select_option
from uav_transport.plant import TransportConfig, TransportEnv
from uav_transport.snapshot import capture


def operation(completion=True, excursion=.02, tension=.3, slack=.5):
    return dict(summary=dict(completion=completion, peak_payload_excursion=excursion,
                             peak_donor_tension=tension, donor_slack_seconds=slack))


def test_delivery_preserves_all_failed_prefixes_and_separates_constraint_score():
    stages = [dict(name=f"leg_{i}", success=True) for i in (1, 2, 3)]
    operations = [operation(), operation()]
    args = dict(finite=True, collision=False, warnings=0)
    full = audit.rescore_delivery(stages, operations, MissionConfig(), **args)
    assert full["delivery_completion"] and full["constrained_completion"]
    operations[0]["summary"]["peak_payload_excursion"] = .2
    result = audit.rescore_delivery(stages, operations, MissionConfig(), **args)
    assert result["delivery_completion"] and not result["constrained_completion"]
    stages[0]["success"] = False
    result = audit.rescore_delivery(stages, operations, MissionConfig(), **args)
    assert not result["delivery_completion"] and result["waypoint_successes"] == [False, True, True]


@pytest.mark.parametrize("change", ["missing_leg", "missing_operation", "slack", "warning", "collision"])
def test_missing_or_unsafe_execution_cannot_be_counted_as_delivery(change):
    stages = [dict(name=f"leg_{i}", success=True) for i in (1, 2, 3)]
    ops = [operation(), operation()]; args = dict(finite=True, collision=False, warnings=0)
    if change == "missing_leg": stages.pop(0)
    if change == "missing_operation": ops.pop()
    if change == "slack": ops[0]["summary"]["donor_slack_seconds"] = .05
    if change == "warning": args["warnings"] = 1
    if change == "collision": args["collision"] = True
    assert not audit.rescore_delivery(stages, ops, MissionConfig(), **args)["delivery_completion"]


def prefix_fixture(planner="wm_small_direct"):
    env = TransportEnv(); snapshot = capture(env)
    state = np.zeros((10, 64)); integration = np.repeat(snapshot.state[None], 10, axis=0)
    return dict(summary=dict(planner=planner, model_seed=0, case_id="a", delivery_completion=True,
        constrained_completion=planner == "wm_small"), case=dict(case_id="a", donor=0),
        initial_snapshot=deepcopy(snapshot), initial_controller_memory=dict(integral=np.zeros(3)),
        stages=[dict(name="leg_1", begin_step=1, end_step=4, success=True,
                     goal_dwell_steps=2, required_dwell_steps=2)],
        decisions=[dict(mission_step=4, start_snapshot=deepcopy(snapshot), controller_memory=dict(integral=np.ones(3)))],
        trace=dict(physical_states=state, integration_states=integration,
                   actions=np.zeros((9, 3, 4)), external_forces=np.zeros((9, 4, 3))))


def test_pairing_stops_before_treatment_and_counts_delivery_separately():
    left, right = prefix_fixture(), prefix_fixture("wm_small")
    right["trace"]["physical_states"][5:, 0] = 9.
    result = audit.compare_prefix(left, right)
    assert result["prefix_matches"]
    assert result["scores"]["constrained_completion"]["improved"]
    assert not result["scores"]["delivery_completion"]["improved"]
    assert result["prefix_steps"] == 4


def test_pairing_checks_motor_state_complete_buffers_and_first_leg_clock():
    left, right = prefix_fixture(), prefix_fixture("wm_small")
    right["trace"]["physical_states"][3, 63] = .1
    result = audit.compare_prefix(left, right)
    assert not result["prefix_matches"] and result["traces"]["physical_states"]["max_absolute_difference"] == .1
    right = prefix_fixture("wm_small"); right["initial_snapshot"].last_external_forces[0, 0] = 1.
    assert not audit.compare_prefix(left, right)["initial_snapshot_equal"]
    right = prefix_fixture("wm_small"); right["stages"][0]["success"] = False
    assert not audit.compare_prefix(left, right)["first_leg_clock_and_score_equal"]
    right = prefix_fixture("wm_small"); right["summary"]["case_id"] = "different"
    with pytest.raises(ValueError, match="same case"):
        audit.compare_prefix(left, right)


class Predictor:
    def __init__(self): self.calls = []
    def predict(self, x):
        self.calls.append(x.copy())
        out = np.repeat(np.array([[.02, .3, 8., .1, 1., .005, .01, 1.]]), len(x), axis=0)
        out[:, 2] = 5.+x[:, FEATURE_NAMES.index("operation_rearrange_seconds")]+x[:, FEATURE_NAMES.index("operation_retension_seconds")]
        return out


def decision_fixture():
    env = TransportEnv(); controller = TransportController(env.config); model = Predictor()
    mode, cfg, record = select_option(env, controller, 0, .07, "wm_small", model, margins=(.01, .1))
    record["mission_step"] = 0; record["operation_index"] = 0
    op = dict(mission_begin_step=0, reconfigure=mode, recovery_config=asdict(cfg), label=record["selected_prediction"].copy())
    run = dict(summary=dict(case_id="a", planner="wm_small", model_seed=0),
        plant_config=asdict(env.config), mission_config=asdict(MissionConfig()),
        decisions=[record], operations=[op])
    manifest = dict(models={"small/seed0": dict(margins=[.01, .1])})
    return run, {"small/seed0": model}, manifest


def test_learned_decisions_rebuild_frozen_predictions_argmin_and_zero_simulator_counts():
    run, models, manifest = decision_fixture()
    checks = audit.audit_decisions(run, models, manifest)
    assert checks[0]["verified"] and checks[0]["predictions_bitwise"]
    assert checks[0]["selected_argmin_matches"] and checks[0]["simulated_candidate_control_steps"] == 0
    np.testing.assert_array_equal(models["small/seed0"].calls[0], models["small/seed0"].calls[1])
    changed = deepcopy(run); changed["decisions"][0]["simulated_candidate_control_steps"] = 1
    assert not audit.audit_decisions(changed, models, manifest)[0]["verified"]


def test_decision_audit_detects_altered_prediction_margin_or_executed_option():
    run, models, manifest = decision_fixture()
    changed = deepcopy(run); changed["decisions"][0]["predictions"][0, 2] += .01
    assert not audit.audit_decisions(changed, models, manifest)[0]["predictions_bitwise"]
    changed = deepcopy(run); changed["decisions"][0]["validation_margins"][0] += .01
    assert not audit.audit_decisions(changed, models, manifest)[0]["verified"]
    changed = deepcopy(run); changed["operations"][0]["reconfigure"] = not changed["operations"][0]["reconfigure"]
    assert not audit.audit_decisions(changed, models, manifest)[0]["verified"]


def test_exact_selected_outcome_is_compared_to_real_execution_with_strict_tolerance():
    run, models, manifest = decision_fixture()
    run["summary"]["planner"] = "true_option"
    record = run["decisions"][0]
    record["prediction_source"] = "true_option_simulator"
    record["validation_margins"][:] = 0.
    from uav_transport.mission import outcome_scores
    scores, feasible, fallback = outcome_scores(record["predictions"])
    record.update(ranking_scores=scores, predicted_feasible=feasible, feasible_count=int(feasible.sum()),
                  minimum_risk_fallback=fallback, feasibility_fallback=fallback)
    check = audit.audit_decisions(run, models, manifest)[0]
    assert check["verified"] and check["oracle_actual_max_absolute_difference"] == 0.
    run["operations"][0]["label"][1] += 1e-7
    check = audit.audit_decisions(run, models, manifest)[0]
    assert not check["verified"] and check["oracle_actual_max_absolute_difference"] > 1e-9


def test_latency_distribution_records_every_decision_without_filtering_slow_cases():
    result = audit.distribution([.001, .002, .003, .5])
    assert result["count"] == 4 and result["maximum"] == .5
    assert result["median"] == pytest.approx(.0025)
    assert result["p95"] > .4
    assert audit.distribution([])["count"] == 0


def test_whole_mission_cable_diagnostic_keeps_registered_donor_score_separate():
    from uav_transport.mission import run_mission, mission_cases
    cfg = MissionConfig(settle_seconds=.04, move_seconds=.04, leg_budget_seconds=.1,
                        goal_dwell_seconds=.02, option_budget_seconds=.1)
    run = run_mission(mission_cases()[0], "rule_reconfigure", config=cfg)
    before = audit.verify_execution(run)
    assert before["scores_reproduced"] and not before["rotor_bound_violation"]
    run["trace"]["substeps"]["cable_tensions"][0, 2] = .9
    result = audit.verify_execution(run)
    assert result["maximum_all_cable_tension"] == .9
    assert result["any_cable_above_registered_donor_limit_seconds"] == pytest.approx(.002)
    assert result["maximum_operation_donor_tension"] == before["maximum_operation_donor_tension"]
    assert result["scores_reproduced"]
