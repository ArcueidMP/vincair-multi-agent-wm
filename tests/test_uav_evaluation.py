from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("mujoco")

from uav_transport.control import RecoveryConfig, RecoveryProtocol, TransportController
from uav_transport.evaluation import TARGET_NAMES, fork, replay_option, run_option
from uav_transport.plant import TransportConfig, TransportEnv
from uav_transport.snapshot import integration_state


def prepared(config=None):
    env=TransportEnv(config);controller=TransportController(env.config)
    goal=env.observe()["payload_pos"].copy()
    for _ in range(20):env.step(controller.command(env.observe(),goal))
    return env,controller


@pytest.fixture(scope="module")
def complete_option():
    env,controller=prepared()
    return run_option(env,controller,RecoveryConfig(),0,True)


def test_fork_preserves_physical_and_controller_state_without_shared_memory():
    env,controller=prepared();other,feedback=fork(env,controller)
    np.testing.assert_array_equal(integration_state(env),integration_state(other))
    np.testing.assert_array_equal(controller.payload_integral,feedback.payload_integral)
    assert controller.calls==feedback.calls
    feedback.payload_integral[0]+=1
    assert controller.payload_integral[0]!=feedback.payload_integral[0]
    other.data.act[0]*=.5
    assert other.data.act[0]!=env.data.act[0]


def test_observer_leaves_physical_commands_controller_and_resources_bitwise_unchanged():
    env,controller=prepared();reference,feedback=fork(env,controller)
    goal=env.observe()["payload_pos"].copy();protocol=RecoveryProtocol(reference.config)
    result=run_option(env,controller,RecoveryConfig(),0,True,max_seconds=.2)
    for _ in range(10):
        r=protocol.reference()
        command=feedback.command(reference.observe(),goal,formation_offsets=r.offsets,
            formation_velocity=r.offset_velocity,support_weights=r.support_weights)
        observation=reference.step(command)
        protocol.update(observation,goal)
    np.testing.assert_array_equal(integration_state(env),integration_state(reference))
    np.testing.assert_array_equal(controller.payload_integral,feedback.payload_integral)
    for key in env.metrics:np.testing.assert_array_equal(env.metrics[key],reference.metrics[key])
    assert "cable_tensions" not in env.__dict__
    assert result["summary"]["physics_steps"]==100


def test_verified_option_really_slackens_and_retensions_and_has_aligned_labels(complete_option):
    result=complete_option;s=result["summary"]
    assert result["label_names"]==TARGET_NAMES and result["label"].shape==(8,)
    np.testing.assert_array_equal(result["label"],[float(s[k]) for k in TARGET_NAMES])
    assert s["completion"] and s["protocol_done"] and s["verification_dwell_passed"]
    assert s["donor_slack_seconds"]>1 and s["donor_slack_observed"]
    assert s["peak_donor_tension"]>0 and s["retension_tension_impulse"]>0
    assert s["absolute_actuator_work"]>0 and s["total_squared_thrust_integral"]>0
    assert s["finite"] and not s["collision"] and not s["warning_count"]
    assert s["minimum_uav_separation"]>.1
    assert np.all(result["trace"]["observations"][-1]["cable_tensions"]>.02)


def test_tension_impulse_and_actual_slack_use_only_the_declared_physical_samples(complete_option):
    result=complete_option;raw=result["trace"]["substeps"];dt=result["plant_config"]["dt"]
    phases=np.asarray(raw["phase"])
    mask=(phases=="retension")|(phases=="confirm")
    expected=raw["cable_tensions"][mask,0].sum()*dt
    assert result["summary"]["retension_tension_impulse"]==expected
    slack=(raw["cable_tensions"][:,0]<.005)&(raw["cable_gaps"][:,0]>.01)
    assert result["summary"]["donor_slack_seconds"]==slack.sum()*dt
    assert len(raw["time"])==result["summary"]["physics_steps"]
    assert len(result["trace"]["physical_states"])==len(result["trace"]["actions"])+1


def test_deadline_cannot_be_relabelled_completed_just_for_holding_the_payload():
    env,controller=prepared()
    result=run_option(env,controller,RecoveryConfig(),0,False,max_seconds=.31,record=False)
    assert result["summary"]["elapsed_seconds"]==.3
    assert not result["summary"]["completion"] and not result["summary"]["protocol_done"]
    assert result["summary"]["stop_reason"]=="physical_time_budget"
    assert result["trace"] is None
    assert result["end_snapshot"].control_steps-result["start_snapshot"].control_steps==15
    with pytest.raises(ValueError,match="recorded"):
        replay_option(result)


def test_fixed_action_replay_is_bitwise_and_refinement_keeps_original_physical_duration(complete_option):
    result=complete_option
    nominal=replay_option(result)
    assert nominal["source_replay_bitwise"]
    np.testing.assert_array_equal(nominal["integration_states"],result["trace"]["integration_states"])
    for name in TARGET_NAMES:
        assert nominal["summary"][name]==result["summary"][name]
    fine=replay_option(result,dt=.001)
    assert fine["summary"]["elapsed_seconds"]==result["summary"]["elapsed_seconds"]
    assert fine["summary"]["physics_steps"]==2*result["summary"]["physics_steps"]
    assert fine["summary"]["finite"] and not fine["summary"]["collision"]


def test_original_snapshots_and_recorded_action_arrays_are_immutable(complete_option):
    result=complete_option
    before=result["start_snapshot"].state.copy();actions=result["trace"]["actions"].copy()
    replay_option(result)
    np.testing.assert_array_equal(result["start_snapshot"].state,before)
    np.testing.assert_array_equal(result["trace"]["actions"],actions)


def test_invalid_local_reference_cannot_advance_either_state():
    env,controller=prepared();state=integration_state(env);calls=controller.calls
    with pytest.raises(ValueError,match="reference"):
        run_option(env,controller,RecoveryConfig(),0,True,goal=[np.nan,0.,1.])
    np.testing.assert_array_equal(integration_state(env),state)
    assert controller.calls==calls
