from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from uav_transport.mission_replay import score_mission,trajectory_errors,validate_clock


def synthetic():
    names=("settle","leg_1","operation_1","leg_2","operation_2","leg_3")
    stages=[];labels=[];operations=[];observations=[];position=np.zeros(3)
    def observation(p):
        return dict(payload_pos=np.asarray(p).copy(),payload_vel=np.zeros(3),cable_tensions=np.full(3,.1),
            cable_gaps=np.zeros(3),finite=True,warnings=np.zeros(8,dtype=int),collision_pairs=[])
    observations.append(observation(position));begin=0
    for index,name in enumerate(names):
        length=2 if name=="settle" else 5
        if name.startswith("leg_"):position=np.array([int(name[-1])*.1,0.,0.])
        end=begin+length
        stages.append(dict(name=name,begin_step=begin,end_step=end,target=position.copy()))
        for _ in range(length):observations.append(observation(position))
        if name.startswith("operation"):
            observations[begin+1]["cable_tensions"][0]=0.
            observations[begin+1]["cable_gaps"][0]=.02
            operations.append(dict(mission_begin_step=begin,mission_end_step=end,goal=position.copy(),donor=0,
                recovery_config=dict(verify_seconds=.3,support_threshold=.02,gap_tolerance=.003,
                    payload_error_limit=.05,payload_speed_limit=.08),
                end_protocol_memory=dict(done=True,failed=False,failure_reason=None)))
        labels.extend([name]*length);begin=end
    n=begin
    physical=np.zeros((n+1,64))
    for j in (0,7,14,21):physical[:,j+3]=1.
    trace=dict(actions=np.zeros((n,3,4)),external_forces=np.zeros((n,4,3)),phases=labels.copy(),
        stages=labels,references=np.stack([x["payload_pos"] for x in observations[1:]]),
        physical_states=physical,integration_states=physical.copy())
    result=dict(initial_snapshot=SimpleNamespace(control_steps=0),summary=dict(control_steps=n),
        plant_config=dict(control_dt=.1),mission_config=dict(goal_dwell_seconds=.3,move_seconds=.2,
            position_tolerance=.025,speed_tolerance=.05,required_slack_seconds=.1,
            payload_excursion_limit=.05,donor_tension_limit=.6),stages=stages,operations=operations,trace=trace)
    raw=dict(control_index=np.arange(n),finite=np.ones(n,dtype=bool),collision=np.zeros(n,dtype=bool),
        payload_position=np.stack([x["payload_pos"] for x in observations[1:]]),
        cable_tensions=np.stack([x["cable_tensions"] for x in observations[1:]]),
        cable_gaps=np.stack([x["cable_gaps"] for x in observations[1:]]),
        uav_tilt=np.zeros((n,3)),minimum_body_height=np.ones(n),minimum_uav_separation=np.full(n,.3))
    return result,dict(dt=.1,observations=observations,substeps=raw)


def test_failed_first_leg_cannot_be_repaired_by_later_delivery():
    result,replay=synthetic()
    assert score_mission(result,replay)["constrained_completion"]
    first=result["stages"][1]
    replay["observations"][first["end_step"]]["payload_pos"]+=np.array([.1,0.,0.])
    scored=score_mission(result,replay)
    assert scored["waypoint_successes"]==[False,True,True]
    assert not scored["delivery_completion"]


def test_timed_out_option_retains_its_failure_even_with_a_valid_late_pose_window():
    result,replay=synthetic()
    result["operations"][0]["end_protocol_memory"].update(done=False,failed=True,failure_reason="verify_gate_timeout")
    scored=score_mission(result,replay)
    assert scored["operations"][0]["verified_final_dwell"]
    assert not scored["operations"][0]["completion"]
    assert scored["option_completions"]==[False,True]
    assert not scored["delivery_completion"]


def test_only_original_stopping_windows_and_active_prefix_contribute():
    result,replay=synthetic();n=result["summary"]["control_steps"]
    baseline=score_mission(result,replay)
    padding=deepcopy(replay["observations"][-1]);padding.update(finite=False,collision_pairs=[(0,1)],warnings=np.ones(8,dtype=int))
    replay["observations"].extend([padding]*3)
    for key,values in replay["substeps"].items():
        tail=np.repeat(values[-1:],3,axis=0)
        if key=="control_index":tail=np.arange(n,n+3)
        if key=="collision":tail[:]=True
        if key=="finite":tail[:]=False
        replay["substeps"][key]=np.concatenate((values,tail),axis=0)
    scored=score_mission(result,replay)
    assert scored["constrained_completion"] and scored["finite"] and not scored["collision"]
    assert scored["physics_steps"]==baseline["physics_steps"]==n
    # A missing original terminal state cannot be replaced with later padding.
    last=result["stages"][-1]
    replay["observations"][last["end_step"]]["payload_pos"]+=np.array([.1,0.,0.])
    assert not score_mission(result,replay)["waypoint_successes"][-1]


def test_clock_validation_requires_one_uninterrupted_stage_prefix():
    result,_=synthetic()
    assert validate_clock(result)==(0,27)
    result["stages"][2]["begin_step"]+=1
    with pytest.raises(ValueError,match="uninterrupted"):
        validate_clock(result)


def test_refinement_checks_uav_attitude_and_quaternion_double_cover():
    source=np.zeros((4,64))
    for j in (0,7,14,21):source[:,j+3]=1.
    same=source.copy();same[:,17:21]*=-1
    assert trajectory_errors(source,same)["maximum_body_attitude_rad"]==0.
    changed=source.copy();changed[2,24:28]=[np.cos(.03/2),0.,0.,np.sin(.03/2)]
    result=trajectory_errors(source,changed)
    assert result["position_pass"] and not result["attitude_pass"]
    assert result["maximum_body_attitude_rad"]==pytest.approx(.03)
