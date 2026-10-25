from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("mujoco")

from uav_transport.physics_audit import compare_replay, paired_summary


def pair():
    summary=dict(elapsed_seconds=1.,peak_donor_tension=.4,retension_tension_impulse=.01,
        completion=True,finite=True,collision=False,warning_count=0)
    states=np.zeros((51,64))
    nominal=dict(trace=dict(physical_states=states),summary=summary)
    replay=dict(physical_states=states.copy(),summary=deepcopy(summary),source_dt=.002,replay_dt=.001)
    return nominal,replay


def test_refinement_preserves_original_clock_and_checks_every_uav_position():
    nominal,replay=pair()
    replay["physical_states"][10,14]=.003
    result=compare_replay(nominal,replay)
    assert result["maximum_payload_position_difference_m"]==0
    assert result["maximum_uav_position_difference_m"]==.003
    assert not result["passes"]["position"]
    replay["physical_states"]=replay["physical_states"][:-1]
    with pytest.raises(ValueError,match="stopping time"):
        compare_replay(nominal,replay)


def test_discrete_completion_changes_remain_visible_when_continuous_checks_pass():
    nominal,replay=pair()
    replay["summary"].update(completion=False,peak_donor_tension=.39,retension_tension_impulse=.0095)
    result=compare_replay(nominal,replay)
    assert result["all_continuous_checks_pass"] and result["completion_changed"]
    assert result["source_completion"] and not result["replay_completion"]
    # A small impulse can have a large relative change while its declared
    # absolute change remains acceptable; both values must remain available.
    nominal["summary"]["retension_tension_impulse"]=.001
    replay["summary"]["retension_tension_impulse"]=.0001
    result=compare_replay(nominal,replay)
    assert result["passes"]["retension_impulse"]
    assert result["retension_impulse_relative_difference"]>8


def test_paired_effects_use_physical_identity_and_retain_all_operators():
    metrics=("peak_payload_excursion","peak_donor_tension","retension_tension_impulse",
        "elapsed_seconds","absolute_actuator_work","total_squared_thrust_integral",
        "final_payload_error","final_payload_speed","donor_slack_seconds")
    rows=[]
    for donor in (0,1):
        for arm in ("fixed_geometry","reconfigured"):
            summary={m:float(2 if arm=="fixed_geometry" else 1) for m in metrics}
            summary["completion"]=donor==1
            rows.append(dict(payload_mass=.03,formation_yaw=0.,donor=donor,arm=arm,summary=summary))
    result=paired_summary(list(reversed(rows)))
    assert result["cases"]==2 and result["fixed_completed"]==result["reconfigured_completed"]==1
    assert result["metrics"]["peak_payload_excursion"]["lower_count"]==2
    assert result["metrics"]["peak_payload_excursion"]["mean_difference"]==-1
    with pytest.raises(ValueError,match="Duplicated"):
        paired_summary(rows+[rows[0]])
    with pytest.raises(ValueError,match="both actual"):
        paired_summary(rows[:-1])
