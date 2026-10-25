from dataclasses import asdict, replace

import numpy as np
import pytest

mujoco=pytest.importorskip("mujoco")

from uav_transport.control import RecoveryConfig, TransportController
from uav_transport.evaluation import fork, replay_option, run_option
from uav_transport.menagerie import (MenagerieTransportConfig,MenagerieTransportEnv,
    OFFICIAL_INERTIA,OFFICIAL_MASS,UPSTREAM_REVISION,compare_official_airframes,verify_vendor)
from uav_transport.plant import TransportEnv,initial_hover
from uav_transport.snapshot import capture,integration_state,restore


def prepared():
    env=MenagerieTransportEnv();controller=TransportController(env.config)
    goal=env.observe()["payload_pos"].copy()
    for _ in range(100):env.step(controller.command(env.observe(),goal))
    return env,controller


def test_official_source_bytes_and_every_compiled_collision_transform_match():
    manifest=verify_vendor()
    assert manifest["revision"]==UPSTREAM_REVISION and len(manifest["files"])==41
    env=MenagerieTransportEnv();comparison=compare_official_airframes(env)
    assert comparison["all_airframes_match"]
    assert comparison["reference_mass"]==OFFICIAL_MASS
    assert comparison["reference_inertia"]==list(OFFICIAL_INERTIA)
    assert all(x["meshes"]==39 and x["collision_geoms"]==32 for x in comparison["vehicles"])
    assert env.model.ngeom==119 and env.model.nmesh==39 and env.model.nu==12
    assert env.model.nsensordata==30
    assert asdict(env.config)["plant_family"]=="menagerie_crazyflie_2"


def test_legacy_defaults_and_independent_model_copies_are_preserved():
    legacy=TransportEnv();new=MenagerieTransportEnv();clone=new.clone()
    assert legacy.config.drone_mass==.034 and legacy.model.nmesh==0
    assert isinstance(clone,MenagerieTransportEnv)
    np.testing.assert_array_equal(integration_state(new),integration_state(clone))
    # Clones own model arrays even though compilation templates are cached.
    clone.model.body_mass[clone.uav_body_ids[0]]*=1.1
    assert new.model.body_mass[new.uav_body_ids[0]]==OFFICIAL_MASS
    another=MenagerieTransportEnv()
    assert another.model.body_mass[another.uav_body_ids[0]]==OFFICIAL_MASS
    assert np.allclose(legacy.model.body_mass[legacy.uav_body_ids],.034)


def test_official_mass_has_balanced_hover_and_filtered_per_rotor_caps():
    env=MenagerieTransportEnv();hover,_=initial_hover(env.config)
    for _ in range(50):obs=env.step(hover)
    assert np.linalg.norm(obs["payload_pos"]-np.array([0.,0.,1.2]))<1e-3
    assert obs["cable_taut"].all() and obs["finite"] and not obs["warnings"].any()
    before=env.data.act.copy();requested=np.full((3,4),1.)
    env.step(requested)
    expected=env.config.rotor_max+(before-env.config.rotor_max)*np.exp(-env.config.control_dt/env.config.motor_tau)
    np.testing.assert_allclose(env.data.act,expected,atol=1e-15)
    assert env.data.act.max()<=env.config.rotor_max
    assert env.last_command.max()==.16
    assert mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_ACTUATOR,"body_thrust")==-1
    with pytest.raises(ValueError,match="official"):
        MenagerieTransportConfig(drone_mass=.034)


def test_official_airframe_snapshot_fork_is_bitwise_and_keeps_source_identity():
    env,controller=prepared();other,feedback=fork(env,controller)
    goal=env.observe()["payload_pos"].copy();goal[0]+=.04
    for _ in range(20):
        a=controller.command(env.observe(),goal);b=feedback.command(other.observe(),goal)
        np.testing.assert_array_equal(a,b)
        env.step(a);other.step(b)
        np.testing.assert_array_equal(integration_state(env),integration_state(other))
    snapshot=capture(env)
    assert snapshot.config["menagerie_revision"]==UPSTREAM_REVISION
    fine=MenagerieTransportEnv(replace(env.config,dt=.001))
    restore(fine,snapshot,allow_timestep_change=True)
    np.testing.assert_array_equal(integration_state(env),integration_state(fine))
    with pytest.raises(ValueError,match="configuration"):
        restore(TransportEnv(),snapshot,allow_timestep_change=True)


def test_real_unloading_recovery_and_refinement_use_the_official_model():
    env,controller=prepared()
    result=run_option(env,controller,RecoveryConfig(),0,True)
    assert result["summary"]["completion"] and result["summary"]["donor_slack_seconds"]>1.
    assert result["plant_config"]["plant_family"]=="menagerie_crazyflie_2"
    assert not result["summary"]["collision"] and result["summary"]["finite"]
    nominal=replay_option(result)
    assert nominal["source_replay_bitwise"]
    np.testing.assert_array_equal(nominal["integration_states"],result["trace"]["integration_states"])
    fine=replay_option(result,dt=.001)
    assert fine["summary"]["completion"]
    discrepancy=np.linalg.norm(fine["physical_states"][:,:3]-result["trace"]["physical_states"][:,:3],axis=1)
    assert discrepancy.max()<.002
