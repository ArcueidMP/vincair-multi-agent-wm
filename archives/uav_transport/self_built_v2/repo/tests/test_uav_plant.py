from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from uav_transport.plant import TransportConfig, TransportEnv, initial_hover
from uav_transport.snapshot import capture, restore, integration_state, physical_state


def test_free_rigid_bodies_motor_activation_and_hover_balance():
    env = TransportEnv()
    assert (env.model.nq, env.model.nv, env.model.na, env.model.nu) == (28,24,12,12)
    assert np.allclose(env.model.body_mass[env.uav_body_ids],env.config.drone_mass)
    hover,_ = initial_hover(env.config)
    initial=env.observe()
    for _ in range(50): final=env.step(hover)
    assert np.linalg.norm(final["payload_pos"]-initial["payload_pos"]) < 1e-3
    assert np.linalg.norm(final["payload_vel"]) < 1e-3
    load_force=(final["cable_tensions"][:,None]*final["cable_unit_vectors"]).sum(0)
    np.testing.assert_allclose(load_force,[0.,0.,env.config.payload_mass*env.config.gravity],atol=2e-4)
    assert final["cable_taut"].all() and final["finite"] and not final["warnings"].any()
    assert not final["collision_pairs"]
    assert env.control_steps==50 and env.physics_steps==500
    assert abs(env.data.time-1.)<1e-12


def test_motor_lag_and_clamps_use_real_activation_state():
    env=TransportEnv(); initial=env.data.act.copy()
    command=np.zeros((3,4));command[1]=1.
    env.step(command)
    bounded=np.clip(command,0.,env.config.rotor_max).reshape(-1)
    expected=bounded+(initial-bounded)*np.exp(-env.config.control_dt/env.config.motor_tau)
    np.testing.assert_allclose(env.data.act,expected,atol=1e-15,rtol=1e-13)
    assert np.all(env.data.act>=0) and np.all(env.data.act<=env.config.rotor_max)
    np.testing.assert_array_equal(env.last_command,np.clip(command,0.,env.config.rotor_max))
    np.testing.assert_allclose(env.metrics["clamp_intervention_time"],[0.,.02,0.])
    assert env.metrics["saturation_time"].max()==0  # Motors have not yet reached the command.
    for _ in range(12):env.step(np.full((3,4),env.config.rotor_max))
    assert env.metrics["saturation_time"].min()>0
    assert env.metrics["command_saturation_time"].min()>.2


def test_rotor_moments_are_transmitted_through_body_frame_not_world_force_control():
    env=TransportEnv();hover,_=initial_hover(env.config)
    command=hover.copy();command[0,0]+=.03;command[0,2]-=.03
    observation=env.step(command)
    assert np.linalg.norm(observation["uav_omega"][0]) > .05
    assert np.linalg.norm(observation["uav_rotation"][0]-initial_hover(env.config)[1][0]) > 1e-5
    # Changing a rotor causes rotation while the other airframes remain untorqued.
    assert np.max(np.abs(observation["uav_omega"][1:]))<1e-6


def test_physical_downward_disturbance_slackens_one_cable_and_redistributes_load():
    env=TransportEnv();hover,_=initial_hover(env.config)
    for _ in range(20):env.step(hover)
    before=env.observe()
    external=np.zeros((4,3));external[1,2]=-.4
    for _ in range(5):after=env.step(hover,external)
    assert after["cable_lengths"][0]<env.config.cable_length-.01
    assert after["cable_tensions"][0]==0 and not after["cable_taut"][0]
    assert (after["cable_tensions"][1:]>before["cable_tensions"][1:]).all()
    assert np.linalg.norm(after["payload_pos"]-before["payload_pos"])>.005
    assert after["finite"] and not after["collision_pairs"]
    env.step(hover)
    assert not env.data.xfrc_applied.any() and not env.last_external_forces.any()


def test_exact_snapshot_restores_motors_external_forces_warmstarts_and_resources():
    env=TransportEnv();hover,_=initial_hover(env.config)
    ext=np.zeros((4,3));ext[1]=[.03,.02,-.2]
    for _ in range(5):env.step(hover,ext)
    snapshot=capture(env); clone=env.clone()
    np.testing.assert_array_equal(integration_state(env),integration_state(clone))
    np.testing.assert_array_equal(clone.data.xfrc_applied,env.data.xfrc_applied)
    np.testing.assert_array_equal(clone.data.qacc_warmstart,env.data.qacc_warmstart)
    for i in range(15):
        cmd=hover.copy();cmd[1]*=1+.1*np.sin(i)
        env.step(cmd);clone.step(cmd)
        np.testing.assert_array_equal(physical_state(env),physical_state(clone))
        np.testing.assert_array_equal(integration_state(env),integration_state(clone))
    for key in env.metrics:
        np.testing.assert_array_equal(env.metrics[key],clone.metrics[key])
    restore(env,snapshot)
    np.testing.assert_array_equal(integration_state(env),snapshot.state)
    assert env.control_steps==5 and env.physics_steps==50


def test_timestep_transfer_requires_permission_and_preserves_all_physical_state():
    env=TransportEnv();hover,_=initial_hover(env.config)
    for _ in range(3):env.step(hover)
    snapshot=capture(env)
    fine=TransportEnv(replace(env.config,dt=.001))
    with pytest.raises(ValueError,match="configuration"):
        restore(fine,snapshot)
    restore(fine,snapshot,allow_timestep_change=True)
    np.testing.assert_array_equal(physical_state(env),physical_state(fine))
    assert fine.physics_steps==60 and fine.control_steps==3
    changed=TransportEnv(replace(env.config,payload_mass=.04))
    with pytest.raises(ValueError,match="configuration"):
        restore(changed,snapshot,allow_timestep_change=True)


def test_observations_are_copies_and_invalid_input_cannot_advance_the_clock():
    env=TransportEnv();before=integration_state(env)
    obs=env.observe();obs["payload_pos"][:]=99;obs["uav_rotation"][:]=0
    np.testing.assert_array_equal(integration_state(env),before)
    for cmd in (np.zeros((3,3)),np.full((3,4),np.nan)):
        with pytest.raises(ValueError):env.step(cmd)
    np.testing.assert_array_equal(integration_state(env),before)
    with pytest.raises(ValueError,match="integer multiple"):
        TransportConfig(dt=.003)


def test_snapshot_source_is_immutable_and_complete_state_detects_hidden_motor_changes():
    env=TransportEnv();snap=capture(env)
    saved=snap.state.copy();env.data.act[0]*=.5
    assert not np.array_equal(integration_state(env),saved)
    restore(env,snap)
    np.testing.assert_array_equal(integration_state(env),saved)
    env.metrics["rotor_force_integral"][0,0]=3.
    assert snap.metrics["rotor_force_integral"][0,0]==0
