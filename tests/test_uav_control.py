import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from uav_transport.control import (ControllerConfig, RecoveryConfig, RecoveryProtocol,
    TransportController, allocate_tensions, desired_rotation, mix_bounded,
    nominal_offsets, rotor_mixer, smooth_transition)


def config(**kwargs):
    values=dict(control_dt=.02,drone_mass=.034,payload_mass=.03,gravity=9.81,
        rotor_max=.16,motor_tau=.03,cable_length=.6,formation_radius=.25,
        formation_yaw=0.,inertia=(1.65717e-5,1.66556e-5,2.92617e-5),
        arm_length=.046,yaw_torque_ratio=.006)
    values.update(kwargs)
    return SimpleNamespace(**values)


def equilibrium(cfg=None):
    cfg=config() if cfg is None else cfg
    offsets=nominal_offsets(cfg)
    unit=offsets/cfg.cable_length
    tension=cfg.payload_mass*cfg.gravity/(3*unit[0,2])
    p=np.array([0.,0.,1.2])
    forces=unit*tension;forces[:,2]+=cfg.drone_mass*cfg.gravity
    obs=dict(payload_pos=p.copy(),payload_vel=np.zeros(3),uav_pos=p+offsets,
        uav_vel=np.zeros((3,3)),uav_rotation=np.stack([desired_rotation(f) for f in forces]),
        uav_omega=np.zeros((3,3)),payload_attachment_pos=p.copy(),
        uav_attachment_pos=p+offsets,cable_lengths=np.full(3,cfg.cable_length),
        cable_tensions=np.full(3,tension),cable_taut=np.ones(3,dtype=bool))
    return cfg,obs,forces


def test_rotor_mixer_agrees_with_lever_arm_and_reaction_torque_signs():
    cfg=config();matrix=rotor_mixer(cfg)
    angles=np.deg2rad([45.,135.,225.,315.])
    for i in range(4):
        force=np.zeros(4);force[i]=.1
        wrench=matrix@force
        r=cfg.arm_length*np.array([np.cos(angles[i]),np.sin(angles[i]),0.])
        expected=np.cross(r,[0,0,.1]);expected[2]=.1*cfg.yaw_torque_ratio*(1 if i%2==0 else -1)
        np.testing.assert_allclose(wrench,np.r_[.1,expected],atol=1e-15)


def test_motor_desaturation_preserves_collective_and_scales_moments_together():
    matrix=rotor_mixer(config())
    request=np.array([.4,.008,-.012,.004])
    actual=mix_bounded(request,matrix,.16)
    assert actual.min()>=0 and actual.max()<=.16
    wrench=matrix@actual
    np.testing.assert_allclose(wrench[0],request[0],atol=1e-14)
    ratios=wrench[1:]/request[1:]
    np.testing.assert_allclose(ratios,ratios[0],atol=1e-14)
    assert 0<ratios[0]<1


def test_desired_attitude_is_a_proper_rotation_with_thrust_aligned_to_force():
    force=np.array([.1,-.2,.5]);rotation=desired_rotation(force,.7)
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(rotation),1.,atol=1e-15)
    np.testing.assert_allclose(rotation[:,2],force/np.linalg.norm(force),atol=1e-15)


def test_actual_geometry_allocation_identifies_missing_horizontal_support():
    cfg=config();directions=nominal_offsets(cfg)/cfg.cable_length
    force=np.array([0,0,cfg.payload_mass*cfg.gravity])
    three,residual=allocate_tensions(directions,force,np.ones(3))
    np.testing.assert_allclose(residual,0.,atol=1e-14)
    expected=cfg.payload_mass*cfg.gravity*cfg.cable_length/(3*np.sqrt(cfg.cable_length**2-cfg.formation_radius**2))
    np.testing.assert_allclose(three,np.full(3,expected),atol=1e-14)
    two,bad=allocate_tensions(directions,force,np.array([0.,1.,1.]))
    assert two[0]==0 and np.linalg.norm(bad)>.05
    directions[1:,:2]=[[0,cfg.formation_radius/cfg.cable_length],[0,-cfg.formation_radius/cfg.cable_length]]
    two,good=allocate_tensions(directions,force,np.array([0.,1.,1.]))
    np.testing.assert_allclose(good,0.,atol=1e-14)
    np.testing.assert_allclose(two[1:],1.5*three[1:],atol=1e-14)


def test_equilibrium_controller_tracks_cable_reaction_with_zero_attitude_moment():
    cfg,obs,forces=equilibrium();controller=TransportController(cfg)
    motors=controller.command(obs,obs['payload_pos'])
    np.testing.assert_allclose(motors,np.repeat((np.linalg.norm(forces,axis=1)/4)[:,None],4,axis=1),atol=1e-14)
    np.testing.assert_allclose(controller.last_diagnostics['desired_moments'],0.,atol=1e-15)


def test_controller_memory_forks_exactly_and_observations_are_read_only():
    cfg,obs,_=equilibrium();controller=TransportController(cfg)
    source=copy.deepcopy(obs);goal=obs['payload_pos']+np.array([.05,.01,.02])
    controller.command(obs,goal);saved=controller.capture_memory()
    first=controller.command(obs,goal)
    controller.restore_memory(saved);second=controller.command(obs,goal)
    np.testing.assert_array_equal(first,second)
    for name in obs:np.testing.assert_array_equal(obs[name],source[name])
    saved['payload_integral'][0]=999
    assert controller.payload_integral[0]!=999


def test_support_maneuver_uses_constant_length_arcs_and_bounded_radial_closure():
    cfg=config();protocol=RecoveryProtocol(cfg)
    for phase,duration in [('rearrange',2.),('retension',2.),('restore',2.)]:
        protocol.phase=phase
        for age in np.linspace(0,duration,51):
            protocol.age=float(age);ref=protocol.reference()
            np.testing.assert_allclose(np.linalg.norm(ref.offsets[1:],axis=1),cfg.cable_length,atol=1e-14)
            assert np.linalg.norm(ref.offset_velocity[0])<=protocol.config.closing_speed_limit+1e-12
            if phase=='rearrange':np.testing.assert_allclose(ref.offsets[0],nominal_offsets(cfg)[0])
    for time in (0,1.):
        _,velocity=smooth_transition(np.zeros(3),np.ones(3),time,1.)
        np.testing.assert_array_equal(velocity,np.zeros(3))


def test_contact_gate_requires_real_tension_and_limited_closing_rate():
    cfg,obs,_=equilibrium();protocol=RecoveryProtocol(cfg)
    protocol.phase='confirm';obs['cable_tensions'][0]=0
    for _ in range(20):protocol.update(obs,obs['payload_pos'])
    assert protocol.phase=='confirm' and protocol.gate_dwell==0
    obs['cable_tensions'][0]=.02;obs['uav_vel'][0]=nominal_offsets(cfg)[0]*2
    for _ in range(20):protocol.update(obs,obs['payload_pos'])
    assert protocol.phase=='confirm' and protocol.gate_dwell==0
    obs['uav_vel'][:]=0
    for _ in range(10):protocol.update(obs,obs['payload_pos'])
    assert protocol.phase=='restore'


def test_final_gate_preserves_failed_payload_progress_and_has_finite_timeout():
    cfg,obs,_=equilibrium();protocol=RecoveryProtocol(cfg)
    protocol.phase='verify';goal=obs['payload_pos']+np.array([.2,0,0])
    for _ in range(300):protocol.update(obs,goal)
    assert protocol.failed and not protocol.done
    assert protocol.failure_reason=='verify_gate_timeout'
    assert protocol.elapsed<4.4


def test_protocol_memory_preserves_gate_dwell_and_next_reference():
    cfg,obs,_=equilibrium();protocol=RecoveryProtocol(cfg);protocol.phase='confirm'
    for _ in range(5):protocol.update(obs,obs['payload_pos'])
    saved=protocol.capture_memory()
    for _ in range(5):protocol.update(obs,obs['payload_pos'])
    ref1=protocol.reference();state1=protocol.capture_memory()
    protocol.restore_memory(saved)
    for _ in range(5):protocol.update(obs,obs['payload_pos'])
    ref2=protocol.reference()
    assert state1==protocol.capture_memory()
    np.testing.assert_array_equal(ref1.offsets,ref2.offsets)


def test_recovery_reference_rejects_a_declared_radial_speed_violation():
    with pytest.raises(ValueError,match='closing-speed'):
        RecoveryProtocol(config(),RecoveryConfig(retension_seconds=.1))
    with pytest.raises(ValueError,match='positive'):
        RecoveryProtocol(config(),RecoveryConfig(contact_confirm_seconds=0.))


def test_rotor_feedback_closes_real_mujoco_hover_and_translation():
    pytest.importorskip('mujoco')
    from uav_transport.plant import TransportConfig,TransportEnv
    cfg=TransportConfig();env=TransportEnv(cfg);controller=TransportController(cfg)
    obs=env.observe();start=obs['payload_pos'].copy();goal=start.copy()
    for step in range(350):
        if step==100:goal=start+np.array([.1,.03,.02])
        env.step(controller.command(obs,goal));obs=env.observe()
        assert obs['finite'] and not obs['floor_contact'] and not obs['uav_collision']
        assert obs['rotor_forces'].max()<=cfg.rotor_max+1e-12
    assert np.linalg.norm(obs['payload_pos']-goal)<.01
    assert np.all(obs['cable_tensions']>.02)
