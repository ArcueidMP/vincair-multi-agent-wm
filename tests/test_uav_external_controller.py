from copy import deepcopy
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
if shutil.which("cc") is None:
    pytest.skip("Pinned native controller requires a C compiler", allow_module_level=True)

from uav_transport import external_controller as external
from uav_transport.plant import TransportEnv
from uav_transport.snapshot import capture


def native(dt=.02):
    return external.NativeLee(.027, (2.3951e-5, 2.3951e-5, 3.2347e-5), dt)


def step(controller, *, goal=None, omega=None):
    z=np.zeros(3)
    return controller.step(z,z,np.eye(3),z if omega is None else omega,
        z if goal is None else goal,z,z)


def test_upstream_bytes_and_per_file_licenses_are_preserved():
    manifest=external.verify_upstream()
    assert manifest["commit"]==external.UPSTREAM_COMMIT
    assert len(manifest["files"])==7 and all(not row["modified"] for row in manifest["files"])
    licenses={row["path"]:row["license_notice"] for row in manifest["files"]}
    assert licenses["controller_lee.c"].startswith("MIT")
    assert licenses["stabilizer_types.h"].startswith("GPL-3.0")


def test_native_controller_keeps_published_gains_and_accepts_calibrated_mass_inertia():
    controller=native();params=controller.parameters()
    assert params["mass"]==pytest.approx(.027)
    np.testing.assert_allclose(params["inertia"],[2.3951e-5,2.3951e-5,3.2347e-5],rtol=1e-6)
    assert params["Kpos_P"]==[9.,9.,9.]
    assert params["Kpos_D"]==[7.,7.,7.]
    assert params["Kpos_I"]==[5.,5.,5.]
    np.testing.assert_allclose(params["KR"],[.0055,.0055,.01],rtol=1e-6)
    np.testing.assert_allclose(params["Komega"],[.0013,.0013,.002],rtol=1e-6)
    np.testing.assert_allclose(params["KI"],[.012,.018,.015],rtol=1e-6)


def test_native_hover_force_position_and_angular_velocity_signs():
    hover=step(native())
    np.testing.assert_allclose(hover,[.027*9.81,0,0,0],atol=3e-8,rtol=0)
    assert step(native(),goal=np.array([.1,0,0]))[2]>0
    assert step(native(),goal=np.array([0,.1,0]))[1]<0
    yaw=step(native(),omega=np.array([0,0,.3]))
    assert yaw[3]==pytest.approx(-.002*.3,rel=1e-6)


def test_clock_adaptation_updates_native_integrals_at_the_declared_period():
    slow,fast=native(.02),native(.002)
    goal=np.array([.1,0,0])
    step(slow,goal=goal);step(fast,goal=goal)
    assert slow.integrals()[0,0]==pytest.approx(.002,rel=1e-6)
    assert fast.integrals()[0,0]==pytest.approx(.0002,rel=1e-6)
    assert slow.parameters()["control_dt"]==pytest.approx(.02)
    with pytest.raises(ValueError,match="integer"):
        native(.03)


def test_native_buffers_are_independent_and_restore_every_controller_byte():
    first,second=native(),native()
    untouched=second.capture()
    step(first,goal=np.array([.1,.2,.03]))
    np.testing.assert_array_equal(second.capture(),untouched)
    saved=first.capture();expected=step(first,goal=np.array([.1,.2,.03]));after=first.capture()
    first.restore(saved);actual=step(first,goal=np.array([.1,.2,.03]))
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(first.capture(),after)


def test_bad_native_shapes_fail_before_c_pointer_access():
    controller=native();z=np.zeros(3)
    with pytest.raises(ValueError,match="vector shapes"):
        controller.step(np.zeros(2),z,np.eye(3),z,z,z,z)
    with pytest.raises(ValueError,match="positive physical"):
        external.NativeLee(.027,[1.,2.],.02)
    with pytest.raises(ValueError,match="wrong size"):
        controller.restore(np.zeros(10,dtype=np.uint8))


def test_shared_allocator_controller_is_explicit_bounded_and_does_not_write_physics():
    env=TransportEnv();controller=external.ExternalLeeController(env.config)
    observation=env.observe();before=capture(env)
    action=controller.command(observation,observation["payload_pos"])
    assert action.shape==(3,4) and np.isfinite(action).all()
    assert np.all((action>=0)&(action<=env.config.rotor_max))
    np.testing.assert_array_equal(capture(env).state,before.state)
    info=controller.configuration()
    assert info["family"]=="upstream_lee_position_attitude_shared_payload_allocator"
    assert info["upstream_controller"]=="controller_lee.c"
    assert info["executed_control_rate_hz"]==50 and info["upstream_default_attitude_rate_hz"]==500


def test_all_three_native_states_and_shared_integral_replay_deterministically():
    env=TransportEnv();controller=external.ExternalLeeController(env.config)
    observation=env.observe();goal=observation["payload_pos"]+np.array([.01,.02,0])
    controller.command(observation,goal);saved=controller.capture_memory()
    assert len(saved["native_states"])==3 and len({id(x) for x in saved["native_states"]})==3
    action=controller.command(observation,goal);after=controller.capture_memory()
    controller.restore_memory(saved)
    np.testing.assert_array_equal(controller.command(observation,goal),action)
    for a,b in zip(after["native_states"],controller.capture_memory()["native_states"]):
        np.testing.assert_array_equal(a,b)
    changed=deepcopy(saved);changed["native_build_key"]="different"
    with pytest.raises(ValueError,match="different native build"):
        controller.restore_memory(changed)


def test_pinned_source_hash_guard_rejects_a_changed_vendor_copy(monkeypatch,tmp_path):
    source=tmp_path/"controller_lee.c";source.write_text("original")
    manifest=dict(commit=external.UPSTREAM_COMMIT,files=[dict(path=source.name,sha256=external._sha(source))])
    (tmp_path/"UPSTREAM.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(external,"VENDOR",tmp_path)
    external.verify_upstream()
    source.write_text("changed")
    with pytest.raises(AssertionError,match="source changed"):
        external.verify_upstream()


def test_external_experiment_exposes_rule_arms_and_rejects_oracle_or_learned_mismatch():
    from uav_transport.external_experiment import schedule
    jobs=schedule()
    assert len(jobs)==24 and {arm for _,arm in jobs}=={"rule_fixed","rule_reconfigure"}
    for arms in (("true_option",),("wm_small",),("rule_fixed","rule_fixed")):
        with pytest.raises(ValueError,match="rule arms only"):
            schedule(arms)
