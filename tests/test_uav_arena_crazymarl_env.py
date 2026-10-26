"""JAX side of CrazyMARL on arena tasks; runs in the CrazyMARL venv with its pinned checkout.

    CRAZYMARL_CHECKOUT=<checkout> JAX_PLATFORMS=cpu PYTHONPATH=$PWD <crazymarl venv>/bin/python -m pytest -q tests/test_uav_arena_crazymarl_env.py
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("brax")
pytest.importorskip("mujoco")
CHECKOUT = Path(os.environ.get("CRAZYMARL_CHECKOUT", Path(__file__).resolve().parents[1]/"data/external/CrazyMARL"))
pytestmark = pytest.mark.skipif(not (CHECKOUT/"crazymarl").is_dir(), reason="needs the pinned CrazyMARL checkout")


@pytest.fixture(scope="module")
def arena(tmp_path_factory):
    from uav_arena import crazymarl_task as task
    from uav_transport.crazymarl_baseline import upstream_context
    path = tmp_path_factory.mktemp("crazymarl")/"task.npz"
    np.savez(path, **_constants(task))
    with upstream_context(CHECKOUT):
        import jax
        from uav_arena.crazymarl_env import ArenaMultiQuadEnv
        env = ArenaMultiQuadEnv(task_path=str(path), obs_noise=0.0)
        yield env, jax.jit(env.reset), jax.jit(env.step)


def _constants(task):
    """Two layouts and the settled-hover constants, without the arena venv's MuJoCo plant."""
    pillars = np.array([[[0.0, 0.8, 0.15], [0.0, -0.8, 0.15], [5.0, 5.0, 0.1], [-5.0, 5.0, 0.1]]]*2)
    movers = np.zeros((2, task.N_MOVERS, 9))
    movers[1, 0] = [0.0, -1.3, 1.6, 0.0, 1.3, 1.6, 0.4, 0.0, 0.15]
    quat = np.array([[1.0, 0.0, 0.0, 0.0]]*3)
    angles = np.arange(3)*2*np.pi/3
    formation = np.column_stack((0.25*np.cos(angles), 0.25*np.sin(angles), np.full(3, np.sqrt(0.6**2-0.25**2))))
    return dict(start=np.array([[-1.6, 0.0, 1.2]]*2), goal=np.array([[1.6, 0.0, 1.2]]*2), pillars=pillars,
                movers=movers, formation=formation, hover_quat=quat, hover_thrust=np.full((3, 4), 0.0872),
                drone_mass=np.array(0.027), inertia=np.array([2.3951e-05, 2.3951e-05, 3.2347e-05]),
                cable_length=np.array(0.6), payload_mass=np.array(0.025), motor_tau=np.array(0.03),
                rotor_max=np.array(0.16))


def test_arena_plant_parameters_and_observation_size(arena):
    from uav_arena import crazymarl_task as task
    env, reset, _ = arena
    model = env.sys.mj_model
    masses = [model.body(f"q{i}_cf2").mass[0] for i in range(3)]
    assert masses == pytest.approx([0.027]*3) and model.body("payload").mass[0] == pytest.approx(0.025)
    assert model.tendon_range[:, 1] == pytest.approx([0.6]*3)
    state = reset(__import__("jax").random.PRNGKey(0))
    assert state.obs.shape == (task.GLOBAL_SIZE,)


def test_teammates_and_payload_are_seen_late(arena):
    import jax
    env, reset, step = arena
    state = reset(jax.random.PRNGKey(1))
    hover = 2*env.hover_thrust/state.metrics["max_thrust"]-1
    steps = int(env.latency_steps[state.metrics["latency"]])
    history = [np.asarray(state.metrics["history"][0])]
    for _ in range(max(steps, 1)+2):
        state = step(state, hover)
        history.append(np.asarray(state.metrics["history"][0]))
    delayed = np.asarray(state.metrics["history"][steps])
    np.testing.assert_allclose(delayed, history[-1-steps], atol=1e-6)


def test_auto_reset_restarts_the_delay_buffer(arena):
    import jax
    import jax.numpy as jp
    env, reset, step = arena
    state = reset(jax.random.PRNGKey(2))
    hover = 2*env.hover_thrust/state.metrics["max_thrust"]-1
    stale = state.replace(metrics=dict(state.metrics, history=jp.zeros_like(state.metrics["history"]),
                                       distance=jp.array(99.0, dtype=jp.float32)))
    after = step(stale, hover)
    np.testing.assert_allclose(np.asarray(after.metrics["history"][1:]), np.asarray(state.metrics["first_history"][:-1]))
    assert abs(float(after.reward)) < 50       # the progress term starts from the episode's first distance
