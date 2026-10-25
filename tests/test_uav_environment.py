from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from uav_transport.environment import (
    LEGACY, MENAGERIE, config_from_dict, family, make_config, make_env,
    require_model_family,
)
from uav_transport.local_model import dataset_family
from uav_transport.plant import TransportConfig, TransportEnv


def test_historical_unlabelled_configuration_keeps_original_plant():
    original = TransportConfig()
    restored = config_from_dict(asdict(original))
    assert type(restored) is TransportConfig and restored == original
    assert family(restored) == LEGACY
    assert type(make_env(restored)) is TransportEnv
    assert dataset_family({}) == LEGACY


def test_new_airframe_is_identified_through_config_and_snapshot_forks():
    config = make_config(MENAGERIE)
    assert family(config) == MENAGERIE
    assert config_from_dict(asdict(config)) == config
    env = make_env(config)
    clone = env.clone()
    assert type(clone) is type(env)
    assert clone.model_hash == env.model_hash
    assert env.config.drone_mass == pytest.approx(.027)
    np.testing.assert_array_equal(clone.data.qpos, env.data.qpos)


def test_old_weights_cannot_silently_control_new_airframe_or_reverse():
    legacy = make_config(LEGACY)
    current = make_config(MENAGERIE)
    old = SimpleNamespace(metadata={})
    new = SimpleNamespace(metadata={"plant_family": MENAGERIE})
    require_model_family(old, legacy)
    require_model_family(new, current)
    with pytest.raises(ValueError, match="retrain"):
        require_model_family(old, current)
    with pytest.raises(ValueError, match="retrain"):
        require_model_family(new, legacy)


def test_dataset_family_is_single_and_explicit():
    assert dataset_family({"plant_family": np.asarray(MENAGERIE)}) == MENAGERIE
    for value in (np.asarray([MENAGERIE, LEGACY]), np.asarray("unknown")):
        with pytest.raises(ValueError, match="one supported"):
            dataset_family({"plant_family": value})
    with pytest.raises(ValueError, match="Unknown plant"):
        make_config("unknown")


def test_external_feedback_cannot_silently_replace_the_model_training_controller():
    from uav_transport.mission import run_mission, mission_cases
    model = SimpleNamespace(metadata={"plant_family": MENAGERIE})
    with pytest.raises(ValueError, match="retain their training controller"):
        run_mission(mission_cases()[0], "wm_small", model,
                    plant_family=MENAGERIE, controller_factory=object)
