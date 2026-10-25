"""Explicit plant-family routing for current experiments and archived traces."""
from dataclasses import asdict
from pathlib import Path

from .plant import TransportConfig, TransportEnv


LEGACY = "self_built_v2"
MENAGERIE = "menagerie_crazyflie_2"
FAMILIES = (LEGACY, MENAGERIE)


def family(config):
    if isinstance(config, dict):
        return config.get("plant_family", LEGACY)
    return getattr(config, "plant_family", LEGACY)


def make_config(plant_family=LEGACY, **values):
    if plant_family == LEGACY:
        return TransportConfig(**values)
    if plant_family == MENAGERIE:
        from .menagerie import MenagerieTransportConfig
        return MenagerieTransportConfig(**values)
    raise ValueError(f"Unknown plant family: {plant_family}")


def config_from_dict(values):
    values = dict(values)
    return make_config(values.pop("plant_family", LEGACY), **values)


def make_env(config):
    if isinstance(config, dict):
        config = config_from_dict(config)
    if family(config) == MENAGERIE:
        from .menagerie import MenagerieTransportEnv
        return MenagerieTransportEnv(config)
    if family(config) == LEGACY:
        return TransportEnv(config)
    raise ValueError(f"Unknown plant family: {family(config)}")


def source_paths(plant_family):
    """Files that identify the plant implementation and its vendored assets."""
    paths = [Path(__file__), Path(__file__).with_name("plant.py")]
    if plant_family == MENAGERIE:
        paths.append(Path(__file__).with_name("menagerie.py"))
        root = Path(__file__).resolve().parents[1] / "third_party/menagerie/bitcraze_crazyflie_2"
        paths.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    elif plant_family != LEGACY:
        raise ValueError(f"Unknown plant family: {plant_family}")
    return paths


def require_model_family(model, config):
    """Prevent historical weights from silently controlling a new airframe."""
    metadata = getattr(model, "metadata", None)
    if metadata is not None and metadata.get("plant_family", LEGACY) != family(config):
        raise ValueError("Frozen model plant family differs from the execution plant; retrain on its local data")
