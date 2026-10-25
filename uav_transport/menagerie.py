"""Cooperative transport using the official Menagerie Crazyflie 2 airframe.

The pinned upstream meshes, 32 collision pieces, mass, inertial frame and
inertia are retained for each aircraft. Team cables, the central payload and
bounded filtered per-rotor actuation are explicit transport-study additions.
Upstream's arbitrary body-wrench actuator ranges are not used.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .plant import TransportConfig, TransportEnv, make_xml as legacy_xml, rotor_geometry


VENDOR = Path(__file__).resolve().parents[1]/"third_party"/"menagerie"/"bitcraze_crazyflie_2"
UPSTREAM_REVISION = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
UPSTREAM_MANIFEST_SHA256 = "de268445f3b5f6fab4f5c4e835d20ae098ab3b8d77e58d4b8a841b873c7d6226"
OFFICIAL_MASS = .027
OFFICIAL_INERTIA = (2.3951e-5, 2.3951e-5, 3.2347e-5)


@dataclass(frozen=True)
class MenagerieTransportConfig(TransportConfig):
    plant_family: str = "menagerie_crazyflie_2"
    drone_mass: float = OFFICIAL_MASS
    inertia: tuple = OFFICIAL_INERTIA
    air_density: float = 1.225
    air_viscosity: float = 1.8e-5
    menagerie_revision: str = UPSTREAM_REVISION
    menagerie_manifest_sha256: str = UPSTREAM_MANIFEST_SHA256

    def __post_init__(self):
        super().__post_init__()
        if self.plant_family!="menagerie_crazyflie_2":
            raise ValueError("The Menagerie configuration must identify its actual plant family")
        if self.drone_mass!=OFFICIAL_MASS or tuple(self.inertia)!=OFFICIAL_INERTIA:
            raise ValueError("This profile preserves the pinned official Crazyflie mass and inertia")
        if self.menagerie_revision!=UPSTREAM_REVISION or self.menagerie_manifest_sha256!=UPSTREAM_MANIFEST_SHA256:
            raise ValueError("The requested airframe source differs from the pinned vendor manifest")
        if not np.isfinite([self.air_density,self.air_viscosity]).all() or min(self.air_density,self.air_viscosity)<0:
            raise ValueError("Air density and viscosity must be finite and nonnegative")


def verify_vendor():
    """Verify every vendored source byte before composing a physical model."""
    manifest_path=VENDOR/"UPSTREAM.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=UPSTREAM_MANIFEST_SHA256:
        raise AssertionError("Menagerie source manifest changed")
    manifest=json.loads(manifest_path.read_text())
    if manifest["revision"]!=UPSTREAM_REVISION:
        raise AssertionError("Unexpected Menagerie revision")
    for relative,digest in manifest["files"].items():
        if hashlib.sha256((VENDOR/relative).read_bytes()).hexdigest()!=digest:
            raise AssertionError("Vendored Menagerie file changed: "+relative)
    return manifest


def make_xml(config):
    """Compose three unchanged official airframe bodies into the transport scene."""
    verify_vendor()
    official=ET.parse(VENDOR/"cf2.xml").getroot()
    root=ET.fromstring(legacy_xml(config))
    root.set("model","menagerie_crazyflie2_team_transport")
    compiler=root.find("compiler")
    compiler.set("meshdir","assets")
    # Each aircraft retains an explicit upstream inertial tag; the added
    # spherical payload obtains its explicitly requested mass from its geom.
    compiler.set("inertiafromgeom","auto")
    option=root.find("option")
    option.set("density",str(config.air_density));option.set("viscosity",str(config.air_viscosity))
    world=root.find("worldbody")
    root.insert(list(root).index(world),deepcopy(official.find("default")))
    root.insert(list(root).index(world),deepcopy(official.find("asset")))
    template=official.find("worldbody/body")
    sensors=ET.SubElement(root,"sensor")
    for index in range(3):
        original=world.find(f"body[@name='uav{index}']")
        aircraft=deepcopy(template)
        for element in aircraft.iter():
            if "name" in element.attrib:
                element.set("name",f"uav{index}_"+element.get("name"))
        aircraft.set("name",f"uav{index}")
        aircraft.set("pos",original.get("pos"));aircraft.set("quat",original.get("quat"))
        aircraft.find("freejoint").set("name",f"uav{index}_joint")
        for j,geom in enumerate(aircraft.findall("geom")):
            geom.set("name",f"uav{index}_meshgeom{j}")
        # Author-added force application and cable sites inherit the official
        # body's frame. Its compiled COM is exactly this frame's origin.
        for site in original.findall("site"):
            aircraft.append(deepcopy(site))
        world.remove(original);world.append(aircraft)
        for sensor in official.find("sensor"):
            item=deepcopy(sensor)
            item.set("name",f"uav{index}_"+item.get("name"))
            for attribute in ("site","objname"):
                if attribute in item.attrib:item.set(attribute,f"uav{index}_"+item.get(attribute))
            sensors.append(item)
    return ET.tostring(root,encoding="unicode")


@lru_cache(maxsize=24)
def _compiled_template(xml,manifest_digest):
    # The digest is part of the cache identity. Models handed to environments
    # are independent native copies, keeping their model arrays isolated.
    assets={str(path.relative_to(VENDOR)):path.read_bytes() for path in (VENDOR/"assets").glob("*.obj")}
    return mujoco.MjModel.from_xml_string(xml,assets=assets)


class MenagerieTransportEnv(TransportEnv):
    """The legacy plant API backed by independently copied official airframes."""
    def __init__(self,config=None):
        self.config=MenagerieTransportConfig() if config is None else config
        if not isinstance(self.config,MenagerieTransportConfig):
            raise TypeError("Use MenagerieTransportConfig for this plant")
        self.xml=make_xml(self.config)
        self.model_hash=hashlib.sha256((self.xml+"\0"+UPSTREAM_MANIFEST_SHA256).encode()).hexdigest()
        self.model=copy(_compiled_template(self.xml,UPSTREAM_MANIFEST_SHA256))
        self.data=mujoco.MjData(self.model)
        self.substeps=round(self.config.control_dt/self.config.dt)
        names=["payload"]+[f"uav{i}" for i in range(3)]
        self.body_ids=np.array([self.model.body(name).id for name in names])
        self.payload_body_id=int(self.body_ids[0]);self.uav_body_ids=self.body_ids[1:]
        joints=[self.model.joint(name+"_joint") for name in names]
        self.qpos_addresses=np.array([int(j.qposadr[0]) for j in joints])
        self.dof_addresses=np.array([int(j.dofadr[0]) for j in joints])
        self.tendon_ids=np.array([self.model.tendon(f"cable{i}").id for i in range(3)])
        self.rotor_site_ids=np.array([[self.model.site(f"uav{i}_rotor{j}").id for j in range(4)] for i in range(3)])
        self.actuator_ids=np.array([[self.model.actuator(f"uav{i}_motor{j}").id for j in range(4)] for i in range(3)])
        self.rotor_positions,self.mixer=rotor_geometry(self.config)
        self.reset()

    def clone(self):
        from .snapshot import capture,restore
        other=MenagerieTransportEnv(self.config)
        restore(other,capture(self))
        return other

    def specification(self):
        return dict(**super().specification(),plant_family=self.config.plant_family,
            airframe_source=dict(repository="https://github.com/google-deepmind/mujoco_menagerie",
                revision=UPSTREAM_REVISION,subdirectory="bitcraze_crazyflie_2",
                manifest_sha256=UPSTREAM_MANIFEST_SHA256),
            preserved_airframe_properties="Official meshes, collision decomposition, materials, mass, inertia, COM and IMU sites",
            added_transport_components="Central payload, unilateral team cables, bounded four-rotor force mapping and first-order motor activation",
            actuator_parameters=dict(rotor_max_n=self.config.rotor_max,motor_tau_s=self.config.motor_tau,
                arm_radius_m=self.config.arm_length,yaw_moment_per_thrust_m=self.config.yaw_torque_ratio),
            actuator_provenance="Study parameters; the upstream body-wrench ctrlrange limits are replaced explicitly",
            fluid_parameters=dict(density=self.config.air_density,viscosity=self.config.air_viscosity),
            integration_note="Transport uses implicitfast with registered dt; the standalone upstream airframe uses RK4")


def compare_official_airframes(env):
    """Verify compiled transforms and inertial/collision properties against upstream."""
    verify_vendor()
    reference=mujoco.MjModel.from_xml_path(str(VENDOR/"cf2.xml"))
    source_body=reference.body("cf2").id
    source_geoms=np.flatnonzero(reference.geom_bodyid==source_body)
    checks=[]
    for vehicle,body in enumerate(env.uav_body_ids):
        target_geoms=np.flatnonzero(env.model.geom_bodyid==body)
        if len(target_geoms)!=len(source_geoms):
            raise AssertionError("Aircraft mesh/collision count differs from the official model")
        fields=("geom_type","geom_group","geom_contype","geom_conaffinity","geom_pos","geom_quat","geom_size","geom_friction")
        differences={name:float(np.max(np.abs(getattr(reference,name)[source_geoms]-getattr(env.model,name)[target_geoms]))) for name in fields}
        inertial={name:float(np.max(np.abs(getattr(reference,name)[source_body]-getattr(env.model,name)[body])))
            for name in ("body_mass","body_ipos","body_inertia","body_iquat")}
        checks.append(dict(vehicle=vehicle,geometry_differences=differences,inertial_differences=inertial,
            meshes=len(source_geoms),collision_geoms=int(np.sum(reference.geom_contype[source_geoms]!=0)),
            matches=all(v==0 for v in (*differences.values(),*inertial.values()))))
    return dict(reference_revision=UPSTREAM_REVISION,reference_mass=float(reference.body_mass[source_body]),
        reference_inertia=reference.body_inertia[source_body].tolist(),vehicles=checks,
        all_airframes_match=all(item["matches"] for item in checks))
