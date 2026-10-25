"""Three rigid quadrotors carrying a central-attachment payload in MuJoCo.

Each airframe has six free degrees of freedom and four positive, bounded rotor
forces with reaction torques and first-order motor dynamics. Spatial tendon
upper limits represent finite-compliance, tension-only fixed-length cables.
The spherical payload has a central attachment, so cable forces do not control
its orientation. No ideal world-frame force controller is hidden in the plant.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TransportConfig:
    dt: float = .002
    control_dt: float = .02
    drone_mass: float = .034
    payload_mass: float = .03
    rotor_max: float = .16
    motor_tau: float = .03
    cable_length: float = .6
    formation_radius: float = .25
    payload_height: float = 1.2
    payload_xy: tuple = (0., 0.)
    formation_yaw: float = 0.
    gravity: float = 9.81
    inertia: tuple = (1.65717e-5, 1.66556e-5, 2.92617e-5)
    arm_length: float = .046
    yaw_torque_ratio: float = .006
    cable_solref: tuple = (.01, 1.)
    cable_solimp: tuple = (.95, .99, .001)
    solver_iterations: int = 50
    solver_tolerance: float = 1e-10
    initial_velocity: tuple = (0., 0., 0.)
    initial_uav_velocity_offsets: tuple = ((0., 0., 0.),) * 3
    initial_uav_omegas: tuple = ((0., 0., 0.),) * 3

    def __post_init__(self):
        positive = (self.dt, self.control_dt, self.drone_mass, self.payload_mass,
                    self.rotor_max, self.motor_tau, self.cable_length,
                    self.payload_height, self.arm_length, self.gravity)
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Time, mass, force, geometry and gravity must be finite and positive")
        if not 0 < self.formation_radius < self.cable_length:
            raise ValueError("Formation radius must be strictly between zero and cable length")
        ratio = self.control_dt / self.dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-10) or round(ratio) < 1:
            raise ValueError("control_dt must be an integer multiple of dt")
        if min(self.inertia) <= 0 or self.solver_iterations < 1:
            raise ValueError("Positive inertia and solver iterations are required")
        if self.cable_solref[0] < 2*self.dt:
            raise ValueError("Keep the cable time constant at least two physics timesteps")


def rotor_geometry(config):
    angles = np.arange(4)*np.pi/2 + np.pi/4
    positions = np.column_stack((config.arm_length*np.cos(angles),
                                  config.arm_length*np.sin(angles), np.zeros(4)))
    signs = np.array([1., -1., 1., -1.])
    mixer = np.vstack((np.ones(4), positions[:, 1], -positions[:, 0],
                       config.yaw_torque_ratio*signs))
    return positions, mixer


def formation_offsets(config):
    angles = config.formation_yaw + np.arange(3)*2*np.pi/3
    height = math.sqrt(config.cable_length**2-config.formation_radius**2)
    return np.column_stack((config.formation_radius*np.cos(angles),
        config.formation_radius*np.sin(angles), np.full(3, height)))


def force_rotation(force, heading=0.):
    """Body-to-world rotation whose local z axis follows the required force."""
    z = np.asarray(force, dtype=float)
    z = z / np.linalg.norm(z)
    heading_axis = np.array([math.cos(heading), math.sin(heading), 0.])
    y = np.cross(z, heading_axis)
    if np.linalg.norm(y) < 1e-9:
        heading_axis = np.array([0., 1., 0.])
        y = np.cross(z, heading_axis)
    y /= np.linalg.norm(y)
    return np.column_stack((np.cross(y, z), y, z))


def _numbers(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).reshape(-1))


def initial_hover(config):
    offsets = formation_offsets(config)
    directions = offsets/config.cable_length
    tension = config.payload_mass*config.gravity/(3*directions[0, 2])
    forces = tension*directions
    forces[:, 2] += config.drone_mass*config.gravity
    thrusts = np.repeat(np.linalg.norm(forces, axis=1)[:, None]/4, 4, axis=1)
    if np.any(thrusts > config.rotor_max):
        raise ValueError("The initial formation exceeds available rotor thrust")
    rotations = np.stack([force_rotation(f) for f in forces])
    return thrusts, rotations


def make_xml(config):
    root = ET.Element("mujoco", model="three_uav_central_payload")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    option = ET.SubElement(root, "option", timestep=str(config.dt),
        gravity=f"0 0 {-config.gravity}", integrator="implicitfast", solver="Newton",
        jacobian="dense", iterations=str(config.solver_iterations),
        tolerance=str(config.solver_tolerance))
    ET.SubElement(option, "flag", energy="enable", autoreset="disable")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="5 5 .05",
                  rgba=".2 .22 .25 1", friction=".7 .005 .0001")
    origin = np.array([*config.payload_xy, config.payload_height])
    payload = ET.SubElement(world, "body", name="payload", pos=_numbers(origin))
    ET.SubElement(payload, "freejoint", name="payload_joint")
    ET.SubElement(payload, "geom", name="payload_geom", type="sphere", size=".025",
                  mass=str(config.payload_mass), rgba=".9 .5 .15 1")
    ET.SubElement(payload, "site", name="payload_attachment", pos="0 0 0", size=".004")
    positions, mixer = rotor_geometry(config)
    _, rotations = initial_hover(config)
    colors = (".2 .5 .9 1", ".2 .8 .4 1", ".8 .25 .3 1")
    for i, offset in enumerate(formation_offsets(config)):
        q = Rotation.from_matrix(rotations[i]).as_quat()
        body = ET.SubElement(world, "body", name=f"uav{i}", pos=_numbers(origin+offset),
            quat=_numbers(q[[3, 0, 1, 2]]))
        ET.SubElement(body, "freejoint", name=f"uav{i}_joint")
        ET.SubElement(body, "inertial", pos="0 0 0", mass=str(config.drone_mass),
                      diaginertia=_numbers(config.inertia))
        ET.SubElement(body, "geom", name=f"uav{i}_geom", type="ellipsoid",
                      size=".045 .045 .012", rgba=colors[i])
        ET.SubElement(body, "site", name=f"uav{i}_attachment", pos="0 0 0", size=".003")
        for j, position in enumerate(positions):
            ET.SubElement(body, "site", name=f"uav{i}_rotor{j}", pos=_numbers(position),
                          type="cylinder", size=".014 .001", rgba=".2 .2 .2 .8")
    tendons = ET.SubElement(root, "tendon")
    for i in range(3):
        cable = ET.SubElement(tendons, "spatial", name=f"cable{i}", limited="true",
            range=f"0 {config.cable_length}", width=".002", stiffness="0", damping="0",
            solreflimit=_numbers(config.cable_solref), solimplimit=_numbers(config.cable_solimp))
        ET.SubElement(cable, "site", site="payload_attachment")
        ET.SubElement(cable, "site", site=f"uav{i}_attachment")
    actuators = ET.SubElement(root, "actuator")
    for i in range(3):
        for j in range(4):
            ET.SubElement(actuators, "general", name=f"uav{i}_motor{j}", site=f"uav{i}_rotor{j}",
                gear=f"0 0 1 0 0 {mixer[3,j]}", dyntype="filterexact", dynprm=str(config.motor_tau),
                actearly="true", actlimited="true", actrange=f"0 {config.rotor_max}",
                ctrllimited="true", ctrlrange=f"0 {config.rotor_max}",
                forcelimited="true", forcerange=f"0 {config.rotor_max}")
    return ET.tostring(root, encoding="unicode")


class TransportEnv:
    def __init__(self, config=None):
        self.config = TransportConfig() if config is None else config
        self.xml = make_xml(self.config)
        self.model_hash = hashlib.sha256(self.xml.encode()).hexdigest()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.substeps = round(self.config.control_dt/self.config.dt)
        names = ["payload"]+[f"uav{i}" for i in range(3)]
        self.body_ids = np.array([self.model.body(name).id for name in names])
        self.payload_body_id = int(self.body_ids[0])
        self.uav_body_ids = self.body_ids[1:]
        joints = [self.model.joint(name+"_joint") for name in names]
        self.qpos_addresses = np.array([int(j.qposadr[0]) for j in joints])
        self.dof_addresses = np.array([int(j.dofadr[0]) for j in joints])
        self.tendon_ids = np.array([self.model.tendon(f"cable{i}").id for i in range(3)])
        self.rotor_site_ids = np.array([[self.model.site(f"uav{i}_rotor{j}").id for j in range(4)] for i in range(3)])
        self.actuator_ids = np.array([[self.model.actuator(f"uav{i}_motor{j}").id for j in range(4)] for i in range(3)])
        self.rotor_positions, self.mixer = rotor_geometry(self.config)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        velocity = np.asarray(self.config.initial_velocity, dtype=float)
        offsets = np.asarray(self.config.initial_uav_velocity_offsets, dtype=float)
        omegas = np.asarray(self.config.initial_uav_omegas, dtype=float)
        if velocity.shape != (3,) or offsets.shape != (3, 3) or omegas.shape != (3, 3):
            raise ValueError("Initial velocities must have shapes 3, 3x3 and 3x3")
        self.data.qvel[self.dof_addresses[0]:self.dof_addresses[0]+3] = velocity
        for i, adr in enumerate(self.dof_addresses[1:]):
            self.data.qvel[adr:adr+3] = velocity+offsets[i]
            self.data.qvel[adr+3:adr+6] = omegas[i]
        hover, _ = initial_hover(self.config)
        self.data.ctrl[:] = hover.reshape(-1)
        self.data.act[:] = hover.reshape(-1)
        self.control_steps = 0
        self.physics_steps = 0
        self.last_command = hover.copy()
        self.last_requested_command = hover.copy()
        self.last_external_forces = np.zeros((4, 3))
        self.metrics = dict(rotor_force_integral=np.zeros((3,4)),
            squared_rotor_force_integral=np.zeros((3,4)), signed_actuator_work=np.zeros((3,4)),
            absolute_actuator_work=np.zeros((3,4)), uav_path_length=np.zeros(3),
            saturation_time=np.zeros(3), clamp_intervention_time=np.zeros(3),
            command_saturation_time=np.zeros(3), command_force_integral=np.zeros((3,4)),
            rotor_upper_bound_time=np.zeros((3,4)),rotor_lower_bound_time=np.zeros((3,4)),
            maximum_rotor_force=np.zeros((3,4)), maximum_uav_tilt=np.zeros(3),
            minimum_body_height=self._positions()[:,2].copy(),maximum_cable_tension=np.zeros(3),
            contact_steps=0,floor_contact_steps=0,uav_collision_steps=0,payload_uav_collision_steps=0,
            maximum_cable_extension=0.)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def cable_tensions(self):
        tensions = np.zeros(3)
        active = np.flatnonzero(self.data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON))
        for constraint in active:
            tendon = int(self.data.efc_id[constraint])
            if tendon in self.tendon_ids:
                tensions[int(np.flatnonzero(self.tendon_ids == tendon)[0])] += max(0., float(self.data.efc_force[constraint]))
        return tensions

    def _positions(self):
        return np.stack([self.data.qpos[a:a+3] for a in self.qpos_addresses])

    def step(self, rotor_commands, external_forces=None):
        requested = np.asarray(rotor_commands, dtype=float)
        forces = np.zeros((4,3)) if external_forces is None else np.asarray(external_forces, dtype=float)
        if requested.shape != (3,4) or forces.shape != (4,3):
            raise ValueError("Expected 3x4 rotor commands and 4x3 external world-frame forces")
        if not np.isfinite(requested).all() or not np.isfinite(forces).all():
            raise ValueError("Commands and external forces must be finite")
        clipped = np.clip(requested, 0., self.config.rotor_max)
        self.last_requested_command = requested.copy()
        self.last_command = clipped.copy()
        self.last_external_forces = forces.copy()
        self.data.ctrl[:] = clipped.reshape(-1)
        self.data.xfrc_applied[:] = 0.
        self.data.xfrc_applied[self.body_ids, :3] = forces
        self.metrics["clamp_intervention_time"] += np.any(requested != clipped, axis=1)*self.config.control_dt
        self.metrics["command_saturation_time"] += np.any(clipped >= .99*self.config.rotor_max,axis=1)*self.config.control_dt
        self.metrics["command_force_integral"] += clipped*self.config.control_dt
        for _ in range(self.substeps):
            before = self._positions()[1:]
            mujoco.mj_step(self.model, self.data)
            actual = self.data.actuator_force[self.actuator_ids].copy()
            power = actual*self.data.actuator_velocity[self.actuator_ids]
            self.metrics["rotor_force_integral"] += actual*self.config.dt
            self.metrics["squared_rotor_force_integral"] += actual**2*self.config.dt
            self.metrics["signed_actuator_work"] += power*self.config.dt
            self.metrics["absolute_actuator_work"] += np.abs(power)*self.config.dt
            self.metrics["uav_path_length"] += np.linalg.norm(self._positions()[1:]-before, axis=1)
            upper,lower=actual >= .99*self.config.rotor_max,actual <= .01*self.config.rotor_max
            self.metrics["rotor_upper_bound_time"] += upper*self.config.dt
            self.metrics["rotor_lower_bound_time"] += lower*self.config.dt
            self.metrics["saturation_time"] += upper.any(1)*self.config.dt
            self.metrics["maximum_rotor_force"] = np.maximum(self.metrics["maximum_rotor_force"],actual)
            quaternions=np.stack([self.data.qpos[a+3:a+7] for a in self.qpos_addresses[1:]])
            tilt=np.arccos(np.clip(1-2*(quaternions[:,1]**2+quaternions[:,2]**2),-1.,1.))
            self.metrics["maximum_uav_tilt"] = np.maximum(self.metrics["maximum_uav_tilt"],tilt)
            self.metrics["minimum_body_height"] = np.minimum(self.metrics["minimum_body_height"],self._positions()[:,2])
            self.metrics["maximum_cable_tension"] = np.maximum(self.metrics["maximum_cable_tension"], self.cable_tensions())
            self.metrics["maximum_cable_extension"] = max(self.metrics["maximum_cable_extension"],
                float(np.maximum(self.data.ten_length-self.config.cable_length,0.).max()))
            pairs=self.collision_pairs()
            self.metrics["contact_steps"] += int(bool(pairs))
            self.metrics["floor_contact_steps"] += int(any(0 in pair for pair in pairs))
            self.metrics["uav_collision_steps"] += int(any(a in self.uav_body_ids and b in self.uav_body_ids for a,b in pairs))
            self.metrics["payload_uav_collision_steps"] += int(any(self.payload_body_id in pair
                and any(x in self.uav_body_ids for x in pair) for pair in pairs))
            self.physics_steps += 1
        self.control_steps += 1
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def collision_pairs(self):
        return [(int(self.model.geom_bodyid[c.geom1]),int(self.model.geom_bodyid[c.geom2]))
                for c in self.data.contact]

    def observe(self):
        positions = self.data.xpos[self.body_ids].copy()
        rotations = self.data.xmat[self.uav_body_ids].reshape(3,3,3).copy()
        velocities, omegas = [], []
        for body in self.body_ids:
            world = np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,int(body),world,0)
            velocities.append(world[3:])
            local = np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,int(body),local,1)
            omegas.append(local[:3])
        velocities, omegas = np.asarray(velocities), np.asarray(omegas)
        lengths = self.data.ten_length[self.tendon_ids].copy()
        directions = positions[1:]-positions[0]
        directions /= np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
        tensions = self.cable_tensions()
        contacts = self.collision_pairs()
        state = np.concatenate((self.data.qpos,self.data.qvel,self.data.act))
        rotor_forces = self.data.actuator_force[self.actuator_ids].copy()
        return dict(time=float(self.data.time), control_steps=self.control_steps,
            payload_pos=positions[0],payload_vel=velocities[0],payload_quat=self.data.xquat[self.payload_body_id].copy(),
            uav_pos=positions[1:],uav_vel=velocities[1:],uav_rotation=rotations,uav_omega=omegas[1:],
            payload_attachment_pos=positions[0].copy(),uav_attachment_pos=positions[1:].copy(),
            cable_unit_vectors=directions,cable_lengths=lengths,cable_gaps=self.config.cable_length-lengths,
            cable_tensions=tensions,cable_taut=tensions>1e-5,
            rotor_forces=rotor_forces,motor_activations=self.data.act.reshape(3,4).copy(),
            commanded_rotor_forces=self.last_command.copy(),requested_rotor_forces=self.last_requested_command.copy(),
            external_forces=self.last_external_forces.copy(),collision_pairs=contacts,
            floor_contact=any(0 in pair for pair in contacts),
            uav_collision=any(a in self.uav_body_ids and b in self.uav_body_ids for a,b in contacts),
            payload_uav_collision=any(self.payload_body_id in pair
                and any(x in self.uav_body_ids for x in pair) for pair in contacts),
            warnings=self.data.warning.number.copy(),finite=bool(np.isfinite(state).all()),
            uav_tilt=np.arccos(np.clip(rotations[:,2,2],-1.,1.)),
            cable_swing=np.arccos(np.clip(directions[:,2],-1.,1.)))

    def clone(self):
        from .snapshot import capture, restore
        other = TransportEnv(self.config)
        restore(other, capture(self))
        return other

    def specification(self):
        return dict(config=asdict(self.config),xml_sha256=self.model_hash,mujoco_version=mujoco.__version__,
            actuator_units="newtons per rotor", work="substep integral of actuator force times transmission velocity",
            saturation_time="time with any actual rotor force at least 99 percent of its upper bound",
            cable_model="finite-compliance unilateral spatial-tendon upper limits",
            payload_model="spherical rigid body with one common central cable attachment")
