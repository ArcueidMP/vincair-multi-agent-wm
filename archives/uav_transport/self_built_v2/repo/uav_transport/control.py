"""Bounded rotor-level feedback and geometric recovery references for transport.

The controller observes physical state and commands individual rotor thrusts.
It never changes body state, cable modes, cable lengths, or attachment points.
All integral/reference memory is explicit so counterfactual rollouts can fork it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class ControllerConfig:
    payload_kp: float = 2.5
    payload_kd: float = 2.5
    payload_ki: float = 0.12
    position_kp: float = 8.0
    position_kd: float = 4.5
    position_ki: float = 0.15
    attitude_frequency: float = 16.0
    attitude_damping: float = 0.85
    max_tilt_deg: float = 50.0
    integral_limit: float = 0.35
    tension_thrust_fraction: float = 0.90
    heading: float = 0.0


def _value(config, name, default):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def rotor_mixer(config):
    """Map individual positive rotor thrusts to body [T, Mx, My, Mz]."""
    radius = float(_value(config, "arm_length", .046))
    ratio = float(_value(config, "yaw_torque_ratio", .006))
    angles = np.deg2rad([45., 135., 225., 315.])
    positions = radius*np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return np.stack((np.ones(4), positions[:, 1], -positions[:, 0],
                     ratio*np.array([1., -1., 1., -1.])))


def mix_bounded(wrench, mixer, maximum):
    """Preserve bounded collective thrust and uniformly desaturate moments."""
    wrench = np.asarray(wrench, dtype=float)
    if wrench.shape != (4,) or not np.isfinite(wrench).all() or maximum <= 0:
        raise ValueError("Expected a finite body wrench and positive rotor limit")
    collective = float(np.clip(wrench[0], 0., 4*maximum))/4
    differential = np.linalg.solve(mixer, np.r_[0., wrench[1:]])
    scale = 1.
    positive, negative = differential > 0, differential < 0
    if positive.any(): scale = min(scale, float(np.min((maximum-collective)/differential[positive])))
    if negative.any(): scale = min(scale, float(np.min(-collective/differential[negative])))
    forces = collective + max(scale, 0.)*differential
    return np.clip(forces, 0., maximum)


def vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def desired_rotation(force_world, heading=0.):
    force = np.asarray(force_world, dtype=float)
    if force.shape != (3,) or not np.isfinite(force).all() or np.linalg.norm(force) < 1e-12:
        raise ValueError("Expected a nonzero finite desired world force")
    b3 = force/np.linalg.norm(force)
    heading_axis = np.array([math.cos(heading), math.sin(heading), 0.])
    b2 = np.cross(b3, heading_axis)
    if np.linalg.norm(b2) < 1e-8:
        heading_axis = np.array([-math.sin(heading), math.cos(heading), 0.])
        b2 = np.cross(b3, heading_axis)
    b2 /= np.linalg.norm(b2)
    return np.column_stack((np.cross(b2, b3), b2, b3))


def allocate_tensions(directions, desired_payload_force, upper_bounds):
    """Nonnegative bounded load sharing with the actual cable geometry.

    A zero upper bound deliberately unloads that cable. The residual records
    unavailable wrench directions when the remaining geometry is insufficient.
    """
    directions = np.asarray(directions, dtype=float)
    desired = np.asarray(desired_payload_force, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if directions.shape != (3, 3) or desired.shape != (3,) or upper.shape != (3,):
        raise ValueError("Expected three cable directions and three tension limits")
    if not all(np.isfinite(x).all() for x in (directions, desired, upper)) or (upper < 0).any():
        raise ValueError("Tension allocation inputs must be finite with nonnegative bounds")
    enabled = upper > 1e-10
    result = np.zeros(3)
    if enabled.any():
        matrix = directions[enabled].T
        unconstrained = np.linalg.lstsq(matrix, desired, rcond=None)[0]
        if (unconstrained >= 0).all() and (unconstrained <= upper[enabled]).all():
            result[enabled] = unconstrained
        else:
            solution = lsq_linear(matrix, desired, bounds=(0., upper[enabled]),
                                  method="bvls", tol=1e-10, max_iter=40)
            if not solution.success:
                raise RuntimeError("Bounded cable tension allocation failed")
            result[enabled] = solution.x
    return result, directions.T@result-desired


def nominal_offsets(config, *, radius=None, yaw=None):
    length = float(_value(config, "cable_length", .6))
    radius = float(_value(config, "formation_radius", .25) if radius is None else radius)
    yaw = float(_value(config, "formation_yaw", 0.) if yaw is None else yaw)
    if not 0 < radius < length:
        raise ValueError("Formation radius must be inside the fixed cable length")
    angles = yaw+np.arange(3)*2*np.pi/3
    return np.column_stack((radius*np.cos(angles), radius*np.sin(angles),
                            np.full(3, np.sqrt(length*length-radius*radius))))


class TransportController:
    """Payload/formation feedback with geometric attitude control and motor caps."""
    def __init__(self, plant_config, config=None):
        self.plant_config = plant_config
        self.config = ControllerConfig() if config is None else config
        self.dt = float(_value(plant_config, "control_dt", .02))
        self.mass = float(_value(plant_config, "drone_mass", .034))
        self.payload_mass = float(_value(plant_config, "payload_mass", .03))
        self.gravity = float(_value(plant_config, "gravity", 9.81))
        self.maximum = float(_value(plant_config, "rotor_max", .16))
        self.inertia = np.asarray(_value(plant_config, "inertia", (1.65717e-5, 1.66556e-5, 2.92617e-5)),dtype=float)
        self.mixer = rotor_mixer(plant_config)
        self.default_offsets = nominal_offsets(plant_config)
        self.reset()

    def reset(self):
        self.payload_integral = np.zeros(3)
        self.position_integral = np.zeros((3, 3))
        self.calls = 0
        self.last_diagnostics = {}

    def capture_memory(self):
        return copy.deepcopy(dict(payload_integral=self.payload_integral,
            position_integral=self.position_integral, calls=self.calls,
            last_diagnostics=self.last_diagnostics))

    def restore_memory(self, memory):
        value = copy.deepcopy(memory)
        if np.asarray(value["payload_integral"]).shape != (3,) or np.asarray(value["position_integral"]).shape != (3,3):
            raise ValueError("Controller memory has an incompatible shape")
        self.payload_integral = np.asarray(value["payload_integral"],dtype=float)
        self.position_integral = np.asarray(value["position_integral"],dtype=float)
        self.calls = int(value["calls"])
        self.last_diagnostics = value["last_diagnostics"]

    def configuration(self):
        return asdict(self.config)

    def command(self, obs, payload_goal, formation_offsets=None, support_weights=None,
                payload_velocity=None, formation_velocity=None, payload_acceleration=None):
        goal = np.asarray(payload_goal, dtype=float)
        velocity = np.zeros(3) if payload_velocity is None else np.asarray(payload_velocity,dtype=float)
        acceleration = np.zeros(3) if payload_acceleration is None else np.asarray(payload_acceleration,dtype=float)
        offsets = self.default_offsets if formation_offsets is None else np.asarray(formation_offsets,dtype=float)
        offset_velocity = np.zeros((3,3)) if formation_velocity is None else np.asarray(formation_velocity,dtype=float)
        weights = np.ones(3) if support_weights is None else np.asarray(support_weights,dtype=float)
        if goal.shape != (3,) or velocity.shape != (3,) or acceleration.shape != (3,) or offsets.shape != (3,3) or offset_velocity.shape != (3,3) or weights.shape != (3,):
            raise ValueError("Invalid payload target, formation, or support-weight shape")
        if not all(np.isfinite(x).all() for x in (goal,velocity,acceleration,offsets,offset_velocity,weights)) or ((weights<0)|(weights>1)).any():
            raise ValueError("References must be finite and support weights in [0,1]")
        p = np.asarray(obs["payload_pos"]); v = np.asarray(obs["payload_vel"])
        qpos = np.asarray(obs["uav_pos"]); qvel = np.asarray(obs["uav_vel"])
        rotations = np.asarray(obs.get("uav_rotation",obs.get("uav_rot")))
        omega = np.asarray(obs.get("uav_omega",obs.get("body_omega")))
        attach_payload = np.asarray(obs.get("payload_attachment_pos",p))
        attach_uav = np.asarray(obs.get("uav_attachment_pos",qpos))
        vectors = attach_uav-attach_payload
        lengths = np.linalg.norm(vectors,axis=1)
        if (lengths<1e-7).any() or not all(np.isfinite(x).all() for x in (p,v,qpos,qvel,rotations,omega)):
            raise ValueError("Invalid or coincident UAV/payload physical state")
        directions = vectors/lengths[:,None]
        cfg=self.config
        error=goal-p
        self.payload_integral=np.clip(self.payload_integral+self.dt*error,-cfg.integral_limit,cfg.integral_limit)
        desired_load_acc=acceleration+cfg.payload_kp*error+cfg.payload_kd*(velocity-v)+cfg.payload_ki*self.payload_integral
        desired_load_force=self.payload_mass*(desired_load_acc+np.array([0.,0.,self.gravity]))
        mg=self.mass*self.gravity
        thrust_cap=4*self.maximum*cfg.tension_thrust_fraction
        upper=-mg*directions[:,2]+np.sqrt(np.maximum(thrust_cap**2-mg**2*(1-directions[:,2]**2),0.))
        tensions,residual=allocate_tensions(directions,desired_load_force,np.maximum(upper,0.)*weights)
        qerror=goal[None]+offsets-qpos
        self.position_integral=np.clip(self.position_integral+self.dt*qerror,-cfg.integral_limit,cfg.integral_limit)
        forces=self.mass*(acceleration[None]+cfg.position_kp*qerror+
            cfg.position_kd*(velocity[None]+offset_velocity-qvel)+cfg.position_ki*self.position_integral)
        forces[:,2]+=mg
        forces+=tensions[:,None]*directions
        commands=np.zeros((3,4)); moment_targets=[]
        for i in range(3):
            force=forces[i].copy()
            force[2]=max(force[2],.05*mg)
            xy=np.linalg.norm(force[:2]); maximum_xy=force[2]*math.tan(math.radians(cfg.max_tilt_deg))
            if xy>maximum_xy:force[:2]*=maximum_xy/xy
            reference=desired_rotation(force,cfg.heading)
            rotation_error=.5*vee(reference.T@rotations[i]-rotations[i].T@reference)
            moment=-self.inertia*cfg.attitude_frequency**2*rotation_error
            moment-=2*cfg.attitude_damping*cfg.attitude_frequency*self.inertia*omega[i]
            moment+=np.cross(omega[i],self.inertia*omega[i])
            thrust=float(force@rotations[i,:,2])
            commands[i]=mix_bounded(np.r_[thrust,moment],self.mixer,self.maximum)
            moment_targets.append(moment)
        self.calls+=1
        self.last_diagnostics=dict(desired_tensions=tensions.copy(),allocation_residual=residual.copy(),
            desired_payload_force=desired_load_force.copy(),desired_uav_forces=forces.copy(),
            desired_moments=np.asarray(moment_targets),rotor_commands=commands.copy(),
            support_weights=weights.copy(),formation_offsets=offsets.copy())
        return commands


@dataclass(frozen=True)
class RecoveryReference:
    offsets: np.ndarray
    offset_velocity: np.ndarray
    support_weights: np.ndarray
    phase: str


@dataclass(frozen=True)
class RecoveryConfig:
    rearrange_seconds: float = 2.0
    settle_seconds: float = .5
    unload_seconds: float = 1.0
    hold_seconds: float = 1.0
    retension_seconds: float = 2.0
    restore_seconds: float = 2.0
    verify_seconds: float = .3
    contact_confirm_seconds: float = .2
    gate_timeout_seconds: float = 4.0
    slack_distance: float = .08
    retension_microarc_deg: float = 5.0
    contact_threshold: float = .001
    support_threshold: float = .02
    gap_tolerance: float = .003
    closing_speed_limit: float = .15
    payload_speed_limit: float = .08
    payload_error_limit: float = .05


def _blend(time, duration):
    if duration <= 0: raise ValueError("A physical maneuver needs positive duration")
    u = float(np.clip(time/duration, 0., 1.))
    value = u**3*(10.-15.*u+6.*u*u)
    derivative = 30*u*u*(1-u)**2/duration if 0 < u < 1 else 0.
    return value, derivative


def smooth_transition(start, end, time, duration):
    """Quintic position reference with zero endpoint velocity/acceleration."""
    value, derivative = _blend(time, duration)
    delta = np.asarray(end)-np.asarray(start)
    return np.asarray(start)+delta*value, delta*derivative


def _formation_on_arcs(config, vehicle, angle, angle_rate, radius_length, radial_rate):
    """Supporting vehicles move on constant cable-length horizontal arcs."""
    nominal = nominal_offsets(config)
    cable = float(_value(config, "cable_length", .6))
    radius = float(_value(config, "formation_radius", .25))
    axis = math.atan2(nominal[vehicle, 1], nominal[vehicle, 0])
    height = math.sqrt(cable*cable-radius*radius)
    offsets, velocities = nominal.copy(), np.zeros((3, 3))
    offsets[vehicle] *= radius_length/cable
    velocities[vehicle] = nominal[vehicle]/cable*radial_rate
    for index, sign in (((vehicle+1)%3, 1.), ((vehicle+2)%3, -1.)):
        theta = axis+sign*angle
        offsets[index] = [radius*math.cos(theta), radius*math.sin(theta), height]
        velocities[index] = radius*sign*angle_rate*np.array([-math.sin(theta), math.cos(theta), 0.])
    return offsets, velocities


class RecoveryProtocol:
    """Physical support transfer with measured re-tension and recovery gates.

    Read ``reference()`` before commanding one control step, then call
    ``update(observation, payload_goal)`` after that step. Timed motion phases
    use smooth geometric references. Contact confirmation and final recovery
    depend on measured tension, distance, velocity and payload progress.
    """
    _motion_durations = dict(rearrange="rearrange_seconds", unload="unload_seconds",
        hold="hold_seconds", retension="retension_seconds", restore="restore_seconds")
    _next = dict(rearrange="settle", settle="unload", unload="hold", hold="retension",
        retension="confirm", confirm="restore", restore="verify", verify="done")

    def __init__(self, plant_config, config=None, *, recovering_uav=0, reconfigure=True):
        self.plant_config = plant_config
        self.config = RecoveryConfig() if config is None else config
        self.dt = float(_value(plant_config, "control_dt", .02))
        self.length = float(_value(plant_config, "cable_length", .6))
        self.vehicle = int(recovering_uav)
        self.reconfigure = bool(reconfigure)
        if self.vehicle not in (0, 1, 2) or not 0 < self.config.slack_distance < self.length/2:
            raise ValueError("Invalid recovery vehicle or slack distance")
        if min(self.config.rearrange_seconds, self.config.unload_seconds,
               self.config.retension_seconds, self.config.restore_seconds, self.config.hold_seconds,
               self.config.settle_seconds, self.config.contact_confirm_seconds,
               self.config.verify_seconds, self.config.gate_timeout_seconds) <= 0:
            raise ValueError("Physical motion phases require positive durations")
        # Quintic radial re-tension has maximum speed 1.875*distance/duration.
        if 1.875*self.config.slack_distance/self.config.retension_seconds > self.config.closing_speed_limit:
            raise ValueError("Declared re-tension reference exceeds its closing-speed bound")
        self.reset()

    def reset(self):
        self.phase = "rearrange"
        self.age = self.elapsed = self.gate_dwell = 0.
        self.done = self.failed = False
        self.failure_reason = None
        self.transitions = []

    def capture_memory(self):
        return copy.deepcopy(dict(phase=self.phase, age=self.age, elapsed=self.elapsed,
            gate_dwell=self.gate_dwell, done=self.done, failed=self.failed,
            failure_reason=self.failure_reason, transitions=self.transitions))

    def restore_memory(self, memory):
        value = copy.deepcopy(memory)
        if value["phase"] not in set(self._next)|{"done"}:
            raise ValueError("Unknown recovery phase")
        for name in ("phase", "age", "elapsed", "gate_dwell", "done", "failed", "failure_reason", "transitions"):
            setattr(self, name, value[name])

    def configuration(self):
        return dict(**asdict(self.config), recovering_uav=self.vehicle, reconfigure=self.reconfigure)

    def reference(self):
        cfg = self.config
        initial_angle = 2*np.pi/3
        support_angle = np.pi/2 if self.reconfigure else initial_angle
        confirmed_angle = support_angle+math.radians(cfg.retension_microarc_deg) if self.reconfigure else initial_angle
        angle, angle_rate = initial_angle, 0.
        radial, radial_rate = self.length, 0.
        weights = np.ones(3)
        phase, age = self.phase, self.age
        if phase == "rearrange":
            blend, rate = _blend(age, cfg.rearrange_seconds)
            angle = initial_angle+(support_angle-initial_angle)*blend
            angle_rate = (support_angle-initial_angle)*rate
        elif phase in ("settle", "unload", "hold"):
            angle = support_angle
            if phase == "unload":
                blend, rate = _blend(age, cfg.unload_seconds)
                radial -= cfg.slack_distance*blend
                radial_rate = -cfg.slack_distance*rate
                weights[self.vehicle] = 1-blend
            elif phase == "hold":
                radial -= cfg.slack_distance
                weights[self.vehicle] = 0.
        elif phase == "retension":
            blend, rate = _blend(age, cfg.retension_seconds)
            radial -= cfg.slack_distance*(1-blend)
            radial_rate = cfg.slack_distance*rate
            weights[self.vehicle] = blend
            # The small physical arc requests a positive donor tension near
            # the end of radial closure, before full triangle restoration.
            micro, micro_rate = _blend(age-.8*cfg.retension_seconds, .2*cfg.retension_seconds)
            angle = support_angle+(confirmed_angle-support_angle)*micro
            angle_rate = (confirmed_angle-support_angle)*micro_rate
        elif phase == "confirm":
            angle = confirmed_angle
        elif phase == "restore":
            blend, rate = _blend(age, cfg.restore_seconds)
            angle = confirmed_angle+(initial_angle-confirmed_angle)*blend
            angle_rate = (initial_angle-confirmed_angle)*rate
        offsets, velocities = _formation_on_arcs(self.plant_config, self.vehicle,
            angle, angle_rate, radial, radial_rate)
        return RecoveryReference(offsets, velocities, weights, phase)

    def _gate(self, obs, payload_goal):
        cfg = self.config
        tensions = np.asarray(obs["cable_tensions"])
        lengths = np.asarray(obs["cable_lengths"])
        speed = float(np.linalg.norm(obs["payload_vel"]))
        if self.phase == "settle":
            support = [i for i in range(3) if i != self.vehicle]
            relative = np.asarray(obs["uav_attachment_pos"])-np.asarray(obs["payload_attachment_pos"])
            target = self.reference().offsets
            return (np.max(np.linalg.norm(relative[support]-target[support], axis=1)) < .04
                and np.all(tensions[support] > cfg.support_threshold) and speed < .15)
        if self.phase == "confirm":
            direction = np.asarray(obs["uav_attachment_pos"])[self.vehicle]-np.asarray(obs["payload_attachment_pos"])
            direction /= np.linalg.norm(direction)
            rate = float((np.asarray(obs["uav_vel"])[self.vehicle]-np.asarray(obs["payload_vel"]))@direction)
            return (tensions[self.vehicle] >= cfg.contact_threshold
                and lengths[self.vehicle] >= self.length-cfg.gap_tolerance
                and abs(rate) <= cfg.closing_speed_limit)
        if self.phase == "verify":
            if payload_goal is None: raise ValueError("Final recovery verification needs the actual payload goal")
            return (np.all(tensions > cfg.support_threshold)
                and np.all(lengths >= self.length-cfg.gap_tolerance)
                and speed <= cfg.payload_speed_limit
                and np.linalg.norm(np.asarray(obs["payload_pos"])-payload_goal) <= cfg.payload_error_limit)
        return False

    def update(self, obs, payload_goal=None):
        if self.done or self.failed: return
        self.age += self.dt; self.elapsed += self.dt
        phase = self.phase
        if phase in self._motion_durations:
            advance = self.age+1e-10 >= getattr(self.config, self._motion_durations[phase])
        else:
            good = self._gate(obs, payload_goal)
            self.gate_dwell = self.gate_dwell+self.dt if good else 0.
            minimum = {"settle":self.config.settle_seconds,
                "confirm":self.config.contact_confirm_seconds,"verify":self.config.verify_seconds}[phase]
            advance = self.gate_dwell+1e-10 >= minimum
            if not advance and self.age >= minimum+self.config.gate_timeout_seconds:
                self.failed = True; self.failure_reason = phase+"_gate_timeout"
                return
        if advance:
            self.phase = self._next[phase]
            self.transitions.append(dict(phase=phase, completed_at=self.elapsed))
            self.age = self.gate_dwell = 0.
            self.done = self.phase == "done"


def recovery_reference(config, time, *, recovering_uav=0, transition_seconds=1.,
                       unloaded_seconds=1., retension_seconds=2., slack_distance=.08,
                       reconfigure=True):
    """Nominal timed references for plotting; runtime recovery uses its gates.

    This convenience function traverses the prescribed minimum phase durations.
    Use RecoveryProtocol for execution so measured contact and completion gates
    own the phase transitions.
    """
    cfg = RecoveryConfig(unload_seconds=transition_seconds, hold_seconds=unloaded_seconds,
                         retension_seconds=retension_seconds, slack_distance=slack_distance)
    protocol = RecoveryProtocol(config,cfg,recovering_uav=recovering_uav,reconfigure=reconfigure)
    schedule = [("rearrange",cfg.rearrange_seconds),("settle",cfg.settle_seconds),
        ("unload",cfg.unload_seconds),("hold",cfg.hold_seconds),
        ("retension",cfg.retension_seconds),("confirm",cfg.contact_confirm_seconds),
        ("restore",cfg.restore_seconds),("verify",cfg.verify_seconds)]
    remaining = max(float(time),0.)
    for phase,duration in schedule:
        if remaining < duration:
            protocol.phase,protocol.age = phase,remaining
            return protocol.reference()
        remaining -= duration
    protocol.phase="done"
    return protocol.reference()
