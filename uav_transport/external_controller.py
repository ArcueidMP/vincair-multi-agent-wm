"""Pinned upstream Lee position/attitude controller with shared load allocation.

The flight controller executes the unchanged C source from IMRCLab's firmware.
The study's payload-force allocator, formation references, bounded rotor mixer
and recovery protocol are shared with the study controller. This adapter is an
external position/attitude baseline, not the upstream Lee payload/QP controller.
"""
from __future__ import annotations

import copy
import ctypes
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from .control import ControllerConfig, _value, allocate_tensions, mix_bounded, nominal_offsets, rotor_mixer


UPSTREAM_COMMIT = "b08d9f3cbcc581d8e7f988cabad22c26d35f4a58"
VENDOR = Path(__file__).resolve().parents[1]/"third_party/crazyflie_firmware_lee"
_WRAPPER = r'''
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "controller_lee.h"

/* The upstream C file and headers remain unchanged. Configure its scheduler
 * for the simulation's sampled control period before including its body. */
#undef ATTITUDE_RATE
#define ATTITUDE_RATE ADAPTER_CONTROL_HZ
#include "controller_lee.c"

size_t adapter_state_size(void) { return sizeof(controllerLee_t); }
float adapter_control_dt(void) { return 1.0f / ADAPTER_CONTROL_HZ; }

void adapter_init(void *memory, float mass, const float *inertia) {
    controllerLee_t *self = (controllerLee_t *)memory;
    memset(self, 0, sizeof(*self));
    controllerLeeInit(self);
    self->mass = mass;
    self->J = mkvec(inertia[0], inertia[1], inertia[2]);
}

void adapter_step(void *memory, const float *position, const float *velocity,
    const float *quaternion_xyzw, const float *omega_rad,
    const float *goal, const float *goal_velocity, const float *acceleration,
    float yaw_rad, float yaw_rate_rad, const float *jerk, float *output) {
    controllerLee_t *self = (controllerLee_t *)memory;
    control_t control;
    setpoint_t setpoint;
    sensorData_t sensors;
    state_t state;
    memset(&control, 0, sizeof(control));
    memset(&setpoint, 0, sizeof(setpoint));
    memset(&sensors, 0, sizeof(sensors));
    memset(&state, 0, sizeof(state));
    state.position.x=position[0]; state.position.y=position[1]; state.position.z=position[2];
    state.velocity.x=velocity[0]; state.velocity.y=velocity[1]; state.velocity.z=velocity[2];
    state.attitudeQuaternion.x=quaternion_xyzw[0];
    state.attitudeQuaternion.y=quaternion_xyzw[1];
    state.attitudeQuaternion.z=quaternion_xyzw[2];
    state.attitudeQuaternion.w=quaternion_xyzw[3];
    sensors.gyro.x=degrees(omega_rad[0]);
    sensors.gyro.y=degrees(omega_rad[1]);
    sensors.gyro.z=degrees(omega_rad[2]);
    setpoint.mode.x=modeAbs; setpoint.mode.y=modeAbs; setpoint.mode.z=modeAbs;
    setpoint.mode.yaw=modeAbs;
    setpoint.position.x=goal[0]; setpoint.position.y=goal[1]; setpoint.position.z=goal[2];
    setpoint.velocity.x=goal_velocity[0]; setpoint.velocity.y=goal_velocity[1]; setpoint.velocity.z=goal_velocity[2];
    setpoint.acceleration.x=acceleration[0]; setpoint.acceleration.y=acceleration[1]; setpoint.acceleration.z=acceleration[2];
    setpoint.attitude.yaw=degrees(yaw_rad); setpoint.attitudeRate.yaw=degrees(yaw_rate_rad);
    setpoint.jerk.x=jerk[0]; setpoint.jerk.y=jerk[1]; setpoint.jerk.z=jerk[2];
    controllerLee(self, &control, &setpoint, &sensors, &state, 0);
    output[0]=control.thrustSI;
    output[1]=control.torque[0]; output[2]=control.torque[1]; output[3]=control.torque[2];
}

void adapter_integrals(const void *memory, float *output) {
    const controllerLee_t *self=(const controllerLee_t *)memory;
    output[0]=self->i_error_pos.x; output[1]=self->i_error_pos.y; output[2]=self->i_error_pos.z;
    output[3]=self->i_error_att.x; output[4]=self->i_error_att.y; output[5]=self->i_error_att.z;
}

void adapter_parameters(const void *memory, float *output) {
    const controllerLee_t *self=(const controllerLee_t *)memory;
    const struct vec values[7]={self->J,self->Kpos_P,self->Kpos_D,self->Kpos_I,self->KR,self->Komega,self->KI};
    output[0]=self->mass;
    for(int i=0;i<7;++i) { output[1+3*i]=values[i].x; output[2+3*i]=values[i].y; output[3+3*i]=values[i].z; }
    output[22]=adapter_control_dt();
}
'''


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_upstream():
    meta = json.loads((VENDOR/"UPSTREAM.json").read_text())
    if meta["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Unexpected upstream controller revision")
    for entry in meta["files"]:
        if _sha(VENDOR/entry["path"]) != entry["sha256"]:
            raise AssertionError("Pinned upstream controller source changed: "+entry["path"])
    return meta


@lru_cache(maxsize=8)
def _native_library(rate_hz):
    if rate_hz not in (25, 50, 100, 250, 500, 1000):
        raise ValueError("Use a declared firmware scheduler rate")
    upstream = verify_upstream()
    compiler = shutil.which("cc")
    if compiler is None: raise RuntimeError("Building the pinned upstream C controller requires a C compiler")
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    key = hashlib.sha256(json.dumps(dict(commit=UPSTREAM_COMMIT, files=upstream["files"],
        wrapper=_WRAPPER, rate_hz=rate_hz, compiler=version), sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get("UAV_EXTERNAL_CONTROLLER_CACHE",
        str(Path(__file__).resolve().parents[1]/"data/uav_transport/external_controller_build")))/key
    cache.mkdir(parents=True, exist_ok=True)
    source, library = cache/"adapter.c", cache/"controller_lee.so"
    with (cache/"build.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not library.exists():
            source.write_text(_WRAPPER)
            temporary = cache/f"controller_lee.{os.getpid()}.so"
            command = [compiler, "-shared", "-fPIC", "-O2", "-std=c99",
                f"-DADAPTER_CONTROL_HZ={rate_hz}", "-I", str(VENDOR), str(source), "-lm", "-o", str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("Pinned upstream controller compilation failed:\n"+result.stderr)
            (cache/"build.json").write_text(json.dumps(dict(build_key=key, compiler=version,
                command=command, source_sha256=_sha(source), library_sha256=_sha(temporary),
                upstream_commit=UPSTREAM_COMMIT, control_rate_hz=rate_hz,
                upstream_default_attitude_rate_hz=500), indent=2)+"\n")
            os.replace(temporary, library)
    build = json.loads((cache/"build.json").read_text())
    if _sha(library) != build["library_sha256"] or _sha(source) != build["source_sha256"]:
        raise AssertionError("Cached external controller build changed")
    lib = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    lib.adapter_state_size.argtypes=[]; lib.adapter_state_size.restype=ctypes.c_size_t
    lib.adapter_control_dt.argtypes=[]; lib.adapter_control_dt.restype=ctypes.c_float
    lib.adapter_init.argtypes=[ctypes.c_void_p, ctypes.c_float, pointer]; lib.adapter_init.restype=None
    lib.adapter_step.argtypes=[ctypes.c_void_p]+[pointer]*7+[ctypes.c_float,ctypes.c_float,pointer,pointer]
    lib.adapter_step.restype=None
    lib.adapter_integrals.argtypes=[ctypes.c_void_p,pointer]; lib.adapter_integrals.restype=None
    lib.adapter_parameters.argtypes=[ctypes.c_void_p,pointer]; lib.adapter_parameters.restype=None
    return lib, build


class NativeLee:
    """One independent upstream controllerLee_t with complete byte snapshots."""
    def __init__(self, mass, inertia, dt):
        inertia=np.asarray(inertia,dtype=np.float32)
        if inertia.shape!=(3,) or not np.isfinite(inertia).all() or (inertia<=0).any() or not np.isfinite(mass) or mass<=0 or not np.isfinite(dt) or dt<=0:
            raise ValueError("Native controller requires positive physical mass/inertia and control period")
        rate = round(1./float(dt))
        if not np.isclose(rate*dt, 1., atol=1e-12, rtol=0):
            raise ValueError("Control period must correspond to an integer firmware rate")
        self.lib, self.build = _native_library(rate)
        self.memory = ctypes.create_string_buffer(self.lib.adapter_state_size())
        self.lib.adapter_init(self.memory, float(mass), inertia)

    def capture(self):
        return np.frombuffer(self.memory.raw, dtype=np.uint8).copy()

    def restore(self, state):
        state = np.asarray(state, dtype=np.uint8)
        if state.shape != (len(self.memory),): raise ValueError("Upstream controller state has the wrong size")
        ctypes.memmove(self.memory, state.tobytes(), len(self.memory))

    def integrals(self):
        values = np.empty(6, dtype=np.float32)
        self.lib.adapter_integrals(self.memory, values)
        return values.reshape(2, 3).astype(float)

    def parameters(self):
        values = np.empty(23, dtype=np.float32)
        self.lib.adapter_parameters(self.memory, values)
        return dict(mass=float(values[0]), **{key:values[1+3*i:4+3*i].tolist()
            for i,key in enumerate(("inertia", "Kpos_P", "Kpos_D", "Kpos_I", "KR", "Komega", "KI"))},
            control_dt=float(values[22]))

    def step(self, position, velocity, rotation, omega, goal, goal_velocity, acceleration, heading=0.):
        quaternion = Rotation.from_matrix(rotation).as_quat()
        values = [np.ascontiguousarray(x, dtype=np.float32) for x in
            (position, velocity, quaternion, omega, goal, goal_velocity, acceleration)]
        if any(x.shape!=(size,) or not np.isfinite(x).all() for x,size in zip(values,(3,3,4,3,3,3,3))):
            raise ValueError("Native controller inputs must have the expected finite vector shapes")
        output = np.empty(4, dtype=np.float32)
        self.lib.adapter_step(self.memory, *values, float(heading), 0., np.zeros(3,dtype=np.float32), output)
        if not np.isfinite(output).all(): raise FloatingPointError("Upstream controller produced a nonfinite wrench")
        return output.astype(float)


class ExternalLeeController:
    """Shared payload allocator plus authentic upstream Lee flight feedback."""
    family = "upstream_lee_position_attitude_shared_payload_allocator"

    def __init__(self, plant_config, config=None):
        self.plant_config = plant_config
        self.config = ControllerConfig() if config is None else config
        self.dt = float(_value(plant_config,"control_dt",.02))
        self.mass = float(_value(plant_config,"drone_mass",.034))
        self.payload_mass = float(_value(plant_config,"payload_mass",.03))
        self.gravity = float(_value(plant_config,"gravity",9.81))
        if self.gravity != 9.81: raise ValueError("The pinned upstream controller uses gravity 9.81 m/s^2")
        self.maximum = float(_value(plant_config,"rotor_max",.16))
        self.inertia = np.asarray(_value(plant_config,"inertia",(1.65717e-5,1.66556e-5,2.92617e-5)))
        self.mixer = rotor_mixer(plant_config)
        self.default_offsets = nominal_offsets(plant_config)
        self.reset()

    def reset(self):
        self.payload_integral = np.zeros(3)
        self.position_integral = np.zeros((3,3))
        self.controllers = [NativeLee(self.mass,self.inertia,self.dt) for _ in range(3)]
        self.calls=0; self.last_diagnostics={}

    def configuration(self):
        return dict(family=self.family, upstream_commit=UPSTREAM_COMMIT,
            upstream_controller="controller_lee.c", upstream_parameters=self.controllers[0].parameters(),
            upstream_default_attitude_rate_hz=500, executed_control_rate_hz=round(1/self.dt),
            shared_allocator={key:getattr(self.config,key) for key in
                ("payload_kp","payload_kd","payload_ki","integral_limit","tension_thrust_fraction","heading")},
            roles="Study payload/tension allocation, formation and recovery references; upstream C position/attitude feedback; shared bounded motor mixing",
            source_sha256={entry["path"]:entry["sha256"] for entry in verify_upstream()["files"]},
            native_build_key=self.controllers[0].build["build_key"])

    def capture_memory(self):
        return copy.deepcopy(dict(payload_integral=self.payload_integral, position_integral=self.position_integral,
            native_states=[controller.capture() for controller in self.controllers],
            native_build_key=self.controllers[0].build["build_key"],calls=self.calls,last_diagnostics=self.last_diagnostics))

    def restore_memory(self, memory):
        value = copy.deepcopy(memory)
        if value["native_build_key"] != self.controllers[0].build["build_key"] or len(value["native_states"]) != 3:
            raise ValueError("Upstream controller snapshot uses a different native build")
        self.payload_integral=np.asarray(value["payload_integral"],dtype=float)
        self.position_integral=np.asarray(value["position_integral"],dtype=float)
        if self.payload_integral.shape!=(3,) or self.position_integral.shape!=(3,3):
            raise ValueError("Invalid shared-controller memory shapes")
        for controller,state in zip(self.controllers,value["native_states"]): controller.restore(state)
        self.calls=int(value["calls"]);self.last_diagnostics=value["last_diagnostics"]

    def command(self, obs, payload_goal, formation_offsets=None, support_weights=None,
                payload_velocity=None, formation_velocity=None, payload_acceleration=None):
        goal=np.asarray(payload_goal,dtype=float)
        velocity=np.zeros(3) if payload_velocity is None else np.asarray(payload_velocity,dtype=float)
        acceleration=np.zeros(3) if payload_acceleration is None else np.asarray(payload_acceleration,dtype=float)
        offsets=self.default_offsets if formation_offsets is None else np.asarray(formation_offsets,dtype=float)
        offset_velocity=np.zeros((3,3)) if formation_velocity is None else np.asarray(formation_velocity,dtype=float)
        weights=np.ones(3) if support_weights is None else np.asarray(support_weights,dtype=float)
        expected=((goal,(3,)),(velocity,(3,)),(acceleration,(3,)),(offsets,(3,3)),(offset_velocity,(3,3)),(weights,(3,)))
        if any(x.shape!=shape or not np.isfinite(x).all() for x,shape in expected) or ((weights<0)|(weights>1)).any():
            raise ValueError("Invalid finite transport references or support weights")
        p,v=np.asarray(obs["payload_pos"]),np.asarray(obs["payload_vel"])
        qpos,qvel=np.asarray(obs["uav_pos"]),np.asarray(obs["uav_vel"])
        rotation,omega=np.asarray(obs["uav_rotation"]),np.asarray(obs["uav_omega"])
        vectors=np.asarray(obs.get("uav_attachment_pos",qpos))-np.asarray(obs.get("payload_attachment_pos",p))
        lengths=np.linalg.norm(vectors,axis=1)
        if (lengths<1e-7).any(): raise ValueError("Coincident cable endpoints")
        directions=vectors/lengths[:,None];cfg=self.config
        error=goal-p
        self.payload_integral=np.clip(self.payload_integral+self.dt*error,-cfg.integral_limit,cfg.integral_limit)
        desired_acceleration=acceleration+cfg.payload_kp*error+cfg.payload_kd*(velocity-v)+cfg.payload_ki*self.payload_integral
        payload_force=self.payload_mass*(desired_acceleration+np.array([0.,0.,self.gravity]))
        mg=self.mass*self.gravity;cap=4*self.maximum*cfg.tension_thrust_fraction
        upper=-mg*directions[:,2]+np.sqrt(np.maximum(cap**2-mg**2*(1-directions[:,2]**2),0.))
        tension,residual=allocate_tensions(directions,payload_force,np.maximum(upper,0.)*weights)
        commands=[];wrenches=[]
        for i,controller in enumerate(self.controllers):
            # Carry the shared allocated cable load through the upstream
            # controller's declared acceleration feedforward input.
            feedforward=acceleration+tension[i]*directions[i]/self.mass
            wrench=controller.step(qpos[i],qvel[i],rotation[i],omega[i],goal+offsets[i],
                velocity+offset_velocity[i],feedforward,cfg.heading)
            wrenches.append(wrench);commands.append(mix_bounded(wrench,self.mixer,self.maximum))
        self.position_integral=np.stack([c.integrals()[0] for c in self.controllers])
        self.calls+=1
        self.last_diagnostics=dict(desired_tensions=tension.copy(),allocation_residual=residual.copy(),
            desired_payload_force=payload_force.copy(),upstream_wrenches=np.asarray(wrenches),
            rotor_commands=np.asarray(commands),support_weights=weights.copy(),formation_offsets=offsets.copy())
        return np.asarray(commands)
