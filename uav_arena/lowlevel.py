"""Batched onboard flight control: the pinned upstream Lee controller at firmware rate.

Every aircraft runs one unchanged `controller_lee.c` instance (IMRCLab firmware,
the commit pinned in `uav_transport.external_controller`) at its native 500 Hz
attitude rate, followed by the study's bounded rotor mixer. One ctypes call
advances all instances, so a physics step costs microseconds instead of a
Python loop per aircraft. Memory is one contiguous buffer, so snapshots are a
plain byte copy.
"""
from __future__ import annotations

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

from uav_transport.external_controller import UPSTREAM_COMMIT, VENDOR, verify_upstream


FIRMWARE_RATE_HZ = 500
_WRAPPER = r'''
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "controller_lee.h"

/* The upstream C file and headers remain unchanged; only the scheduler rate is set. */
#undef ATTITUDE_RATE
#define ATTITUDE_RATE ARENA_CONTROL_HZ
#include "controller_lee.c"

size_t arena_state_size(void) { return sizeof(controllerLee_t); }

void arena_init(void *memory, int count, float mass, const double *inertia) {
    controllerLee_t *items = (controllerLee_t *)memory;
    for (int i = 0; i < count; ++i) {
        controllerLee_t *self = items + i;
        memset(self, 0, sizeof(*self));
        controllerLeeInit(self);
        self->mass = mass;
        self->J = mkvec((float)inertia[0], (float)inertia[1], (float)inertia[2]);
    }
}

/* Same arithmetic as uav_transport.control.mix_bounded: bounded collective
 * thrust, then uniformly desaturated differential moments. */
static void arena_mix(const double *wrench, const double *inverse, double maximum, double *forces) {
    double collective = wrench[0] < 0.0 ? 0.0 : wrench[0];
    if (collective > 4.0*maximum) collective = 4.0*maximum;
    collective /= 4.0;
    double differential[4], scale = 1.0;
    for (int r = 0; r < 4; ++r)
        differential[r] = inverse[4*r+1]*wrench[1] + inverse[4*r+2]*wrench[2] + inverse[4*r+3]*wrench[3];
    for (int r = 0; r < 4; ++r) {
        double limit = 1.0;
        if (differential[r] > 0.0) limit = (maximum-collective)/differential[r];
        else if (differential[r] < 0.0) limit = -collective/differential[r];
        if (limit < scale) scale = limit;
    }
    if (scale < 0.0) scale = 0.0;
    for (int r = 0; r < 4; ++r) {
        double force = collective + scale*differential[r];
        forces[r] = force < 0.0 ? 0.0 : (force > maximum ? maximum : force);
    }
}

void arena_step(void *memory, int count, const double *position, const double *quaternion_wxyz,
    const double *velocity, const double *omega_body, const double *goal, const double *goal_velocity,
    const double *acceleration, const double *inverse_mixer, double maximum, double *forces, double *wrenches) {
    controllerLee_t *items = (controllerLee_t *)memory;
    for (int i = 0; i < count; ++i) {
        control_t control; setpoint_t setpoint; sensorData_t sensors; state_t state;
        memset(&control, 0, sizeof(control)); memset(&setpoint, 0, sizeof(setpoint));
        memset(&sensors, 0, sizeof(sensors)); memset(&state, 0, sizeof(state));
        const double *p = position+3*i, *q = quaternion_wxyz+4*i, *v = velocity+3*i;
        const double *w = omega_body+3*i, *g = goal+3*i, *gv = goal_velocity+3*i, *a = acceleration+3*i;
        state.position.x = (float)p[0]; state.position.y = (float)p[1]; state.position.z = (float)p[2];
        state.velocity.x = (float)v[0]; state.velocity.y = (float)v[1]; state.velocity.z = (float)v[2];
        state.attitudeQuaternion.x = (float)q[1]; state.attitudeQuaternion.y = (float)q[2];
        state.attitudeQuaternion.z = (float)q[3]; state.attitudeQuaternion.w = (float)q[0];
        sensors.gyro.x = degrees((float)w[0]); sensors.gyro.y = degrees((float)w[1]); sensors.gyro.z = degrees((float)w[2]);
        setpoint.mode.x = modeAbs; setpoint.mode.y = modeAbs; setpoint.mode.z = modeAbs; setpoint.mode.yaw = modeAbs;
        setpoint.position.x = (float)g[0]; setpoint.position.y = (float)g[1]; setpoint.position.z = (float)g[2];
        setpoint.velocity.x = (float)gv[0]; setpoint.velocity.y = (float)gv[1]; setpoint.velocity.z = (float)gv[2];
        setpoint.acceleration.x = (float)a[0]; setpoint.acceleration.y = (float)a[1]; setpoint.acceleration.z = (float)a[2];
        controllerLee(items+i, &control, &setpoint, &sensors, &state, 0);
        double *wrench = wrenches+4*i;
        wrench[0] = control.thrustSI; wrench[1] = control.torque[0];
        wrench[2] = control.torque[1]; wrench[3] = control.torque[2];
        arena_mix(wrench, inverse_mixer, maximum, forces+4*i);
    }
}

/* One onboard feedback tick read straight from MuJoCo's arrays: gather each
 * aircraft's state, add the nominal cable-load feedforward, run Lee and the
 * mixer, and write rotor forces into ctrl. */
void arena_control(void *memory, int count, const double *qpos, const double *qvel, double *ctrl,
    const int *qpos_uav, const int *qvel_uav, int qpos_payload, const int *ctrl_index,
    const double *goal, const double *goal_velocity, double nominal_mass, double gravity, double drone_mass,
    const double *inverse_mixer, double maximum, double *scratch) {
    double *position = scratch, *quaternion = scratch+3*count, *velocity = scratch+7*count;
    double *omega = scratch+10*count, *acceleration = scratch+13*count, *forces = scratch+16*count;
    double *wrenches = scratch+20*count;
    const double *payload = qpos+qpos_payload;
    for (int i = 0; i < count; ++i) {
        const double *p = qpos+qpos_uav[i], *v = qvel+qvel_uav[i];
        double cable[3], length = 0.0;
        for (int k = 0; k < 3; ++k) {
            position[3*i+k] = p[k]; velocity[3*i+k] = v[k]; omega[3*i+k] = v[3+k];
            cable[k] = p[k]-payload[k]; length += cable[k]*cable[k];
        }
        for (int k = 0; k < 4; ++k) quaternion[4*i+k] = p[3+k];
        length = sqrt(length); if (length < 1e-9) length = 1e-9;
        double vertical = cable[2]/length; if (vertical < 0.2) vertical = 0.2;
        double tension = nominal_mass*gravity/(3.0*vertical);
        for (int k = 0; k < 3; ++k) acceleration[3*i+k] = tension*cable[k]/length/drone_mass;
    }
    arena_step(memory, count, position, quaternion, velocity, omega, goal, goal_velocity, acceleration,
               inverse_mixer, maximum, forces, wrenches);
    for (int i = 0; i < 4*count; ++i) ctrl[ctrl_index[i]] = forces[i];
}
'''


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def native_library(rate_hz=FIRMWARE_RATE_HZ):
    if rate_hz not in (50, 100, 250, 500, 1000):
        raise ValueError("Use a declared firmware scheduler rate")
    upstream = verify_upstream()
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("Building the pinned upstream C controller requires a C compiler")
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    key = hashlib.sha256(json.dumps(dict(commit=UPSTREAM_COMMIT, files=upstream["files"],
        wrapper=_WRAPPER, rate_hz=rate_hz, compiler=version), sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get("UAV_ARENA_BUILD_CACHE",
        str(Path(__file__).resolve().parents[1]/"data/uav_arena/native_build")))/key
    cache.mkdir(parents=True, exist_ok=True)
    source, library = cache/"arena_lee.c", cache/"arena_lee.so"
    with (cache/"build.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not library.exists():
            source.write_text(_WRAPPER)
            temporary = cache/f"arena_lee.{os.getpid()}.so"
            command = [compiler, "-shared", "-fPIC", "-O2", "-std=c99", f"-DARENA_CONTROL_HZ={rate_hz}",
                       "-I", str(VENDOR), str(source), "-lm", "-o", str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("Pinned upstream controller compilation failed:\n"+result.stderr)
            (cache/"build.json").write_text(json.dumps(dict(build_key=key, compiler=version, command=command,
                source_sha256=_sha(source), library_sha256=_sha(temporary), upstream_commit=UPSTREAM_COMMIT,
                control_rate_hz=rate_hz), indent=2)+"\n")
            os.replace(temporary, library)
    build = json.loads((cache/"build.json").read_text())
    if _sha(library) != build["library_sha256"] or _sha(source) != build["source_sha256"]:
        raise AssertionError("Cached arena controller build changed")
    lib = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.arena_state_size.argtypes = []; lib.arena_state_size.restype = ctypes.c_size_t
    lib.arena_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, array]
    lib.arena_init.restype = None
    lib.arena_step.argtypes = [ctypes.c_void_p, ctypes.c_int]+[array]*8+[ctypes.c_double, array, array]
    lib.arena_step.restype = None
    # The bound fast path passes raw addresses of arrays that outlive the call.
    lib.arena_control.argtypes = ([ctypes.c_void_p, ctypes.c_int]+[ctypes.c_void_p]*5+[ctypes.c_int]
        +[ctypes.c_void_p]*3+[ctypes.c_double]*3+[ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p])
    lib.arena_control.restype = None
    return lib, build


class BatchedLee:
    """`count` independent upstream Lee instances plus the bounded rotor mixer."""

    def __init__(self, count, *, mass, inertia, mixer, rotor_max, rate_hz=FIRMWARE_RATE_HZ):
        self.lib, self.build = native_library(rate_hz)
        self.count, self.mass, self.rate_hz, self.rotor_max = int(count), float(mass), rate_hz, float(rotor_max)
        self.inertia = np.ascontiguousarray(inertia, dtype=np.float64)
        self.inverse_mixer = np.ascontiguousarray(np.linalg.inv(np.asarray(mixer, dtype=float)).reshape(-1))
        self.memory = ctypes.create_string_buffer(self.lib.arena_state_size()*self.count)
        self.forces = np.zeros(4*self.count)
        self.wrenches = np.zeros(4*self.count)
        self.reset()

    def reset(self):
        self.lib.arena_init(self.memory, self.count, self.mass, self.inertia)

    def step(self, position, quaternion_wxyz, velocity, omega_body, goal, acceleration, goal_velocity=None):
        """All inputs are (count, 3) or (count, 4) float64 arrays; returns rotor forces (count, 4)."""
        goal_velocity = np.zeros((self.count, 3)) if goal_velocity is None else goal_velocity
        arrays = [np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
                  for x in (position, quaternion_wxyz, velocity, omega_body, goal, goal_velocity, acceleration)]
        self.lib.arena_step(self.memory, self.count, *arrays, self.inverse_mixer, self.rotor_max,
                            self.forces, self.wrenches)
        return self.forces.reshape(self.count, 4)

    def bind(self, data, *, qpos_uav, qvel_uav, qpos_payload, ctrl_index, goal, goal_velocity, nominal_mass, gravity):
        """A zero-argument callable running one feedback tick on `data` in place.

        `goal` and `goal_velocity` are the (count, 3) float64 position and velocity setpoint
        arrays; update them in place (goal[:] = ...) so the bound addresses stay valid.
        MuJoCo array buffers keep their address for the lifetime of `data`, including
        across mj_setState.
        """
        for array in (goal, goal_velocity):
            if array.dtype != np.float64 or not array.flags.c_contiguous or array.shape != (self.count, 3):
                raise ValueError("Bind C-contiguous float64 (count, 3) setpoint arrays")
        keep = dict(qpos_uav=np.ascontiguousarray(qpos_uav, dtype=np.int32),
                    qvel_uav=np.ascontiguousarray(qvel_uav, dtype=np.int32),
                    ctrl_index=np.ascontiguousarray(ctrl_index, dtype=np.int32),
                    scratch=np.zeros(24*self.count), goal=goal, goal_velocity=goal_velocity)
        qpos, qvel, ctrl = data.qpos, data.qvel, data.ctrl
        args = (ctypes.c_void_p(ctypes.addressof(self.memory)), self.count, qpos.ctypes.data, qvel.ctypes.data,
                ctrl.ctypes.data, keep["qpos_uav"].ctypes.data, keep["qvel_uav"].ctypes.data, int(qpos_payload),
                keep["ctrl_index"].ctypes.data, goal.ctypes.data, goal_velocity.ctypes.data, float(nominal_mass),
                float(gravity), self.mass, self.inverse_mixer.ctypes.data, self.rotor_max, keep["scratch"].ctypes.data)
        function = self.lib.arena_control

        def tick():
            function(*args)
        tick.keep = (keep, qpos, qvel, ctrl)   # hold every referenced buffer alive
        return tick

    def capture(self):
        return self.memory.raw

    def restore(self, raw):
        if len(raw) != len(self.memory):
            raise ValueError("Controller snapshot has the wrong size")
        ctypes.memmove(self.memory, raw, len(raw))
