"""
CTRNN controller for the Beer (2003) agent.

Neurons [p.214, p.223, p.225]:
  sensory  0-6   (paper: 1-7)    one per ray, left to right
  inter    0-4   (paper: 8-12)   mirror pairs (0,4), (1,3); 2 is central
  motor    0-1   (paper: 13-14)  mirror pair

State equation [p.214, Eq. 1; check the PDF for the exact form]:
  sensory: tau_s ds/dt = -s + I_ray
  inter:   tau_j ds/dt = -s + sum_i W_sh[i,j] sigma(g_s (s_i + theta_s))
                            + sum_k W_hh[k,j] sigma(s_k + theta_k)
  motor:   tau_m ds/dt = -s + sum_j W_hm[j,m] sigma(s_j + theta_j)

Bilateral symmetry [p.214]: every parameter equals its mirror image.
The genome stores one number per mirror pair ("orbit"); `decode` expands it.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from catperc import config as C

NS, NH, NM = C.N_SENSORY, C.N_INTER, C.N_MOTOR


# ---------------------------------------------------------------- containers
class Params(NamedTuple):
    tau_s: jnp.ndarray     # scalar, shared by all sensory neurons
    bias_s: jnp.ndarray    # scalar
    gain_s: jnp.ndarray    # scalar
    W_sh: jnp.ndarray      # (7, 5) sensory -> inter
    W_hh: jnp.ndarray      # (5, 5) inter -> inter (incl. self-connections)
    W_hm: jnp.ndarray      # (5, 2) inter -> motor
    bias_h: jnp.ndarray    # (5,)
    tau_h: jnp.ndarray     # (5,)
    bias_m: jnp.ndarray    # (2,)
    tau_m: jnp.ndarray     # (2,)


class State(NamedTuple):
    s: jnp.ndarray         # (7,) sensory states
    h: jnp.ndarray         # (5,) inter states
    m: jnp.ndarray         # (2,) motor states


def zero_state():
    return State(jnp.zeros(NS), jnp.zeros(NH), jnp.zeros(NM))


# ---------------------------------------------------------------- symmetry bookkeeping
def _mirror(n):
    """Index map i -> n-1-i (left-right reflection of a layer of n neurons)."""
    return np.arange(n)[::-1]


def _orbit_index(shape, mirrors):
    """
    For an array of `shape`, label each entry with the id of its mirror orbit.
    Entry (i, j) and (mirror_i, mirror_j) get the same label.
    Returns (labels with same shape, number of distinct orbits).
    """
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    mirrored = idx[np.ix_(*mirrors)] if len(shape) > 1 else idx[mirrors[0]]
    canonical = np.minimum(idx, mirrored)               # one representative per orbit
    _, labels = np.unique(canonical, return_inverse=True)
    labels = labels.reshape(shape)
    return labels, int(labels.max()) + 1


# Built once at import time with plain numpy (no tracing involved).
_LAYOUT = {
    "W_sh":   _orbit_index((NS, NH), (_mirror(NS), _mirror(NH))),
    "W_hh":   _orbit_index((NH, NH), (_mirror(NH), _mirror(NH))),
    "W_hm":   _orbit_index((NH, NM), (_mirror(NH), _mirror(NM))),
    "bias_h": _orbit_index((NH,), (_mirror(NH),)),
    "tau_h":  _orbit_index((NH,), (_mirror(NH),)),
    "bias_m": _orbit_index((NM,), (_mirror(NM),)),
    "tau_m":  _orbit_index((NM,), (_mirror(NM),)),
}
_SCALARS = ["tau_s", "bias_s", "gain_s"]

# Genome = [3 sensory scalars] + [orbit values of each array, in _LAYOUT order]
_OFFSETS = {}
_pos = len(_SCALARS)
for _name, (_labels, _n) in _LAYOUT.items():
    _OFFSETS[_name] = _pos
    _pos += _n
N_GENES = _pos


# ---------------------------------------------------------------- genome -> parameters
def _scale(g, lo_hi):
    """Map a gene in [-1, 1] linearly onto the range (lo, hi)."""
    lo, hi = lo_hi
    g = jnp.clip(g, -1.0, 1.0)
    return lo + (g + 1.0) * 0.5 * (hi - lo)


_RANGES = {
    "tau_s": C.TAU_RANGE, "bias_s": C.BIAS_RANGE, "gain_s": C.SENSORY_GAIN_RANGE,
    "W_sh": C.WEIGHT_RANGE, "W_hh": C.WEIGHT_RANGE, "W_hm": C.WEIGHT_RANGE,
    "bias_h": C.BIAS_RANGE, "tau_h": C.TAU_RANGE,
    "bias_m": C.BIAS_RANGE, "tau_m": C.TAU_RANGE,
}


def decode(genome):
    """Expand a genome of N_GENES numbers in [-1, 1] into full, symmetric Params."""
    out = {}
    for k, name in enumerate(_SCALARS):
        out[name] = _scale(genome[k], _RANGES[name])
    for name, (labels, n) in _LAYOUT.items():
        start = _OFFSETS[name]
        orbit_values = genome[start:start + n]
        out[name] = _scale(orbit_values[labels], _RANGES[name])   # fan out to full array
    return Params(**out)


# ---------------------------------------------------------------- dynamics
def outputs(params, state):
    """Firing rates o = sigma(g (s + theta)) of every layer."""
    o_s = jax.nn.sigmoid(params.gain_s * (state.s + params.bias_s))
    o_h = jax.nn.sigmoid(state.h + params.bias_h)
    o_m = jax.nn.sigmoid(state.m + params.bias_m)
    return o_s, o_h, o_m


def step(params, state, sensor_input, dt=C.DT):
    """One forward-Euler step of the whole network. sensor_input has shape (7,)."""
    o_s, o_h, _ = outputs(params, state)
    ds = (-state.s + sensor_input) / params.tau_s
    dh = (-state.h + o_s @ params.W_sh + o_h @ params.W_hh) / params.tau_h
    dm = (-state.m + o_h @ params.W_hm) / params.tau_m
    return State(state.s + dt * ds, state.h + dt * dh, state.m + dt * dm)


@jax.jit
def run(params, inputs, state0=None):
    """Drive the network with a sequence of sensor inputs, shape (T, 7). Returns all states."""
    if state0 is None:
        state0 = zero_state()

    def body(state, inp):
        new = step(params, state, inp)
        return new, new

    _, states = jax.lax.scan(body, state0, inputs)
    return states