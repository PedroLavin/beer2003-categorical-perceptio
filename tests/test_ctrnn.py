import numpy as np
import jax
import jax.numpy as jnp
import pytest

from catperc import config as C
from catperc.ctrnn import N_GENES, Params, decode, run, step, zero_state


def random_genome(seed=0):
    return jax.random.uniform(jax.random.PRNGKey(seed), (N_GENES,), minval=-1.0, maxval=1.0)


def test_genome_size():
    # 3 sensory + 18 W_sh + 13 W_hh + 5 W_hm + 3 bias_h + 3 tau_h + 1 bias_m + 1 tau_m
    assert N_GENES == 47


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decoded_params_are_mirror_symmetric(seed):
    p = decode(random_genome(seed))
    assert np.allclose(p.W_sh, p.W_sh[::-1, ::-1])
    assert np.allclose(p.W_hh, p.W_hh[::-1, ::-1])
    assert np.allclose(p.W_hm, p.W_hm[::-1, ::-1])
    for v in (p.bias_h, p.tau_h, p.bias_m, p.tau_m):
        assert np.allclose(v, v[::-1])


def test_every_gene_is_used():
    """Changing any single gene must change the decoded parameters."""
    base = jax.tree_util.tree_leaves(decode(jnp.zeros(N_GENES)))
    for k in range(N_GENES):
        g = jnp.zeros(N_GENES).at[k].set(0.5)
        new = jax.tree_util.tree_leaves(decode(g))
        assert any(not np.allclose(a, b) for a, b in zip(base, new)), f"gene {k} unused"


def test_genes_map_onto_ranges():
    lo = decode(-jnp.ones(N_GENES))
    hi = decode(jnp.ones(N_GENES))
    assert np.allclose(lo.W_sh, C.WEIGHT_RANGE[0]) and np.allclose(hi.W_sh, C.WEIGHT_RANGE[1])
    assert np.allclose(lo.tau_h, C.TAU_RANGE[0]) and np.allclose(hi.tau_h, C.TAU_RANGE[1])
    assert np.isclose(hi.gain_s, C.SENSORY_GAIN_RANGE[1])


def test_sensory_neuron_relaxes_to_input():
    """Zero weights: each sensory state follows exact Euler decay toward its input."""
    p = decode(jnp.zeros(N_GENES))._replace(
        W_sh=jnp.zeros((7, 5)), W_hh=jnp.zeros((5, 5)), W_hm=jnp.zeros((5, 2)))
    I, T = 3.0, 200
    states = run(p, jnp.full((T, 7), I))
    expected = I * (1.0 - (1.0 - C.DT / p.tau_s) ** np.arange(1, T + 1))
    assert np.allclose(states.s[:, 0], expected, atol=1e-4)
    assert abs(float(states.s[-1, 0]) - I) < 1e-3          # converged


def test_inter_and_motor_decay_without_input():
    p = decode(jnp.zeros(N_GENES))._replace(
        W_sh=jnp.zeros((7, 5)), W_hh=jnp.zeros((5, 5)), W_hm=jnp.zeros((5, 2)))
    state0 = zero_state()._replace(h=jnp.ones(5), m=jnp.ones(2))
    states = run(p, jnp.zeros((300, 7)), state0)
    assert np.allclose(states.h[-1], 0.0, atol=1e-3)
    assert np.allclose(states.m[-1], 0.0, atol=1e-3)


@pytest.mark.parametrize("seed", [0, 1])
def test_mirrored_input_gives_mirrored_trajectory(seed):
    """The key consequence of bilateral symmetry: flip the input, flip every state."""
    p = decode(random_genome(seed))
    inputs = jax.random.uniform(jax.random.PRNGKey(10 + seed), (150, 7), maxval=10.0)
    a = run(p, inputs)
    b = run(p, inputs[:, ::-1])
    assert np.allclose(a.s, b.s[:, ::-1], atol=1e-4)
    assert np.allclose(a.h, b.h[:, ::-1], atol=1e-4)
    assert np.allclose(a.m, b.m[:, ::-1], atol=1e-4)


def test_step_is_jittable():
    p = decode(random_genome())
    out = jax.jit(step)(p, zero_state(), jnp.ones(7))
    assert out.h.shape == (5,)