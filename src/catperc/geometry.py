"""
Ray sensing for the Beer (2003) agent.

Coordinates are agent-centred [p.214]: the agent's centre is the origin,
(x, y) is the object's centre relative to it, y > 0 means "above".
Rays start at the agent's centre [ours] and point upward; a ray's angle is
measured from vertical, positive to the right.
"""
import jax
import jax.numpy as jnp

from catperc import config as C

CIRCLE, DIAMOND = 0, 1
NO_HIT = jnp.inf                                   # distance returned when a ray misses
DIAMOND_HALF_DIAG = C.DIAMOND_SIDE / jnp.sqrt(2.0)  # centre-to-vertex distance (~21.2)


def ray_direction(angle):
    """Unit vector of a ray at `angle` radians from vertical."""
    return jnp.array([jnp.sin(angle), jnp.cos(angle)])


def _cross(a, b):
    """z-component of the 2D cross product a x b."""
    return a[0] * b[1] - a[1] * b[0]


def ray_circle_distance(angle, center, radius):
    """Distance along the ray to the first hit on a circle, or inf if it misses."""
    u = ray_direction(angle)
    b = jnp.dot(u, center)                         # projection of centre onto ray
    c = jnp.dot(center, center) - radius**2        # > 0 iff origin is outside the circle
    disc = b**2 - c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t_near = b - sqrt_disc
    t_far = b + sqrt_disc
    t = jnp.where(t_near >= 0.0, t_near,           # normal case: entry point ahead
        jnp.where(t_far >= 0.0, 0.0, NO_HIT))      # origin inside -> 0; both behind -> miss
    return jnp.where(disc >= 0.0, t, NO_HIT)


def ray_segment_distance(angle, p, q):
    """Distance along the ray to segment p-q, or inf if it misses."""
    u = ray_direction(angle)
    e = q - p
    denom = _cross(u, e)
    safe = jnp.where(jnp.abs(denom) < 1e-9, 1.0, denom)   # avoid 0/0 for parallel rays
    t = _cross(p, e) / safe                        # distance along the ray
    s = _cross(p, u) / safe                        # position along the segment, 0..1
    hit = (jnp.abs(denom) >= 1e-9) & (t >= 0.0) & (s >= 0.0) & (s <= 1.0)
    return jnp.where(hit, t, NO_HIT)


def ray_diamond_distance(angle, center, half_diag=DIAMOND_HALF_DIAG):
    """Distance to a diamond (square rotated 45 deg, vertices up/down/left/right)."""
    h = half_diag
    verts = center + jnp.array([[h, 0.0], [0.0, h], [-h, 0.0], [0.0, -h]])
    p = verts
    q = jnp.roll(verts, -1, axis=0)                # each vertex paired with the next one
    d = jax.vmap(ray_segment_distance, in_axes=(None, 0, 0))(angle, p, q)
    inside = jnp.abs(center[0]) + jnp.abs(center[1]) <= h
    return jnp.where(inside, 0.0, jnp.min(d))


def ray_distance(angle, center, shape):
    """Dispatch on shape without Python `if`, so it works under jit/vmap."""
    d_circle = ray_circle_distance(angle, center, C.CIRCLE_RADIUS)
    d_diamond = ray_diamond_distance(angle, center)
    return jnp.where(shape == CIRCLE, d_circle, d_diamond)


def distance_to_input(d):
    """Linear map: max input at d=0, zero at/after max ray length [p.214; linear form ours]."""
    d = jnp.minimum(d, C.RAY_MAX_LENGTH)
    # (L - d) / L rather than 1 - d / L: the compiler may turn d / L into
    # d * (1/L), which in float32 leaves ~3e-7 instead of an exact 0 at d = L.
    return C.MAX_SENSOR_INPUT * (C.RAY_MAX_LENGTH - d) / C.RAY_MAX_LENGTH


RAY_ANGLES = jnp.array(C.RAY_ANGLES)


@jax.jit
def ray_distances(obj_x, obj_y, shape):
    """Hit distance for all 7 rays (inf where a ray misses)."""
    center = jnp.array([obj_x, obj_y])
    return jax.vmap(ray_distance, in_axes=(0, None, None))(RAY_ANGLES, center, shape)


@jax.jit
def sense(obj_x, obj_y, shape):
    """Sensor inputs for all 7 rays, ordered left to right."""
    return distance_to_input(ray_distances(obj_x, obj_y, shape))