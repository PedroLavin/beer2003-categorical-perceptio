import numpy as np
import pytest

from src.catperc import config as C
from src.catperc.geometry import (CIRCLE, DIAMOND, DIAMOND_HALF_DIAG,
                              ray_distances, sense)

CENTER_RAY = C.N_RAYS // 2


def test_circle_straight_above_centre_ray():
    d = ray_distances(0.0, 100.0, CIRCLE)
    assert d[CENTER_RAY] == pytest.approx(100.0 - C.CIRCLE_RADIUS, abs=1e-4)


def test_diamond_straight_above_centre_ray():
    d = ray_distances(0.0, 100.0, DIAMOND)
    assert d[CENTER_RAY] == pytest.approx(100.0 - DIAMOND_HALF_DIAG, abs=1e-4)


def test_circle_on_an_angled_ray():
    angle = C.RAY_ANGLES[-1]
    dist = 80.0
    x, y = dist * np.sin(angle), dist * np.cos(angle)
    d = ray_distances(x, y, CIRCLE)
    assert d[-1] == pytest.approx(dist - C.CIRCLE_RADIUS, abs=1e-3)


@pytest.mark.parametrize("shape", [CIRCLE, DIAMOND])
def test_out_of_range_gives_zero_input(shape):
    assert np.allclose(sense(0.0, 400.0, shape), 0.0)


@pytest.mark.parametrize("shape", [CIRCLE, DIAMOND])
def test_far_to_the_side_gives_zero_input(shape):
    assert np.allclose(sense(150.0, 50.0, shape), 0.0)


@pytest.mark.parametrize("shape", [CIRCLE, DIAMOND])
@pytest.mark.parametrize("x", [3.0, 12.5, 30.0])
def test_mirror_symmetry(shape, x):
    left = np.asarray(sense(-x, 120.0, shape))
    right = np.asarray(sense(x, 120.0, shape))
    assert np.allclose(left, right[::-1], atol=1e-4)


def test_diamond_is_seen_by_at_least_as_many_rays_as_circle():
    for y in [60.0, 100.0, 150.0, 200.0]:
        n_circle = int(np.sum(sense(0.0, y, CIRCLE) > 0))
        n_diamond = int(np.sum(sense(0.0, y, DIAMOND) > 0))
        assert n_diamond >= n_circle


def test_input_decreases_with_distance():
    near = sense(0.0, 50.0, CIRCLE)[CENTER_RAY]
    far = sense(0.0, 150.0, CIRCLE)[CENTER_RAY]
    assert near > far > 0.0


def test_overlap_gives_max_input():
    assert sense(0.0, 0.0, CIRCLE)[CENTER_RAY] == pytest.approx(C.MAX_SENSOR_INPUT)