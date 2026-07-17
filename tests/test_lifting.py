# Unit tests for engine_grounder.spatial.lifting.

import numpy as np
import pytest

from engine_grounder.spatial.lifting import BBox2D, BBoxLifter

INTRINSICS = (500.0, 500.0, 160.0, 120.0)


def make_plane_depth(z: float = 2.0, shape=(240, 320)) -> np.ndarray:
    return np.full(shape, z, dtype=np.float32)


class TestBBox2D:
    def test_rejects_zero_area(self):
        with pytest.raises(ValueError):
            BBox2D(10, 10, 10, 20)

    def test_rejects_negative_coords(self):
        with pytest.raises(ValueError):
            BBox2D(-1, 0, 10, 10)

    def test_clamped_to_image(self):
        box = BBox2D(300, 200, 400, 300).clamped(320, 240)
        assert (box.x_max, box.y_max) == (320, 240)


class TestBBoxLifter:
    def test_clean_plane_z_within_5cm(self):
        depth = make_plane_depth(z=2.0)
        result = BBoxLifter().lift(depth, INTRINSICS, BBox2D(100, 80, 220, 160))
        assert result is not None
        assert abs(result.z_est - 2.0) < 0.05

    def test_corrupted_plane_z_within_5cm(self):
        rng = np.random.default_rng(0)
        depth = make_plane_depth(z=2.0)
        depth += rng.normal(0.0, 0.015, size=depth.shape).astype(np.float32)
        void = rng.random(depth.shape) < 0.5
        depth[void] = 0.0
        result = BBoxLifter().lift(depth, INTRINSICS, BBox2D(100, 80, 220, 160))
        assert result is not None
        assert abs(result.z_est - 2.0) < 0.05

    def test_vertices_form_valid_aabb(self):
        result = BBoxLifter().lift(make_plane_depth(), INTRINSICS, BBox2D(100, 80, 220, 160))
        assert result.vertices.shape == (8, 3)
        mins, maxs = result.vertices.min(axis=0), result.vertices.max(axis=0)
        expected = {
            (x, y, z)
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        }
        assert {tuple(v) for v in result.vertices} == expected
        assert result.z_min <= result.z_est <= result.z_max

    def test_fully_void_crop_returns_none_without_context(self):
        depth = np.zeros((240, 320), dtype=np.float32)
        assert BBoxLifter().lift(depth, INTRINSICS, BBox2D(100, 80, 220, 160)) is None

    def test_degenerate_bbox_returns_none(self):
        depth = make_plane_depth()
        assert BBoxLifter().lift(depth, INTRINSICS, BBox2D(0, 0, 4, 4)) is None
