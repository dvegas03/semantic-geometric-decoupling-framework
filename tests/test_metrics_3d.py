# Tests for 2D/3D IoU metric primitives.

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.metrics import aabb_iou_3d, bbox_iou_2d
from engine_grounder.spatial.lifting import BBox2D


def _box_vertices(mins, maxs) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for z in (mins[2], maxs[2])
            for y in (mins[1], maxs[1])
            for x in (mins[0], maxs[0])
        ],
        dtype=np.float64,
    )


class TestBBoxIoU2D:
    def test_self_iou_is_one(self):
        b = BBox2D(10, 20, 40, 60)
        assert bbox_iou_2d(b, b) == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        a = BBox2D(0, 0, 10, 10)
        b = BBox2D(20, 20, 30, 30)
        assert bbox_iou_2d(a, b) == 0.0

    def test_half_overlap_analytic(self):
        a = BBox2D(0, 0, 10, 10)
        b = BBox2D(5, 0, 15, 10)
        assert bbox_iou_2d(a, b) == pytest.approx(50 / 150)


class TestAabbIoU3D:
    def test_self_iou_is_one(self):
        v = _box_vertices((0, 0, 0), (1, 1, 1))
        assert aabb_iou_3d(v, v) == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        a = _box_vertices((0, 0, 0), (1, 1, 1))
        b = _box_vertices((2, 2, 2), (3, 3, 3))
        assert aabb_iou_3d(a, b) == 0.0

    def test_half_overlap_analytic(self):
        a = _box_vertices((0, 0, 0), (1, 1, 1))
        b = _box_vertices((0.5, 0, 0), (1.5, 1, 1))
        assert aabb_iou_3d(a, b) == pytest.approx(0.5 / 1.5)
