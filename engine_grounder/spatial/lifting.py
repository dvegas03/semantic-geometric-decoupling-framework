# 2D bounding box → metric 3D AABB lifting.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine_grounder.geometry.depth_filter import RobustDepthEstimator
from engine_grounder.spatial.projector import Projector


@dataclass(frozen=True)
class BBox2D:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError(f"BBox2D coordinates must be non-negative, got {self}")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(f"BBox2D must have positive area, got {self}")

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def clamped(self, image_width: int, image_height: int) -> BBox2D:
        return BBox2D(
            x_min=max(self.x_min, 0),
            y_min=max(self.y_min, 0),
            x_max=min(self.x_max, image_width),
            y_max=min(self.y_max, image_height),
        )

    def as_slices(self) -> tuple[slice, slice]:
        return slice(self.y_min, self.y_max), slice(self.x_min, self.x_max)


@dataclass(frozen=True)
class LiftResult:
    vertices: np.ndarray
    z_est: float
    z_min: float
    z_max: float
    bbox: BBox2D
    inlier_ratio: float


class BBoxLifter:
    def __init__(
        self,
        estimator: RobustDepthEstimator | None = None,
        z_percentiles: tuple[float, float] = (5.0, 95.0),
        min_crop_px: int = 8,
    ):
        lo, hi = z_percentiles
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(f"Invalid z_percentiles {z_percentiles}")
        self.estimator = estimator if estimator is not None else RobustDepthEstimator()
        self.z_percentiles = z_percentiles
        self.min_crop_px = min_crop_px

    def lift(
        self,
        depth: np.ndarray,
        intrinsics: tuple[float, float, float, float],
        bbox: BBox2D,
    ) -> LiftResult | None:
        if depth.ndim != 2:
            raise ValueError(f"depth must be HxW, got shape {depth.shape}")
        h, w = depth.shape
        box = bbox.clamped(w, h)
        if box.width < self.min_crop_px or box.height < self.min_crop_px:
            return None

        rows, cols = box.as_slices()
        crop = depth[rows, cols].astype(np.float32)

        valid = RobustDepthEstimator.void_mask(crop)
        inlier_ratio = float(valid.mean())

        z_est = self.estimator.get_stable_z(
            crop,
            full_depth_map=depth,
            bbox=(box.y_min, box.y_max, box.x_min, box.x_max),
        )
        if z_est is None:
            return None

        restored = self.estimator.restore(crop)
        restored_valid = restored[RobustDepthEstimator.void_mask(restored)]
        if restored_valid.size == 0:
            return None

        lo, hi = self.z_percentiles
        z_min = float(np.percentile(restored_valid, lo))
        z_max = float(np.percentile(restored_valid, hi))

        vertices = self._aabb_vertices(box, z_min, z_max, intrinsics)
        return LiftResult(
            vertices=vertices,
            z_est=float(z_est),
            z_min=z_min,
            z_max=z_max,
            bbox=box,
            inlier_ratio=inlier_ratio,
        )

    @staticmethod
    def _aabb_vertices(
        box: BBox2D,
        z_min: float,
        z_max: float,
        intrinsics: tuple[float, float, float, float],
    ) -> np.ndarray:
        fx, fy, cx, cy = intrinsics
        projector = Projector(fx, fy, cx, cy)
        corners_px = (
            (box.x_min, box.y_min),
            (box.x_max, box.y_min),
            (box.x_max, box.y_max),
            (box.x_min, box.y_max),
        )
        frustum = np.array(
            [projector.backproject(u, v, z) for (u, v) in corners_px for z in (z_min, z_max)]
        )
        mins = frustum.min(axis=0)
        maxs = frustum.max(axis=0)
        return np.array(
            [
                [x, y, z]
                for z in (mins[2], maxs[2])
                for y in (mins[1], maxs[1])
                for x in (mins[0], maxs[0])
            ],
            dtype=np.float64,
        )
