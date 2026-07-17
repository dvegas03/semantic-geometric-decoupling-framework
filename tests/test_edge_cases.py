# Edge-case tests: empty arrays, all-NaN, sub-kernel images, coplanar points, degenerate meshes.

import math

import numpy as np
import open3d as o3d
import pytest

from engine_grounder.geometry.depth_filter import RobustDepthEstimator
from engine_grounder.geometry.mesh_builder import MeshBuilder
from engine_grounder.perception.shape_descriptor import ShapeDescriptor
from engine_grounder.perception.shape_encoder import ShapeEncoder


class TestDepthFilterEdgeCases:
    @pytest.fixture
    def estimator(self):
        return RobustDepthEstimator(bilateral_d=9)

    def test_all_zeros_depth(self, estimator):
        depth = np.zeros((100, 100), dtype=np.float32)
        outlier_mask, sigma, thresh = estimator.bilateral_outlier_mask(depth)

        assert not outlier_mask.any(), "Zeros should not be flagged as outliers"
        assert sigma.max() == 0.0, "Sigma should be 0 for empty image"
        assert estimator.get_stable_z(depth) is None, "Stable Z should be None"

    def test_all_nan_inf_depth(self, estimator):
        depth = np.full((50, 50), np.nan, dtype=np.float32)
        depth[10:20, 10:20] = np.inf

        mask = estimator.void_mask(depth)
        assert not mask.any(), "NaN and Inf must be masked out"

        outlier_mask, _, _ = estimator.bilateral_outlier_mask(depth)
        assert not outlier_mask.any(), "Should not crash on NaN/Inf math"

    def test_micro_depth_map(self, estimator):
        depth = np.ones((5, 5), dtype=np.float32)

        z_est = estimator.get_stable_z(depth)
        assert z_est == 1.0, "Micro arrays should fallback to raw median"


class TestMeshBuilderEdgeCases:
    @pytest.fixture
    def builder(self):
        return MeshBuilder(voxel_size=0.01)

    def test_empty_point_cloud(self, builder):
        pts = np.empty((0, 3), dtype=np.float64)
        mesh = builder.from_point_cloud(pts)
        assert len(mesh.vertices) == 0, "Must return empty mesh safely"

    def test_under_30_points(self, builder):
        pts = np.random.rand(29, 3)
        mesh = builder.from_point_cloud(pts)
        assert len(mesh.vertices) == 0, "Must abort if under 30 points"

    def test_collapse_during_voxelization(self, builder):
        pts = np.ones((100, 3), dtype=np.float64)
        mesh = builder.from_point_cloud(pts)
        assert len(mesh.vertices) == 0, "Must abort if voxelization drops count < 30"

    def test_simplify_empty_mesh(self, builder):
        mesh = o3d.geometry.TriangleMesh()
        simp = builder.simplify(mesh, target_faces=100)
        assert len(simp.triangles) == 0

    def test_sample_empty_mesh(self, builder):
        mesh = o3d.geometry.TriangleMesh()
        pts = builder.sample_points(mesh, n=2048)
        assert pts.shape == (2048, 3)
        assert np.all(pts == 0), "Should return zeroed array for empty mesh"


class TestShapeDescriptorEdgeCases:
    def test_empty_mesh_description(self):
        mesh = o3d.geometry.TriangleMesh()
        desc = ShapeDescriptor.describe(mesh)

        assert math.isnan(desc["volume"])
        assert desc["surface_area"] == 0.0
        assert desc["n_vertices"] == 0
        assert ShapeDescriptor.classify(desc) == "unknown (non-watertight)"

    def test_flat_coplanar_mesh(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.array(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                dtype=np.float64,
            )
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        )

        desc = ShapeDescriptor.describe(mesh)

        assert math.isnan(desc["volume"]) or desc["volume"] == 0.0
        assert desc["n_vertices"] == 4


class TestShapeEncoderEdgeCases:
    @pytest.fixture
    def encoder(self):
        return ShapeEncoder(device="cpu")

    def test_zero_scale_points(self, encoder):
        pts = np.zeros((2048, 3), dtype=np.float32)
        emb = encoder.encode(pts)

        assert emb.shape == (256,)
        assert not np.isnan(emb).any(), "Network should not produce NaNs on zero inputs"

    def test_extreme_scale_points(self, encoder):
        pts = np.random.rand(1024, 3).astype(np.float32) * 1e9
        emb = encoder.encode(pts)

        assert emb.shape == (256,)
        assert not np.isnan(emb).any()
