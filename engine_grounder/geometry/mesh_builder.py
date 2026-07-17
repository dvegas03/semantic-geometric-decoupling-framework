# Point cloud to mesh reconstruction via Open3D.

from __future__ import annotations

import numpy as np
import open3d as o3d


class MeshBuilder:
    def __init__(
        self,
        poisson_depth: int = 8,
        density_quantile: float = 0.05,
        voxel_size: float = 0.01,
    ):
        self.poisson_depth = poisson_depth
        self.density_quantile = density_quantile
        self.voxel_size = voxel_size

    def from_point_cloud(
        self,
        points: np.ndarray,
        normals: np.ndarray | None = None,
    ) -> o3d.geometry.TriangleMesh:
        if len(points) < 30:
            return o3d.geometry.TriangleMesh()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        if len(pcd.points) < 30:
            return o3d.geometry.TriangleMesh()

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(pcd.points) < 30:
            return o3d.geometry.TriangleMesh()

        pts_np = np.asarray(pcd.points)
        bbox_diag = float(np.linalg.norm(pts_np.max(axis=0) - pts_np.min(axis=0)))
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        mean_nn = float(dists.mean()) if dists.size > 0 else 0.05
        alpha_candidates = sorted(
            {max(mean_nn * scale, 0.05) for scale in (3.0, 6.0, 12.0, 24.0, 30.0)}
            | {bbox_diag * frac for frac in (0.25, 0.35, 0.5, 0.75)}
        )

        for alpha in alpha_candidates:
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            except Exception:
                continue
            if len(mesh.triangles) > 0:
                mesh.compute_vertex_normals()
                return mesh

        return o3d.geometry.TriangleMesh()

    @staticmethod
    def simplify(
        mesh: o3d.geometry.TriangleMesh,
        target_faces: int = 10_000,
    ) -> o3d.geometry.TriangleMesh:
        if len(mesh.triangles) <= target_faces or len(mesh.triangles) == 0:
            return mesh

        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        simplified.compute_vertex_normals()
        return simplified

    @staticmethod
    def sample_points(
        mesh: o3d.geometry.TriangleMesh,
        n: int = 2048,
    ) -> np.ndarray:
        if len(mesh.triangles) == 0:
            return np.zeros((n, 3), dtype=np.float64)

        pcd = mesh.sample_points_uniformly(number_of_points=n)
        return np.asarray(pcd.points, dtype=np.float64)
