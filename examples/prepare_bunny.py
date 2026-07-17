# Generates a simulated pinhole depth map from the Stanford Bunny mesh.

import os

import numpy as np
import open3d as o3d

IMG_H, IMG_W = 480, 640
FX = FY = 580.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0
CAMERA_DIST = 2.5
N_SAMPLES = 800_000
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_REPO_ROOT, "data")


def fill_mesh_holes(mesh):
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=100_000)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)

    watertight, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, linear_fit=True
    )
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    watertight = watertight.crop(bbox)
    watertight.compute_vertex_normals()
    return watertight


def build_depth_map(points):
    z = points[:, 2]
    u = (FX * points[:, 0] / z + CX).astype(np.int32)
    v = (FY * points[:, 1] / z + CY).astype(np.int32)

    in_bounds = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H) & (z > 0)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]

    order = np.argsort(-z)
    depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    depth[v[order], u[order]] = z[order].astype(np.float32)
    return depth


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading Stanford Bunny mesh via Open3D …")
    dataset = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)
    mesh.compute_vertex_normals()
    print(f"  Original mesh : {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    print("  Filling bottom hole (Poisson reconstruction) …")
    mesh = fill_mesh_holes(mesh)
    print(f"  Watertight mesh : {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    print(f"  Sampling {N_SAMPLES:,} surface points …")
    pcd = mesh.sample_points_uniformly(number_of_points=N_SAMPLES)
    pts = np.asarray(pcd.points, dtype=np.float64)

    pts -= pts.mean(axis=0)
    pts /= np.abs(pts).max()

    pts[:, 1] *= -1.0

    pts[:, 2] = CAMERA_DIST + pts[:, 2] * 0.9

    print("  Projecting to depth image …")
    depth = build_depth_map(pts)

    valid = depth > 0
    print(f"  Valid pixel coverage : {valid.mean() * 100:.1f} %")
    print(f"  Depth range          : [{depth[valid].min():.3f}, {depth[valid].max():.3f}] m")

    depth_path = os.path.join(OUT_DIR, "bunny_depth.npy")
    intrinsics_path = os.path.join(OUT_DIR, "bunny_intrinsics.npy")
    np.save(depth_path, depth)
    np.save(intrinsics_path, np.array([FX, FY, CX, CY], dtype=np.float32))

    print(f"\nSaved depth map    → {depth_path}")
    print(f"Saved intrinsics   → {intrinsics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
