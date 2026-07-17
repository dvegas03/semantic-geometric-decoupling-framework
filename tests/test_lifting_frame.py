# Canary: datagen's GT 3D boxes and BBoxLifter's back-projection must agree.

from __future__ import annotations

import numpy as np
import pytest

from training.canary import measure_gt_clean_3d_iou
from training.datagen import RayCastRenderer

_MIN_MEDIAN_IOU = 0.30
_N_FRAMES = 30


def test_gt_aabb_and_lift_agree_on_clean_depth() -> None:
    median_iou = measure_gt_clean_3d_iou(n_frames=_N_FRAMES)
    assert median_iou >= _MIN_MEDIAN_IOU, (
        f"GT+clean-depth median 3D IoU={median_iou:.3f} < {_MIN_MEDIAN_IOU} "
        "— datagen GT boxes and BBoxLifter disagree on the camera frame "
        "(WS0 coordinate-convention regression)"
    )


def test_renderer_intrinsics_match_horizontal_fov() -> None:
    import open3d as o3d

    from training.datagen import PlacedObject

    renderer = RayCastRenderer(image_hw=(480, 640))
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.15, height=0.15, depth=0.15)
    mesh.compute_vertex_normals()
    mesh.translate((0.05, 0.15, 0.3))
    obj = PlacedObject(mesh=mesh, category="box", color=(1.0, 0.0, 0.0))

    _, _, instance = renderer.render([obj])
    ys, xs = np.where(instance == 1)
    assert xs.size > 0, "test object must be visible"
    rendered_w = float(xs.max() - xs.min())
    rendered_h = float(ys.max() - ys.min())

    fx, fy, cx, cy = renderer.intrinsics
    eye, center = renderer.cam_eye, renderer.cam_lookat
    up = np.array([0.0, 1.0, 0.0])
    forward = (center - eye) / np.linalg.norm(center - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    r_mat = np.stack([-right, true_up, forward], axis=0)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    cam = (r_mat @ (verts - eye).T).T
    u = fx * cam[:, 0] / cam[:, 2] + cx
    v = fy * cam[:, 1] / cam[:, 2] + cy
    predicted_w = float(u.max() - u.min())
    predicted_h = float(v.max() - v.min())

    assert predicted_w == pytest.approx(rendered_w, rel=0.15)
    assert predicted_h == pytest.approx(rendered_h, rel=0.15)
