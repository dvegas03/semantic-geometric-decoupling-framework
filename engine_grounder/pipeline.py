# End-to-end pipeline: RGB-D ingest -> depth filter -> 3-D backproject -> mesh reconstruction -> shape encoding -> shape description -> VLM context.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import open3d as o3d

from engine_grounder.geometry.depth_filter import RobustDepthEstimator
from engine_grounder.geometry.mesh_builder import MeshBuilder
from engine_grounder.perception.shape_descriptor import ShapeDescriptor
from engine_grounder.perception.shape_encoder import ShapeEncoder
from engine_grounder.perception.vlm_agent import VLMAgent
from engine_grounder.spatial.projector import Projector


@dataclass
class PipelineResult:
    raw_depth: np.ndarray = field(repr=False)
    intrinsics: Tuple[float, float, float, float] | None = None
    rgb: np.ndarray | None = field(default=None, repr=False)

    clean_depth: np.ndarray = field(default=None, repr=False)
    outlier_mask: np.ndarray = field(default=None, repr=False)
    sigma_map: np.ndarray = field(default=None, repr=False)
    thresh_map: np.ndarray = field(default=None, repr=False)

    point_cloud: np.ndarray = field(default=None, repr=False)

    mesh: o3d.geometry.TriangleMesh | None = field(default=None, repr=False)
    mesh_simplified: o3d.geometry.TriangleMesh | None = field(default=None, repr=False)

    embedding: np.ndarray | None = field(default=None, repr=False)
    per_point_features: np.ndarray | None = field(default=None, repr=False)
    encoder_points: np.ndarray | None = field(default=None, repr=False)

    shape_descriptors: Dict[str, float] | None = None
    shape_label: str | None = None
    shape_text: str | None = None

    z_est: float | None = None
    z_true: float | None = None

    timings: Dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(
        self,
        stream,
        outlier_k: float = 3.0,
        tau: float = 0.10,
        encoder_npts: int = 2048,
        mesh_faces: int = 10_000,
    ):
        self.stream = stream
        self.estimator = RobustDepthEstimator(
            tau_threshold=tau,
            outlier_k=outlier_k,
        )
        self.mesh_builder = MeshBuilder()
        self.encoder = ShapeEncoder()
        self.descriptor = ShapeDescriptor()
        self.vlm = VLMAgent()
        self.encoder_npts = encoder_npts
        self.mesh_faces = mesh_faces

    def run(self, corrupt_fn=None) -> PipelineResult:
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        frame = self.stream.get_frame()
        raw_depth = frame["depth"].astype(np.float32)
        intrinsics = frame.get("intrinsics")
        rgb = frame.get("rgb")
        timings["ingest"] = time.perf_counter() - t0

        depth_input = corrupt_fn(raw_depth) if corrupt_fn else raw_depth.copy()
        z_true = float(np.median(raw_depth[raw_depth > 0])) if (raw_depth > 0).any() else None

        t1 = time.perf_counter()
        outlier_mask, sigma_map, thresh_map = self.estimator.bilateral_outlier_mask(depth_input)
        clean_depth = depth_input.copy()
        clean_depth[outlier_mask] = 0.0
        z_est = self.estimator.get_stable_z(
            depth_input,
            full_depth_map=depth_input,
            bbox=(0, depth_input.shape[0], 0, depth_input.shape[1]),
        )
        timings["filter"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        if intrinsics is not None:
            fx, fy, cx, cy = intrinsics
            proj = Projector(fx, fy, cx, cy)
            cloud_3d = proj.backproject_depth_map(clean_depth)
        else:
            h, w = clean_depth.shape
            cloud_3d = np.zeros((h, w, 3), dtype=np.float64)
            U, V = np.meshgrid(np.arange(w), np.arange(h))
            cloud_3d[..., 0] = U.astype(np.float64)
            cloud_3d[..., 1] = V.astype(np.float64)
            cloud_3d[..., 2] = clean_depth.astype(np.float64)

        valid = clean_depth > 0
        points = cloud_3d[valid]
        timings["backproject"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        mesh = self.mesh_builder.from_point_cloud(points)
        mesh_simp = MeshBuilder.simplify(mesh, target_faces=self.mesh_faces)
        timings["mesh"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        enc_pts = MeshBuilder.sample_points(mesh_simp, n=self.encoder_npts)
        enc_pts_c = enc_pts - enc_pts.mean(axis=0)
        scale = np.abs(enc_pts_c).max()
        if scale > 0:
            enc_pts_c /= scale
        embedding = self.encoder.encode(enc_pts_c.astype(np.float32))
        per_point = self.encoder.encode_per_point(enc_pts_c.astype(np.float32))
        timings["encode"] = time.perf_counter() - t4

        t5 = time.perf_counter()
        if len(mesh.triangles) == 0:
            desc = {
                "volume": float("nan"),
                "surface_area": 0.0,
                "compactness": float("nan"),
                "aspect_ratio": float("nan"),
                "convex_hull_ratio": float("nan"),
                "n_vertices": 0,
                "n_faces": 0,
            }
            label = "unknown (no mesh)"
            text = self.descriptor.to_text(desc, label)
        else:
            desc = self.descriptor.describe(mesh)
            label = self.descriptor.classify(desc)
            text = self.descriptor.to_text(desc, label)
        timings["describe"] = time.perf_counter() - t5

        self.vlm.set_spatial_context(embedding, text, z_est)

        timings["total"] = time.perf_counter() - t0

        return PipelineResult(
            raw_depth=depth_input,
            intrinsics=intrinsics,
            rgb=rgb,
            clean_depth=clean_depth,
            outlier_mask=outlier_mask,
            sigma_map=sigma_map,
            thresh_map=thresh_map,
            point_cloud=points,
            mesh=mesh,
            mesh_simplified=mesh_simp,
            embedding=embedding,
            per_point_features=per_point,
            encoder_points=enc_pts_c.astype(np.float32),
            shape_descriptors=desc,
            shape_label=label,
            shape_text=text,
            z_est=z_est,
            z_true=z_true,
            timings=timings,
        )
