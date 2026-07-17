# Preflight invariants the evolving loop must pass before it spends compute.

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("training.canary")

_DEFAULT_MIN_3D_IOU = 0.30
_DEFAULT_N_FRAMES = 20


class SystemHalt(RuntimeError):
    pass


def measure_gt_clean_3d_iou(n_frames: int = _DEFAULT_N_FRAMES, seed_offset: int = 0) -> float:
    from benchmarks.metrics import aabb_iou_3d
    from engine_grounder.spatial.lifting import BBoxLifter
    from training.datagen import (
        CategorySplit,
        FrameComposer,
        PrimitiveAssetLibrary,
        QueryGenerator,
        QuestDepthCorrupter,
        RayCastRenderer,
        SceneSampler,
    )

    library = PrimitiveAssetLibrary()
    composer = FrameComposer(
        library=library,
        split=CategorySplit(library.categories(), 0.2),
        sampler=SceneSampler(library, max_objects=10),
        renderer=RayCastRenderer(image_hw=(480, 640)),
        corrupter=QuestDepthCorrupter(),
        queries=QueryGenerator(),
    )
    lifter = BBoxLifter()
    ious: list[float] = []
    for i in range(n_frames):
        record = composer.compose(i, np.random.default_rng(seed_offset + i))
        lift = lifter.lift(record.clean_depth, record.intrinsics, record.bbox_2d)
        if lift is None:
            continue
        ious.append(aabb_iou_3d(lift.vertices, record.aabb_3d))
    if not ious:
        return 0.0
    return float(np.median(ious))


def check_bbox_format_roundtrip() -> bool:
    import json

    from engine_grounder.perception.vlm_agent import VLMAgent
    from engine_grounder.spatial.lifting import BBox2D

    bbox = BBox2D(258, 121, 321, 188)
    answer = json.dumps({"bbox_2d": [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]})
    coords = VLMAgent._extract_bbox(answer)
    if coords is None:
        return False
    parsed = VLMAgent._to_pixel_bbox(coords, width=640, height=480)
    return parsed is not None and (
        parsed.x_min,
        parsed.y_min,
        parsed.x_max,
        parsed.y_max,
    ) == (258, 121, 321, 188)


def check_pointnet_export_loads() -> bool:
    import tempfile
    from pathlib import Path

    import torch

    from engine_grounder.perception.shape_encoder import PointNetEncoder
    from training.train_pointnet import BestModelTracker, PointNetClassifier

    model = PointNetClassifier(num_classes=40)
    tracker = BestModelTracker()
    tracker.observe(model, acc=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = tracker.export(Path(tmp))
        encoder = PointNetEncoder(embed_dim=256)
        encoder.load_state_dict(torch.load(path, weights_only=True))
        emb, _ = encoder(torch.randn(2, 3, 1024))
    return tuple(emb.shape) == (2, 256)


@dataclass(frozen=True)
class CanaryCheck:
    name: str
    run: Callable[[], bool]
    description: str


class CanarySuite:
    def __init__(self, checks: tuple[CanaryCheck, ...]):
        if not checks:
            raise ValueError("CanarySuite requires at least one check")
        self._checks = checks

    def assert_healthy(self) -> None:
        failed: list[str] = []
        for check in self._checks:
            try:
                ok = check.run()
            except Exception:
                log.exception("canary check %r raised", check.name)
                ok = False
            if ok:
                log.info("canary check %r passed", check.name)
            else:
                log.error("canary check %r FAILED: %s", check.name, check.description)
                failed.append(check.name)
        if failed:
            raise SystemHalt(
                f"canary checks failed: {failed} — refusing to consume compute until fixed"
            )

    @classmethod
    def default(cls, min_3d_iou: float = _DEFAULT_MIN_3D_IOU) -> CanarySuite:
        return cls(
            (
                CanaryCheck(
                    name="gt_clean_3d_iou",
                    run=lambda: measure_gt_clean_3d_iou() >= min_3d_iou,
                    description=(
                        f"median GT-vs-lifted 3D IoU on clean depth must be "
                        f">= {min_3d_iou} (WS0 coordinate-frame regression)"
                    ),
                ),
                CanaryCheck(
                    name="bbox_format_roundtrip",
                    run=check_bbox_format_roundtrip,
                    description="LoRA collator answer format must parse back via VLMAgent",
                ),
                CanaryCheck(
                    name="pointnet_export_loads",
                    run=check_pointnet_export_loads,
                    description="PointNet best-model export must load into ShapeEncoder",
                ),
            )
        )
