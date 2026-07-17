# Held-out-shard evaluator with naive-vs-robust lifting (H2) and VLM IoU (H1).

from __future__ import annotations

import io
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from benchmarks.metrics import aabb_iou_3d, bbox_iou_2d, z_error_cm
from engine_grounder.spatial.lifting import BBox2D, BBoxLifter, LiftResult
from training.base import Trainer, TrainerOutcome
from training.detectors import Detector, DetectorFactory

log = logging.getLogger("training.eval")


class NaiveLifter:
    def lift(
        self,
        depth: np.ndarray,
        intrinsics: tuple[float, float, float, float],
        bbox: BBox2D,
    ) -> LiftResult | None:
        crop = depth[bbox.as_slices()]
        valid = crop[crop > 0.0]
        if valid.size == 0:
            return None
        z_est = float(np.median(valid))
        z_min = float(valid.min())
        z_max = float(valid.max())
        vertices = BBoxLifter._aabb_vertices(bbox, z_min, z_max, intrinsics)
        return LiftResult(
            vertices=vertices,
            z_est=z_est,
            z_min=z_min,
            z_max=z_max,
            bbox=bbox,
            inlier_ratio=float((crop > 0.0).mean()),
        )


@dataclass(frozen=True)
class EvalSample:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: tuple[float, float, float, float]
    query: str
    gt_bbox: BBox2D
    gt_aabb: np.ndarray
    category: str
    split: str
    severity: float
    meta: dict[str, Any]


class SampleSource(ABC):
    name: str = "base"

    @abstractmethod
    def __iter__(self) -> Iterator[EvalSample]: ...


class ShardSampleSource(SampleSource):
    name = "synthetic"

    def __init__(
        self,
        shards: Sequence[Path],
        split: str = "heldout",
        max_samples: int | None = None,
    ):
        self.shards = list(shards)
        self.split = split
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[EvalSample]:
        import webdataset as wds

        yielded = 0
        for shard in self.shards:
            dataset = wds.WebDataset(str(shard), shardshuffle=False).decode()
            for sample in dataset:
                if self.max_samples is not None and yielded >= self.max_samples:
                    return
                meta = sample["meta.json"]
                if isinstance(meta, (bytes, bytearray, str)):
                    meta = json.loads(meta)
                if meta.get("split") != self.split and self.split != "all":
                    continue

                rgb = sample["rgb.png"]
                if isinstance(rgb, Image.Image):
                    rgb_np = np.asarray(rgb.convert("RGB"))
                else:
                    rgb_np = np.asarray(Image.open(io.BytesIO(rgb)).convert("RGB"))

                depth = sample["corrupted_depth.npy"]
                if not isinstance(depth, np.ndarray):
                    depth = np.load(io.BytesIO(depth))

                gt_bbox = BBox2D(*[int(v) for v in meta["bbox_2d"]])
                gt_aabb = np.asarray(meta["aabb_3d"], dtype=np.float64)
                intrinsics = tuple(float(x) for x in meta["intrinsics"])
                severity = float(meta.get("corruption_params", {}).get("void_ratio", 0.0))
                yielded += 1
                yield EvalSample(
                    rgb=rgb_np,
                    depth=depth,
                    intrinsics=intrinsics,  # type: ignore[arg-type]
                    query=str(meta.get("query_text", "")),
                    gt_bbox=gt_bbox,
                    gt_aabb=gt_aabb,
                    category=str(meta.get("category", "")),
                    split=str(meta.get("split", "")),
                    severity=severity,
                    meta=meta,
                )


class SunRgbdSampleSource(SampleSource):
    name = "sunrgbd"

    def __init__(self, root: Path, max_frames: int = 50):
        self.root = Path(root)
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[EvalSample]:
        image_dir = self.root / "image"
        depth_dir = self.root / "depth"
        meta_dir = self.root / "meta"
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"SUN RGB-D mini root missing image/: {self.root} "
                "(run scripts/fetch_sunrgbd_mini.sh)"
            )
        frames = sorted(image_dir.glob("*"))[: self.max_frames]
        for img_path in frames:
            stem = img_path.stem
            depth_path = depth_dir / f"{stem}.png"
            if not depth_path.exists():
                depth_path = depth_dir / f"{stem}.npy"
            meta_path = meta_dir / f"{stem}.json"
            if not depth_path.exists() or not meta_path.exists():
                continue
            rgb = np.asarray(Image.open(img_path).convert("RGB"))
            if depth_path.suffix == ".npy":
                depth = np.load(depth_path).astype(np.float32)
            else:
                depth_mm = np.asarray(Image.open(depth_path))
                depth = depth_mm.astype(np.float32) / 1000.0
            meta = json.loads(meta_path.read_text())
            label = str(meta.get("label", meta.get("category", "object")))
            bbox = meta["bbox_2d"]
            gt_bbox = BBox2D(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            gt_aabb = np.asarray(meta["aabb_3d"], dtype=np.float64)
            K = meta["intrinsics"]
            intrinsics = (float(K[0]), float(K[1]), float(K[2]), float(K[3]))
            yield EvalSample(
                rgb=rgb,
                depth=depth,
                intrinsics=intrinsics,
                query=str(meta.get("query_text", f"the {label}")),
                gt_bbox=gt_bbox,
                gt_aabb=gt_aabb,
                category=label,
                split="sunrgbd",
                severity=0.0,
                meta=meta,
            )


class SampleSourceFactory:
    def __init__(self, profile: Any):
        self._profile = profile

    def create(self, cfg: dict[str, Any]) -> SampleSource:
        kind = str(cfg.get("source", "synthetic"))
        split = str(cfg.get("split", "heldout"))
        max_samples = cfg.get("max_samples")
        max_n = int(max_samples) if isinstance(max_samples, (int, float, str)) else None
        if kind == "synthetic":
            shards = sorted(self._profile.paths.data_root.glob("shard-*.tar"))
            if not shards:
                raise FileNotFoundError(f"no shards under {self._profile.paths.data_root}")
            return ShardSampleSource(shards, split=split, max_samples=max_n)
        if kind == "sunrgbd":
            root = self._profile.paths.data_root.parent / "sunrgbd"
            return SunRgbdSampleSource(root, max_frames=max_n or 50)
        raise ValueError(f"unknown sample source {kind!r}")


@dataclass
class EvalRow:
    detector: str
    category: str
    split: str
    source: str
    severity: float
    lifter: str
    iou_2d: float
    z_err_cm: float
    iou_3d: float
    success: bool


@dataclass
class EvalReport:
    rows: list[EvalRow] = field(default_factory=list)
    aggregates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [asdict(r) for r in self.rows],
            "aggregates": self.aggregates,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Grounding eval report",
            "",
            "| detector | source | lifter | severity≥ | acc@0.5 | median IoU₂d | "
            "median Z err (cm) | median 3D IoU | n |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for key, stats in sorted(self.aggregates.items()):
            lines.append(
                f"| {stats.get('detector', '')} | {stats.get('source', '')} | "
                f"{stats.get('lifter', key)} | {stats.get('severity_min', '')} | "
                f"{stats.get('grounding_acc@0.5', float('nan')):.3f} | "
                f"{stats.get('median_iou_2d', float('nan')):.3f} | "
                f"{stats.get('median_z_err_cm', float('nan')):.2f} | "
                f"{stats.get('median_iou_3d', float('nan')):.3f} | "
                f"{stats.get('n', 0)} |"
            )
        return "\n".join(lines) + "\n"


class GroundingEvaluator:
    def __init__(
        self,
        detector: Detector,
        lifter: Any,
        source: SampleSource,
        lifter_name: str = "full",
    ):
        self.detector = detector
        self.lifter = lifter
        self.source = source
        self.lifter_name = lifter_name

    def evaluate(self) -> EvalReport:
        rows: list[EvalRow] = []
        detector_name = getattr(self.detector, "name", "unknown")
        source_name = getattr(self.source, "name", "unknown")
        for s in self.source:
            pred_bbox = self.detector.detect(s.rgb, s.query, s.meta)
            if pred_bbox is None:
                rows.append(
                    EvalRow(
                        detector=detector_name,
                        category=s.category,
                        split=s.split,
                        source=source_name,
                        severity=s.severity,
                        lifter=self.lifter_name,
                        iou_2d=0.0,
                        z_err_cm=float("nan"),
                        iou_3d=0.0,
                        success=False,
                    )
                )
                continue

            iou2 = bbox_iou_2d(pred_bbox, s.gt_bbox)
            lift = self.lifter.lift(s.depth, s.intrinsics, pred_bbox)
            if lift is None:
                rows.append(
                    EvalRow(
                        detector=detector_name,
                        category=s.category,
                        split=s.split,
                        source=source_name,
                        severity=s.severity,
                        lifter=self.lifter_name,
                        iou_2d=iou2,
                        z_err_cm=float("nan"),
                        iou_3d=0.0,
                        success=False,
                    )
                )
                continue

            gt_z = float(0.5 * (s.gt_aabb[:, 2].min() + s.gt_aabb[:, 2].max()))
            rows.append(
                EvalRow(
                    detector=detector_name,
                    category=s.category,
                    split=s.split,
                    source=source_name,
                    severity=s.severity,
                    lifter=self.lifter_name,
                    iou_2d=iou2,
                    z_err_cm=z_error_cm(lift.z_est, gt_z),
                    iou_3d=aabb_iou_3d(lift.vertices, s.gt_aabb),
                    success=True,
                )
            )

        report = EvalReport(rows=rows)
        report.aggregates = self._aggregate(rows)
        return report

    @staticmethod
    def _aggregate(rows: list[EvalRow]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not rows:
            return out
        for sev_min in (0.0, 0.5):
            keys: dict[tuple[str, str, str], list[EvalRow]] = {}
            for r in rows:
                if r.severity < sev_min:
                    continue
                gkey = (r.detector, r.source, r.lifter)
                keys.setdefault(gkey, []).append(r)
            for (detector, source, lifter), group in keys.items():
                key = f"{detector}|{source}|{lifter}_sev>={sev_min}"
                successes = [r for r in group if r.success]
                out[key] = {
                    "detector": detector,
                    "source": source,
                    "lifter": lifter,
                    "severity_min": sev_min,
                    "grounding_acc@0.5": float(np.mean([r.iou_2d >= 0.5 for r in group])),
                    "median_iou_2d": float(np.nanmedian([r.iou_2d for r in group])),
                    "median_z_err_cm": float(
                        np.nanmedian([r.z_err_cm for r in successes]) if successes else float("nan")
                    ),
                    "median_iou_3d": float(
                        np.nanmedian([r.iou_3d for r in successes]) if successes else float("nan")
                    ),
                    "n": len(group),
                }
        return out


class EvalJob(Trainer):
    def run(self) -> TrainerOutcome:
        cfg = dict(self.spec.config)
        lifting = str(cfg.get("lifting", "both"))

        detector = DetectorFactory(self.profile).create(cfg)
        source = SampleSourceFactory(self.profile).create(cfg)
        lifters = self._build_lifters(lifting)

        all_rows: list[EvalRow] = []
        aggregates: dict[str, Any] = {}
        for name, lifter in lifters:
            if self.stop_event.is_set():
                return TrainerOutcome.PREEMPTED
            evaluator = GroundingEvaluator(
                detector=detector,
                lifter=lifter,
                source=source,
                lifter_name=name,
            )
            report = evaluator.evaluate()
            all_rows.extend(report.rows)
            aggregates.update(report.aggregates)

        final = EvalReport(rows=all_rows, aggregates=aggregates)
        out_dir = self.checkpoints.root
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(final.to_dict(), indent=2))
        (out_dir / "report.md").write_text(final.to_markdown())
        log.info("wrote eval report → %s", out_dir / "report.json")
        self._record_result(cfg, aggregates)

        def _mark_done(d: Path) -> None:
            (d / "done.json").write_text(json.dumps({"ok": True}))

        self.checkpoints.save(1, _mark_done)
        return TrainerOutcome.COMPLETED

    def _record_result(self, cfg: dict[str, Any], aggregates: dict[str, Any]) -> None:
        try:
            from training.results import ExperimentResult, ResultStore

            hypothesis = str(cfg.get("hypothesis") or self._infer_hypothesis(cfg))
            metrics: dict[str, float] = {}
            for key, stats in aggregates.items():
                for stat_name in (
                    "grounding_acc@0.5",
                    "median_iou_2d",
                    "median_z_err_cm",
                    "median_iou_3d",
                ):
                    value = stats.get(stat_name)
                    if value is not None:
                        metrics[f"{key}__{stat_name}"] = float(value)

            store = ResultStore(self.profile.paths.ckpt_root / "results.db")
            store.record(
                ExperimentResult(
                    experiment=self.spec.name,
                    hypothesis=hypothesis,
                    kind="eval",
                    config=cfg,
                    metrics=metrics,
                    status="done",
                )
            )
        except Exception:
            log.exception("failed to record eval result into ResultStore (non-fatal)")

    @staticmethod
    def _infer_hypothesis(cfg: dict[str, Any]) -> str:
        if cfg.get("source") == "sunrgbd":
            return "H7"
        if cfg.get("detector") == "gt":
            return "H2"
        return "H1"

    def _build_lifters(self, lifting: str) -> list[tuple[str, Any]]:
        if lifting == "naive":
            return [("naive", NaiveLifter())]
        if lifting == "full":
            return [("full", BBoxLifter())]
        if lifting == "both":
            return [("naive", NaiveLifter()), ("full", BBoxLifter())]
        raise ValueError(f"unknown lifting mode {lifting!r}")
