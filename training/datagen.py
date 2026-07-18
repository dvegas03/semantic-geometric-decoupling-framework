# Mini-corpus generator: synthetic RGB-D + GT labels as WebDataset shards.

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import open3d as o3d

from engine_grounder.spatial.lifting import BBox2D, BBoxLifter
from training.base import Trainer, TrainerOutcome

log = logging.getLogger("training.datagen")

GLOBAL_SEED = 0

_PRIMITIVE_CATEGORIES: tuple[str, ...] = (
    "box",
    "sphere",
    "cylinder",
    "cone",
    "torus",
    "mug",
    "bottle",
    "bowl",
    "plate",
    "table",
    "chair",
    "lamp",
    "book",
    "phone",
    "keyboard",
    "monitor",
    "plant",
    "pillow",
    "drill",
    "screwdriver",
)


@dataclass(frozen=True)
class CorruptionParams:
    void_ratio: float
    noise_std_1m: float
    edge_bleed_px: int
    seed: int


@dataclass(frozen=True)
class PlacedObject:
    category: str
    mesh: o3d.geometry.TriangleMesh
    color: tuple[float, float, float]


@dataclass(frozen=True)
class FrameRecord:
    rgb: np.ndarray
    clean_depth: np.ndarray
    corrupted_depth: np.ndarray
    instance_map: np.ndarray
    category: str
    query_text: str
    bbox_2d: BBox2D
    aabb_3d: np.ndarray
    aabb_3d_lifted_clean: np.ndarray | None
    intrinsics: tuple[float, float, float, float]
    corruption: CorruptionParams
    split: str


class AssetLibrary(ABC):
    @abstractmethod
    def categories(self) -> Sequence[str]: ...

    @abstractmethod
    def sample_mesh(self, category: str, rng: np.random.Generator) -> o3d.geometry.TriangleMesh: ...


class PrimitiveAssetLibrary(AssetLibrary):
    def __init__(self, n_categories: int = 20):
        cats = list(_PRIMITIVE_CATEGORIES[:n_categories])
        if len(cats) < n_categories:
            raise ValueError(
                f"requested {n_categories} categories but only "
                f"{len(_PRIMITIVE_CATEGORIES)} primitives are defined"
            )
        self._categories = cats

    def categories(self) -> Sequence[str]:
        return tuple(self._categories)

    def sample_mesh(self, category: str, rng: np.random.Generator) -> o3d.geometry.TriangleMesh:
        if category not in self._categories:
            raise KeyError(f"unknown category {category!r}")
        scale = float(rng.uniform(0.08, 0.22))
        mesh = self._build(category, scale, rng)
        mesh.compute_vertex_normals()
        return mesh

    def _build(
        self, category: str, scale: float, rng: np.random.Generator
    ) -> o3d.geometry.TriangleMesh:
        builders = {
            "box": lambda: o3d.geometry.TriangleMesh.create_box(
                width=scale * float(rng.uniform(0.7, 1.3)),
                height=scale * float(rng.uniform(0.7, 1.3)),
                depth=scale * float(rng.uniform(0.7, 1.3)),
            ),
            "sphere": lambda: o3d.geometry.TriangleMesh.create_sphere(
                radius=scale * 0.5, resolution=20
            ),
            "cylinder": lambda: o3d.geometry.TriangleMesh.create_cylinder(
                radius=scale * 0.35, height=scale * 1.2, resolution=24
            ),
            "cone": lambda: o3d.geometry.TriangleMesh.create_cone(
                radius=scale * 0.4, height=scale * 1.2, resolution=24
            ),
            "torus": lambda: o3d.geometry.TriangleMesh.create_torus(
                torus_radius=scale * 0.4, tube_radius=scale * 0.12, radial_resolution=24
            ),
            "mug": lambda: self._compound_mug(scale, rng),
            "bottle": lambda: o3d.geometry.TriangleMesh.create_cylinder(
                radius=scale * 0.22, height=scale * 1.6, resolution=20
            ),
            "bowl": lambda: self._compound_bowl(scale),
            "plate": lambda: o3d.geometry.TriangleMesh.create_cylinder(
                radius=scale * 0.7, height=scale * 0.08, resolution=32
            ),
            "table": lambda: o3d.geometry.TriangleMesh.create_box(
                width=scale * 2.0, height=scale * 0.12, depth=scale * 1.2
            ),
            "chair": lambda: self._compound_chair(scale),
            "lamp": lambda: self._compound_lamp(scale),
            "book": lambda: o3d.geometry.TriangleMesh.create_box(
                width=scale * 0.7, height=scale * 0.12, depth=scale * 1.0
            ),
            "phone": lambda: o3d.geometry.TriangleMesh.create_box(
                width=scale * 0.35, height=scale * 0.08, depth=scale * 0.7
            ),
            "keyboard": lambda: o3d.geometry.TriangleMesh.create_box(
                width=scale * 1.6, height=scale * 0.08, depth=scale * 0.55
            ),
            "monitor": lambda: o3d.geometry.TriangleMesh.create_box(
                width=scale * 1.4, height=scale * 0.9, depth=scale * 0.12
            ),
            "plant": lambda: self._compound_plant(scale),
            "pillow": lambda: o3d.geometry.TriangleMesh.create_sphere(
                radius=scale * 0.45, resolution=16
            ).scale(1.4, np.zeros(3)),
            "drill": lambda: self._compound_drill(scale),
            "screwdriver": lambda: o3d.geometry.TriangleMesh.create_cylinder(
                radius=scale * 0.08, height=scale * 1.4, resolution=12
            ),
        }
        mesh = builders[category]()
        mesh.translate(-mesh.get_center())
        aabb = mesh.get_axis_aligned_bounding_box()
        mesh.translate((0.0, -float(aabb.min_bound[1]), 0.0))
        return mesh

    @staticmethod
    def _compound_mug(scale: float, rng: np.random.Generator) -> o3d.geometry.TriangleMesh:
        body = o3d.geometry.TriangleMesh.create_cylinder(
            radius=scale * 0.3, height=scale * 0.7, resolution=20
        )
        handle = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=scale * 0.22,
            tube_radius=scale * 0.05,
            radial_resolution=16,
        )
        handle.rotate(handle.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)), center=(0, 0, 0))
        handle.translate((scale * 0.3, 0.0, 0.0))
        return body + handle

    @staticmethod
    def _compound_bowl(scale: float) -> o3d.geometry.TriangleMesh:
        outer = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.5, resolution=20)
        verts = np.asarray(outer.vertices)
        verts[:, 1] *= 0.45
        outer.vertices = o3d.utility.Vector3dVector(verts)
        return outer

    @staticmethod
    def _compound_chair(scale: float) -> o3d.geometry.TriangleMesh:
        seat = o3d.geometry.TriangleMesh.create_box(width=scale, height=scale * 0.12, depth=scale)
        back = o3d.geometry.TriangleMesh.create_box(
            width=scale, height=scale * 0.9, depth=scale * 0.1
        )
        back.translate((0.0, scale * 0.5, -scale * 0.45))
        return seat + back

    @staticmethod
    def _compound_lamp(scale: float) -> o3d.geometry.TriangleMesh:
        stem = o3d.geometry.TriangleMesh.create_cylinder(
            radius=scale * 0.06, height=scale * 1.2, resolution=12
        )
        shade = o3d.geometry.TriangleMesh.create_cone(
            radius=scale * 0.35, height=scale * 0.4, resolution=16
        )
        shade.translate((0.0, scale * 0.7, 0.0))
        return stem + shade

    @staticmethod
    def _compound_plant(scale: float) -> o3d.geometry.TriangleMesh:
        pot = o3d.geometry.TriangleMesh.create_cylinder(
            radius=scale * 0.25, height=scale * 0.4, resolution=16
        )
        foliage = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.4, resolution=16)
        foliage.translate((0.0, scale * 0.55, 0.0))
        return pot + foliage

    @staticmethod
    def _compound_drill(scale: float) -> o3d.geometry.TriangleMesh:
        body = o3d.geometry.TriangleMesh.create_box(
            width=scale * 0.35, height=scale * 0.35, depth=scale * 1.1
        )
        bit = o3d.geometry.TriangleMesh.create_cylinder(
            radius=scale * 0.06, height=scale * 0.5, resolution=10
        )
        bit.rotate(bit.get_rotation_matrix_from_xyz((0, 0, np.pi / 2)), center=(0, 0, 0))
        bit.translate((0.0, 0.0, scale * 0.7))
        return body + bit


class CategorySplit:
    def __init__(self, categories: Sequence[str], holdout_fraction: float):
        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
        self.categories = tuple(categories)
        self.holdout_fraction = holdout_fraction
        n_holdout = max(1, round(holdout_fraction * len(self.categories)))
        ranked = sorted(
            self.categories,
            key=lambda c: hashlib.sha1(c.encode("utf-8")).hexdigest(),
        )
        self.heldout = frozenset(ranked[:n_holdout])

    def is_heldout(self, category: str) -> bool:
        return category in self.heldout

    def split_name(self, category: str) -> str:
        return "heldout" if self.is_heldout(category) else "train"


class SceneSampler:
    def __init__(self, library: AssetLibrary, max_objects: int = 10):
        self.library = library
        self.max_objects = max_objects

    def sample(self, rng: np.random.Generator) -> list[PlacedObject]:
        n = int(rng.integers(3, self.max_objects + 1))
        cats = list(self.library.categories())
        placed: list[PlacedObject] = []
        centres: list[np.ndarray] = []
        for _ in range(n):
            category = str(rng.choice(cats))
            mesh = self.library.sample_mesh(category, rng)
            for _attempt in range(40):
                x = float(rng.uniform(-0.7, 0.7))
                z = float(rng.uniform(-0.3, 0.9))
                yaw = float(rng.uniform(0, 2 * np.pi))
                candidate = np.array([x, 0.0, z])
                if any(np.linalg.norm(candidate[[0, 2]] - c[[0, 2]]) < 0.28 for c in centres):
                    continue
                posed = o3d.geometry.TriangleMesh(mesh)
                posed.rotate(
                    posed.get_rotation_matrix_from_xyz((0.0, yaw, 0.0)),
                    center=posed.get_center(),
                )
                posed.translate((x, 0.0, z))
                colour_arr = rng.uniform(0.2, 0.9, size=3)
                colour = (float(colour_arr[0]), float(colour_arr[1]), float(colour_arr[2]))
                posed.paint_uniform_color(colour)
                placed.append(PlacedObject(category=category, mesh=posed, color=colour))
                centres.append(candidate)
                break
        if not placed:
            category = str(rng.choice(cats))
            mesh = self.library.sample_mesh(category, rng)
            colour = (0.6, 0.5, 0.4)
            mesh.paint_uniform_color(colour)
            placed.append(PlacedObject(category=category, mesh=mesh, color=colour))
        return placed


class RayCastRenderer:
    def __init__(
        self,
        image_hw: tuple[int, int],
        fov_deg: float = 60.0,
        cam_eye: tuple[float, float, float] = (0.0, 0.55, -1.15),
        cam_lookat: tuple[float, float, float] = (0.0, 0.15, 0.2),
    ):
        self.h, self.w = image_hw
        self.fov_deg = fov_deg
        self.cam_eye: np.ndarray = np.asarray(cam_eye, dtype=np.float64)
        self.cam_lookat: np.ndarray = np.asarray(cam_lookat, dtype=np.float64)
        fx = 0.5 * self.w / np.tan(np.deg2rad(fov_deg) / 2.0)
        fy = fx
        cx, cy = self.w / 2.0, self.h / 2.0
        self.intrinsics: tuple[float, float, float, float] = (fx, fy, cx, cy)

    def render(self, objects: Sequence[PlacedObject]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scene = o3d.t.geometry.RaycastingScene()
        instance_ids: list[int] = []
        colours: list[np.ndarray] = []
        for i, obj in enumerate(objects):
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh)
            scene.add_triangles(tmesh)
            instance_ids.append(i + 1)
            colours.append(np.asarray(obj.color, dtype=np.float32))

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=self.fov_deg,
            center=self.cam_lookat,
            eye=self.cam_eye,
            up=np.array([0.0, 1.0, 0.0]),
            width_px=self.w,
            height_px=self.h,
        )
        ans = scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy().reshape(self.h, self.w)
        geom_ids = ans["geometry_ids"].numpy().reshape(self.h, self.w)
        normals = ans["primitive_normals"].numpy().reshape(self.h, self.w, 3)

        depth = np.where(np.isfinite(t_hit), t_hit.astype(np.float32), 0.0)
        instance: np.ndarray = np.zeros((self.h, self.w), dtype=np.int32)
        rgb: np.ndarray = np.full((self.h, self.w, 3), 0.85, dtype=np.float32)

        light_dir = np.array([0.3, 0.8, -0.4], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        for local_idx, inst_id in enumerate(instance_ids):
            mask = geom_ids == local_idx
            if not mask.any():
                continue
            instance[mask] = inst_id
            n = normals[mask]
            n_norm = np.linalg.norm(n, axis=1, keepdims=True)
            n = np.divide(n, np.maximum(n_norm, 1e-8))
            lambert = np.clip(n @ light_dir, 0.05, 1.0)
            rgb[mask] = colours[local_idx][None, :] * lambert[:, None]

        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb_u8, depth, instance


class QuestDepthCorrupter:
    def corrupt(self, clean_depth: np.ndarray, params: CorruptionParams) -> np.ndarray:
        rng = np.random.default_rng(params.seed)
        depth = clean_depth.copy()

        valid = depth > 0.0
        depth[valid] += rng.normal(0.0, params.noise_std_1m, int(valid.sum())) * depth[valid] ** 2
        np.clip(depth, 0.0, None, out=depth)

        bleed = max(int(params.edge_bleed_px), 1)
        grad = np.maximum(
            np.abs(np.gradient(clean_depth, axis=0)),
            np.abs(np.gradient(clean_depth, axis=1)),
        )
        edge_void = cv2.dilate(
            (grad > 0.05).astype(np.uint8),
            np.ones((bleed, bleed), np.uint8),
        ).astype(bool)

        target_voids = int(depth.size * params.void_ratio)
        uniform_budget = max(target_voids - int(edge_void.sum()), 0)
        flat = rng.choice(depth.size, uniform_budget, replace=False)
        void = edge_void.copy().reshape(-1)
        void[flat] = True

        depth.reshape(-1)[void] = 0.0
        return depth


class QueryGenerator:
    _TEMPLATES = (
        "the {category}",
        "a {category}",
        "the {category} on the table",
        "the {category} near the centre",
        "find the {category}",
    )
    _RELATIONAL = (
        "the {category} left of the {other}",
        "the {category} right of the {other}",
        "the {category} in front of the {other}",
        "the {category} behind the {other}",
    )

    def generate(
        self,
        target: PlacedObject,
        others: Sequence[PlacedObject],
        rng: np.random.Generator,
    ) -> str:
        if others and float(rng.random()) < 0.4:
            other = others[int(rng.integers(0, len(others)))]
            tmpl = str(rng.choice(self._RELATIONAL))
            return tmpl.format(category=target.category, other=other.category)
        tmpl = str(rng.choice(self._TEMPLATES))
        return tmpl.format(category=target.category)


class FrameComposer:
    def __init__(
        self,
        library: AssetLibrary,
        split: CategorySplit,
        sampler: SceneSampler,
        renderer: RayCastRenderer,
        corrupter: QuestDepthCorrupter,
        queries: QueryGenerator,
        lifter: BBoxLifter | None = None,
    ):
        self.library = library
        self.split = split
        self.sampler = sampler
        self.renderer = renderer
        self.corrupter = corrupter
        self.queries = queries
        self.lifter = lifter if lifter is not None else BBoxLifter()

    def compose(self, frame_idx: int, rng: np.random.Generator) -> FrameRecord:
        objects = self.sampler.sample(rng)
        rgb, clean_depth, instance = self.renderer.render(objects)

        visible = [(i, obj) for i, obj in enumerate(objects) if (instance == (i + 1)).any()]
        if not visible:
            target_idx, target = 0, objects[0]
        else:
            target_idx, target = visible[int(rng.integers(0, len(visible)))]

        inst_id = target_idx + 1
        mask = instance == inst_id
        if mask.any():
            ys, xs = np.where(mask)
            bbox = BBox2D(
                x_min=int(xs.min()),
                y_min=int(ys.min()),
                x_max=int(xs.max()) + 1,
                y_max=int(ys.max()) + 1,
            )
        else:
            bbox = BBox2D(0, 0, 8, 8)

        aabb_3d = self._mesh_aabb_camera(target.mesh)

        lift = self.lifter.lift(clean_depth, self.renderer.intrinsics, bbox)
        aabb_lifted = None if lift is None else lift.vertices.copy()

        params = CorruptionParams(
            void_ratio=float(rng.uniform(0.3, 0.8)),
            noise_std_1m=float(rng.uniform(0.01, 0.04)),
            edge_bleed_px=int(rng.integers(2, 6)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        corrupted = self.corrupter.corrupt(clean_depth, params)

        others = [o for j, o in enumerate(objects) if j != target_idx]
        query = self.queries.generate(target, others, rng)

        return FrameRecord(
            rgb=rgb,
            clean_depth=clean_depth,
            corrupted_depth=corrupted,
            instance_map=instance,
            category=target.category,
            query_text=query,
            bbox_2d=bbox,
            aabb_3d=aabb_3d,
            aabb_3d_lifted_clean=aabb_lifted,
            intrinsics=self.renderer.intrinsics,
            corruption=params,
            split=self.split.split_name(target.category),
        )

    def _mesh_aabb_camera(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        eye = self.renderer.cam_eye
        center = self.renderer.cam_lookat
        up = np.array([0.0, 1.0, 0.0])
        forward = center - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        R = np.stack([-right, true_up, forward], axis=0)
        cam = (R @ (verts - eye).T).T
        mins, maxs = cam.min(axis=0), cam.max(axis=0)
        return cast(
            np.ndarray,
            np.array(
                [
                    [x, y, z]
                    for z in (mins[2], maxs[2])
                    for y in (mins[1], maxs[1])
                    for x in (mins[0], maxs[0])
                ],
                dtype=np.float64,
            ),
        )


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _png_bytes(rgb: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("png encode failed")
    return encoded.tobytes()


class CorpusPreviewWriter:
    def write(self, shards: Sequence[Path], out_path: Path, n_frames: int = 5) -> None:
        import tarfile

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        frames: list[dict[str, object]] = []
        for shard in shards:
            if len(frames) >= n_frames:
                break
            with tarfile.open(shard, "r") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                stems = sorted({n.split(".", 1)[0] for n in members})
                for stem in stems:
                    if len(frames) >= n_frames:
                        break
                    meta_m = members.get(f"{stem}.meta.json")
                    rgb_m = members.get(f"{stem}.rgb.png")
                    corr_m = members.get(f"{stem}.corrupted_depth.npy")
                    clean_m = members.get(f"{stem}.clean_depth.npy")
                    if not (meta_m and rgb_m and corr_m):
                        continue
                    meta = json.loads(tar.extractfile(meta_m).read())  # type: ignore[union-attr]
                    rgb_bgr = cv2.imdecode(
                        np.frombuffer(tar.extractfile(rgb_m).read(), np.uint8),  # type: ignore[union-attr]
                        cv2.IMREAD_COLOR,
                    )
                    if rgb_bgr is None:
                        continue
                    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    corr = np.load(io.BytesIO(tar.extractfile(corr_m).read()))  # type: ignore[union-attr]
                    clean = (
                        np.load(io.BytesIO(tar.extractfile(clean_m).read()))  # type: ignore[union-attr]
                        if clean_m is not None
                        else corr
                    )
                    frames.append(
                        {
                            "rgb": rgb,
                            "corr": corr,
                            "clean": clean,
                            "meta": meta,
                        }
                    )

        if not frames:
            raise RuntimeError("no frames available for corpus preview")

        fig, axes = plt.subplots(len(frames), 3, figsize=(9, 3 * len(frames)))
        if len(frames) == 1:
            axes = np.array([axes])
        for row, fr in enumerate(frames):
            meta = fr["meta"]  # type: ignore[assignment]
            assert isinstance(meta, dict)
            bbox = meta.get("bbox_2d", [0, 0, 0, 0])
            title = f"{meta.get('category', '')} | {meta.get('query_text', '')}"
            ax0, ax1, ax2 = axes[row]
            ax0.imshow(fr["rgb"])  # type: ignore[arg-type]
            ax0.add_patch(
                Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.5,
                )
            )
            ax0.set_title(title, fontsize=8)
            ax0.axis("off")
            ax1.imshow(fr["corr"], cmap="viridis")
            ax1.set_title("corrupted depth", fontsize=8)
            ax1.axis("off")
            ax2.imshow(fr["clean"], cmap="viridis")
            ax2.set_title("clean depth", fontsize=8)
            ax2.axis("off")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


class DatagenJob(Trainer):
    def run(self) -> TrainerOutcome:
        composer = self._build_composer()
        total_shards = -(-self.profile.corpus.frames // self.profile.corpus.shard_size)

        start_shard = 0
        latest = self.checkpoints.latest()
        if latest is not None:
            start_shard = json.loads((latest / "progress.json").read_text())["next_shard"]
            log.info("resuming datagen from shard %d", start_shard)

        out_dir = self.profile.paths.data_root
        out_dir.mkdir(parents=True, exist_ok=True)
        for _stale in out_dir.glob("shard-*.tar.*.tmp"):
            if (out_dir / (_stale.name.split(".tar.")[0] + ".tar")).exists():
                _stale.unlink(missing_ok=True)
        for shard_idx in range(start_shard, total_shards):
            if self.stop_event.is_set():
                return TrainerOutcome.PREEMPTED
            self._write_shard(composer, out_dir, shard_idx)

            def _write_progress(d: Path, n: int = shard_idx + 1) -> None:
                (d / "progress.json").write_text(json.dumps({"next_shard": n}))

            self.checkpoints.save(shard_idx + 1, _write_progress)
            log.info("wrote shard %d / %d", shard_idx + 1, total_shards)

        preview_path = out_dir.parent / "corpus_preview.png"
        try:
            CorpusPreviewWriter().write(
                sorted(out_dir.glob("shard-*.tar")),
                preview_path,
                n_frames=5,
            )
            log.info("wrote corpus preview → %s", preview_path)
        except Exception:
            log.exception("corpus preview failed (non-fatal)")
        return TrainerOutcome.COMPLETED

    def _build_composer(self) -> FrameComposer:
        n_cat = self.profile.corpus.categories
        library = PrimitiveAssetLibrary(n_categories=n_cat)
        split = CategorySplit(library.categories(), self.profile.corpus.holdout_category_fraction)
        hw = self.profile.corpus.image_hw
        return FrameComposer(
            library=library,
            split=split,
            sampler=SceneSampler(library),
            renderer=RayCastRenderer((int(hw[0]), int(hw[1]))),
            corrupter=QuestDepthCorrupter(),
            queries=QueryGenerator(),
        )

    def _write_shard(self, composer: FrameComposer, out_dir: Path, shard_idx: int) -> None:
        import tarfile

        shard_size = self.profile.corpus.shard_size
        frames = self.profile.corpus.frames
        start = shard_idx * shard_size
        end = min(start + shard_size, frames)

        final = out_dir / f"shard-{shard_idx:05d}.tar"
        # Unique per process: a job reclaimed after the queue TTL can overlap a
        # still-dying predecessor writing the same shard. Both os.replace onto the
        # same final atomically, and shard bytes are deterministic per shard_idx,
        # so the last writer wins with identical content — no tmp stomping, no crash.
        tmp = out_dir / f"shard-{shard_idx:05d}.tar.{os.getpid()}.tmp"
        if tmp.exists():
            tmp.unlink()

        rng = np.random.default_rng(GLOBAL_SEED + shard_idx)
        with tarfile.open(tmp, "w") as tar:
            for _local_i, frame_idx in enumerate(range(start, end)):
                frame_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                record = composer.compose(frame_idx, frame_rng)
                meta = {
                    "category": record.category,
                    "query_text": record.query_text,
                    "bbox_2d": [
                        record.bbox_2d.x_min,
                        record.bbox_2d.y_min,
                        record.bbox_2d.x_max,
                        record.bbox_2d.y_max,
                    ],
                    "aabb_3d": record.aabb_3d.tolist(),
                    "aabb_3d_lifted_clean": (
                        None
                        if record.aabb_3d_lifted_clean is None
                        else record.aabb_3d_lifted_clean.tolist()
                    ),
                    "intrinsics": list(record.intrinsics),
                    "corruption_params": asdict(record.corruption),
                    "split": record.split,
                    "frame_idx": frame_idx,
                }
                key = f"{frame_idx:06d}"
                members = {
                    f"{key}.rgb.png": _png_bytes(record.rgb),
                    f"{key}.clean_depth.npy": _npy_bytes(record.clean_depth),
                    f"{key}.corrupted_depth.npy": _npy_bytes(record.corrupted_depth),
                    f"{key}.meta.json": json.dumps(meta, sort_keys=True).encode("utf-8"),
                }
                for name, payload in members.items():
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    tar.addfile(info, io.BytesIO(payload))
        os.replace(tmp, final)
