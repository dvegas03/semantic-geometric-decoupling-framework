# PointNet / ModelNet40 trainer (H6 arm 1) — converges on the local 4070.

from __future__ import annotations

import dataclasses
import logging
import urllib.request
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from engine_grounder.perception.shape_encoder import PointNetEncoder
from training.base import Trainer, TrainerOutcome
from training.train_lora import MetricsLog

log = logging.getLogger("training.train_pointnet")

_MODELNET40_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
_N_POINTS = 1024


@dataclass(frozen=True)
class PointNetConfig:
    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 5

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError(f"epochs/batch_size must be positive, got {self}")
        if not 0 < self.warmup_epochs < self.epochs:
            raise ValueError(f"warmup_epochs must be in (0, epochs), got {self}")

    @classmethod
    def from_spec(cls, config: dict[str, object]) -> PointNetConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(config) - known
        if unknown:
            raise ValueError(f"unknown pointnet config keys: {sorted(unknown)}")
        kwargs = {}
        for k, v in config.items():
            default = getattr(cls, k)
            kwargs[k] = type(default)(v)
        return cls(**kwargs)


class BestModelTracker:
    def __init__(self) -> None:
        self.best_acc = 0.0
        self._best_encoder_state: OrderedDict[str, torch.Tensor] | None = None

    def observe(self, model: PointNetClassifier, acc: float) -> bool:
        if acc <= self.best_acc:
            return False
        self.best_acc = acc
        self._best_encoder_state = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in model.encoder_state_dict().items()
        )
        return True

    def export(self, export_dir: Path) -> Path:
        if self._best_encoder_state is None:
            raise RuntimeError("export() before any observe() — no best model recorded")
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / "pointnet_modelnet40.pt"
        torch.save(self._best_encoder_state, path)
        return path


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 40, embed_dim: int = 256):
        super().__init__()
        self.encoder = PointNetEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.encoder(x)
        return self.head(embedding)  # type: ignore[no-any-return]

    def encoder_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        prefix = "encoder."
        return OrderedDict(
            (k[len(prefix) :], v) for k, v in self.state_dict().items() if k.startswith(prefix)
        )


class ModelNet40Dataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        n_points: int = _N_POINTS,
        augment: bool = False,
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be train|test, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.n_points = n_points
        self.augment = augment
        self.seed = seed
        self.cache_path = self.root / f"modelnet40_{split}_{n_points}.npz"
        self._ensure_data()
        payload = np.load(self.cache_path, allow_pickle=False)
        self.points = payload["points"].astype(np.float32)
        self.labels = payload["labels"].astype(np.int64)
        self.class_names: list[str] = list(payload["class_names"])

    def _ensure_data(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            return
        raw_dir = self.root / "ModelNet40"
        zip_path = self.root / "ModelNet40.zip"
        if not raw_dir.exists():
            if not zip_path.exists():
                log.info("downloading ModelNet40 to %s", zip_path)
                urllib.request.urlretrieve(_MODELNET40_URL, zip_path)
            log.info("extracting %s", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.root)
        self._build_cache(raw_dir)

    def _build_cache(self, raw_dir: Path) -> None:
        class_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
        class_names = [p.name for p in class_dirs]
        points_list: list[np.ndarray] = []
        labels: list[int] = []
        skipped = 0
        for label, class_dir in enumerate(class_dirs):
            split_dir = class_dir / self.split
            if not split_dir.exists():
                continue
            for off_path in sorted(split_dir.glob("*.off")):
                mesh = self._read_off_mesh(off_path)
                if mesh is None or not mesh.has_triangles() or len(mesh.vertices) == 0:
                    skipped += 1
                    continue
                mesh.compute_vertex_normals()
                pcd = mesh.sample_points_uniformly(number_of_points=self.n_points)
                pts = np.asarray(pcd.points, dtype=np.float32)
                pts = self._normalize(pts)
                points_list.append(pts)
                labels.append(label)
        if not points_list:
            raise RuntimeError(f"no ModelNet40 meshes found under {raw_dir}/{self.split}")
        if skipped:
            log.warning("skipped %d unreadable OFF meshes in %s split", skipped, self.split)
        np.savez_compressed(
            self.cache_path,
            points=np.stack(points_list),
            labels=np.asarray(labels, dtype=np.int64),
            class_names=np.asarray(class_names),
        )
        log.info("cached %d %s samples → %s", len(points_list), self.split, self.cache_path)

    @staticmethod
    def _read_off_mesh(path: Path) -> o3d.geometry.TriangleMesh | None:
        import tempfile

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            text = path.read_text(encoding="utf-8", errors="replace")
            body = text.lstrip("\ufeff \t")
            if body.startswith("OFF") and len(body) > 3 and body[3] not in "\r\n":
                rest = body[3:].lstrip()
                fixed = "OFF\n" + rest
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".off", delete=False, encoding="utf-8"
                ) as tmp:
                    tmp.write(fixed)
                    tmp_path = tmp.name
                try:
                    mesh = o3d.io.read_triangle_mesh(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                mesh = o3d.io.read_triangle_mesh(str(path))
            if not mesh.has_triangles() or len(mesh.vertices) == 0:
                return None
            return mesh

    @staticmethod
    def _normalize(pts: np.ndarray) -> np.ndarray:
        pts = pts - pts.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 0:
            pts = pts / scale
        return pts.astype(np.float32)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        pts = self.points[idx].copy()
        if self.augment:
            rng = np.random.default_rng()
            theta = float(rng.uniform(0, 2 * np.pi))
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pts = pts @ rot.T
            pts += rng.normal(0.0, 0.01, size=pts.shape).astype(np.float32)
            pts *= float(rng.uniform(0.8, 1.2))
        x = torch.from_numpy(pts.T)
        return x, int(self.labels[idx])


class PointNetTrainer(Trainer):
    def run(self) -> TrainerOutcome:
        cfg = PointNetConfig.from_spec(self.spec.config)
        device = torch.device(self.profile.device if torch.cuda.is_available() else "cpu")

        probe = None
        try:
            from training.telemetry import GpuTelemetryProbe

            probe = GpuTelemetryProbe(device)
        except ImportError:
            log.warning("training.telemetry unavailable — GPU telemetry disabled")

        data_root = self.profile.paths.data_root.parent / "modelnet40"
        train_ds = ModelNet40Dataset(data_root, split="train", augment=True)
        test_ds = ModelNet40Dataset(data_root, split="test", augment=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=min(2, self.profile.dataloader_workers),
            drop_last=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=min(2, self.profile.dataloader_workers),
        )

        model = PointNetClassifier(num_classes=40).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs - cfg.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs]
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        tracker = BestModelTracker()
        metrics = MetricsLog(self.checkpoints.root / "metrics.jsonl")

        start_epoch = 0
        latest = self.checkpoints.latest()
        if latest is not None:
            path = latest / "train_state.pt"
            try:
                state = torch.load(path, map_location=device, weights_only=False)
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                start_epoch = int(state["epoch"])
                tracker.best_acc = float(state.get("best_acc", 0.0))
                tracker._best_encoder_state = state.get("best_encoder_state")
                if "torch_rng" in state:
                    torch.set_rng_state(state["torch_rng"])
                if "cuda_rng" in state and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
                if "numpy_rng" in state:
                    np.random.set_state(state["numpy_rng"])
                log.info(
                    "resumed PointNet from epoch %d (best_acc=%.3f)",
                    start_epoch,
                    tracker.best_acc,
                )
            except Exception:
                log.exception(
                    "failed to restore PointNet train_state from %s — starting fresh",
                    path,
                )

        for epoch in range(start_epoch, cfg.epochs):
            if self.stop_event.is_set():
                self._save(model, optimizer, scheduler, tracker, epoch)
                return TrainerOutcome.PREEMPTED

            model.train()
            running = 0.0
            for x, y in train_loader:
                if self.stop_event.is_set():
                    self._save(model, optimizer, scheduler, tracker, epoch)
                    return TrainerOutcome.PREEMPTED
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running += float(loss.item())

            scheduler.step()
            acc = self._evaluate(model, test_loader, device)
            is_best = tracker.observe(model, acc)
            avg_loss = running / max(len(train_loader), 1)
            metrics.log(
                epoch=epoch + 1,
                loss=avg_loss,
                acc=acc,
                best_acc=tracker.best_acc,
                lr=scheduler.get_last_lr()[0],
                **(probe.sample() if probe is not None else {}),
            )
            log.info(
                "epoch %d/%d loss=%.4f acc=%.3f best=%.3f%s",
                epoch + 1,
                cfg.epochs,
                avg_loss,
                acc,
                tracker.best_acc,
                " (new best)" if is_best else "",
            )
            self._save(model, optimizer, scheduler, tracker, epoch + 1)

        export_path = tracker.export(self.checkpoints.root / "export")
        log.info("exported best encoder weights (acc=%.3f) → %s", tracker.best_acc, export_path)
        return TrainerOutcome.COMPLETED

    def _save(
        self,
        model: PointNetClassifier,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        tracker: BestModelTracker,
        epoch: int,
    ) -> None:
        def write(d: Path) -> None:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_acc": tracker.best_acc,
                    "best_encoder_state": tracker._best_encoder_state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    ),
                    "numpy_rng": np.random.get_state(),
                },
                d / "train_state.pt",
            )

        self.checkpoints.save(epoch, write)

    @staticmethod
    def _evaluate(model: PointNetClassifier, loader: DataLoader, device: torch.device) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        return correct / max(total, 1)
