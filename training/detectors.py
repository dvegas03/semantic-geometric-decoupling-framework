# Detector strategy hierarchy for grounding evaluation (H1/H8 arms).

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from engine_grounder.spatial.lifting import BBox2D
from training.profile import Profile


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, rgb: np.ndarray, query: str, meta: dict[str, Any]) -> BBox2D | None: ...


class GtDetector(Detector):
    name = "gt"

    def detect(self, rgb: np.ndarray, query: str, meta: dict[str, Any]) -> BBox2D | None:
        box = meta.get("bbox_2d")
        if not box or len(box) != 4:
            return None
        return BBox2D(int(box[0]), int(box[1]), int(box[2]), int(box[3]))


class VlmDetector(Detector):
    name = "vlm"

    def __init__(self, profile: Profile):
        from engine_grounder.perception.vlm_agent import VLMAgent

        self._agent = VLMAgent(model_name=profile.vlm_model, device=profile.device)
        self._agent.load_model()

    def detect(self, rgb: np.ndarray, query: str, meta: dict[str, Any]) -> BBox2D | None:
        return self._agent.detect(rgb, query)


class AdapterNotFoundError(RuntimeError):
    pass


class AdapterResolver:
    def __init__(self, ckpt_root: Path):
        self._ckpt_root = ckpt_root

    def resolve(self, experiment: str) -> Path:
        pointer = self._ckpt_root / experiment / "LATEST"
        if not pointer.exists():
            raise AdapterNotFoundError(f"no LATEST pointer under {pointer.parent}")
        adapter_dir = pointer.parent / pointer.read_text().strip() / "adapter"
        if not adapter_dir.is_dir():
            raise AdapterNotFoundError(f"checkpoint has no adapter dir: {adapter_dir}")
        return adapter_dir


class LoraVlmDetector(VlmDetector):
    name = "vlm+adapter"

    def __init__(self, profile: Profile, adapter_dir: Path):
        from peft import PeftModel

        super().__init__(profile)
        self._agent.model = PeftModel.from_pretrained(self._agent.model, str(adapter_dir))
        self._agent.model.eval()
        self.adapter_dir = adapter_dir


class DetectorFactory:
    def __init__(self, profile: Profile):
        self._profile = profile

    def create(self, cfg: dict[str, Any]) -> Detector:
        kind = str(cfg.get("detector", "gt"))
        if kind == "gt":
            return GtDetector()
        if kind == "vlm":
            return VlmDetector(self._profile)
        if kind == "vlm+adapter":
            experiment = cfg.get("adapter_from")
            if not isinstance(experiment, str) or not experiment:
                raise ValueError("detector 'vlm+adapter' requires config key 'adapter_from'")
            resolver = AdapterResolver(self._profile.paths.ckpt_root)
            return LoraVlmDetector(self._profile, resolver.resolve(experiment))
        raise ValueError(f"unknown detector {kind!r}")
