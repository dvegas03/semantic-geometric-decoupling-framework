# Trainer abstraction: preemptible, checkpointed units of queue work.

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum

from training.checkpoint import CheckpointManager
from training.profile import Profile
from training.queue import ExperimentSpec


class TrainerOutcome(Enum):
    COMPLETED = "completed"
    PREEMPTED = "preempted"
    FAILED = "failed"


class Trainer(ABC):
    def __init__(
        self,
        spec: ExperimentSpec,
        profile: Profile,
        checkpoints: CheckpointManager,
        stop_event: threading.Event,
    ):
        self.spec = spec
        self.profile = profile
        self.checkpoints = checkpoints
        self.stop_event = stop_event

    @abstractmethod
    def run(self) -> TrainerOutcome:
        pass


class TrainerRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], type[Trainer]]] = {}

    def register(self, kind: str, factory: Callable[[], type[Trainer]]) -> None:
        if kind in self._factories:
            raise ValueError(f"duplicate trainer kind {kind!r}")
        self._factories[kind] = factory

    def create(
        self,
        spec: ExperimentSpec,
        profile: Profile,
        checkpoints: CheckpointManager,
        stop_event: threading.Event,
    ) -> Trainer:
        if spec.kind not in self._factories:
            raise KeyError(f"no trainer registered for kind {spec.kind!r}")
        trainer_cls = self._factories[spec.kind]()
        return trainer_cls(spec, profile, checkpoints, stop_event)

    @classmethod
    def default(cls) -> TrainerRegistry:
        registry = cls()
        registry.register("datagen", lambda: _imported("training.datagen", "DatagenJob"))
        registry.register("lora", lambda: _imported("training.train_lora", "LoRATrainer"))
        registry.register(
            "pointnet", lambda: _imported("training.train_pointnet", "PointNetTrainer")
        )
        registry.register("eval", lambda: _imported("training.eval", "EvalJob"))
        return registry


def _imported(module: str, attr: str) -> type[Trainer]:
    import importlib

    return getattr(importlib.import_module(module), attr)  # type: ignore[no-any-return]
