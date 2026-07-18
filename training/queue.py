# File-locked experiment queue shared by all chain jobs.

from __future__ import annotations

import dataclasses
import fcntl
import os
import socket
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def default_owner_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    kind: str
    config: dict[str, object] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    status: ExperimentStatus = ExperimentStatus.PENDING
    retries: int = 0
    max_retries: int = 2
    owner: str | None = None
    heartbeat: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError(f"ExperimentSpec requires name and kind, got {self}")

    def replaced(self, **changes: object) -> ExperimentSpec:
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> ExperimentSpec:
        config_raw = raw.get("config") or {}
        requires_raw = raw.get("requires") or ()
        return cls(
            name=str(raw["name"]),
            kind=str(raw["kind"]),
            config=dict(config_raw) if isinstance(config_raw, dict) else {},
            requires=tuple(requires_raw) if isinstance(requires_raw, (list, tuple)) else (),
            status=ExperimentStatus(str(raw.get("status", "pending"))),
            retries=_as_int(raw.get("retries"), 0),
            max_retries=_as_int(raw.get("max_retries"), 2),
            owner=None if raw.get("owner") is None else str(raw["owner"]),
            heartbeat=_as_float(raw.get("heartbeat"), None),
        )

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "config": self.config,
            "status": self.status.value,
            "retries": self.retries,
            "max_retries": self.max_retries,
        }
        if self.requires:
            out["requires"] = list(self.requires)
        if self.owner is not None:
            out["owner"] = self.owner
        if self.heartbeat is not None:
            out["heartbeat"] = self.heartbeat
        return out


class QueueError(RuntimeError):
    pass


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    return cast(int, int(cast(Any, value)))


def _as_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    return cast(float, float(cast(Any, value)))


class ExperimentQueue:
    def __init__(
        self,
        path: Path,
        claim_ttl_s: float = 900.0,
        clock: Callable[[], float] = time.time,
    ):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.claim_ttl_s = claim_ttl_s
        self._clock = clock
        if not self.path.exists():
            raise QueueError(f"queue file not found: {self.path}")

    @contextmanager
    def _exclusive(self) -> Iterator[list[ExperimentSpec]]:
        with open(self.lock_path, "a") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                specs = self._read()
                yield specs
                self._write(specs)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def _read(self) -> list[ExperimentSpec]:
        raw = yaml.safe_load(self.path.read_text())
        entries = (raw or {}).get("experiments")
        if not isinstance(entries, list):
            raise QueueError(f"{self.path} must contain an 'experiments' list")
        specs = [ExperimentSpec.from_dict(e) for e in entries]
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise QueueError(f"duplicate experiment names in {self.path}")
        return specs

    def _write(self, specs: list[ExperimentSpec]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            yaml.safe_dump(
                {"experiments": [s.to_dict() for s in specs]},
                sort_keys=False,
            )
        )
        os.replace(tmp, self.path)

    def _claimable(self, spec: ExperimentSpec, done: set) -> bool:
        if any(r not in done for r in spec.requires):
            return False
        if spec.status is ExperimentStatus.PENDING:
            return True
        if spec.status is ExperimentStatus.RUNNING:
            return spec.heartbeat is None or (self._clock() - spec.heartbeat) > self.claim_ttl_s
        return False

    def claim_next(self, owner: str) -> ExperimentSpec | None:
        with self._exclusive() as specs:
            done = {s.name for s in specs if s.status is ExperimentStatus.DONE}
            for i, spec in enumerate(specs):
                if self._claimable(spec, done):
                    claimed = spec.replaced(
                        status=ExperimentStatus.RUNNING,
                        owner=owner,
                        heartbeat=self._clock(),
                    )
                    specs[i] = claimed
                    return claimed
        return None

    def heartbeat(self, name: str, owner: str) -> None:
        with self._exclusive() as specs:
            for i, spec in enumerate(specs):
                if spec.name == name:
                    if spec.owner != owner:
                        raise QueueError(
                            f"heartbeat by {owner} but {name} is owned by {spec.owner}"
                        )
                    specs[i] = spec.replaced(heartbeat=self._clock())
                    return
        raise QueueError(f"unknown experiment {name!r}")

    def release(self, name: str, owner: str, outcome: object) -> None:
        from training.base import TrainerOutcome

        with self._exclusive() as specs:
            for i, spec in enumerate(specs):
                if spec.name != name:
                    continue
                if spec.owner != owner:
                    raise QueueError(f"release by {owner} but {name} is owned by {spec.owner}")
                if outcome is TrainerOutcome.COMPLETED:
                    new_status, retries = ExperimentStatus.DONE, spec.retries
                elif outcome is TrainerOutcome.PREEMPTED:
                    new_status, retries = ExperimentStatus.PENDING, spec.retries
                else:
                    retries = spec.retries + 1
                    new_status = (
                        ExperimentStatus.PENDING
                        if retries <= spec.max_retries
                        else ExperimentStatus.FAILED
                    )
                specs[i] = spec.replaced(
                    status=new_status,
                    retries=retries,
                    owner=None,
                    heartbeat=None,
                )
                return
        raise QueueError(f"unknown experiment {name!r}")

    def extend(self, new_specs: Sequence[ExperimentSpec]) -> int:
        added = 0
        with self._exclusive() as specs:
            existing = {s.name for s in specs}
            for spec in new_specs:
                if spec.name in existing:
                    continue
                specs.append(spec)
                existing.add(spec.name)
                added += 1
        return added

    def snapshot(self) -> list[ExperimentSpec]:
        with self._exclusive() as specs:
            return list(specs)

    def fail_unsatisfiable(self) -> list[str]:
        failed_names: list[str] = []
        with self._exclusive() as specs:
            permanently_failed = {s.name for s in specs if s.status is ExperimentStatus.FAILED}
            if not permanently_failed:
                return failed_names
            changed = True
            while changed:
                changed = False
                for i, spec in enumerate(specs):
                    if spec.status is not ExperimentStatus.PENDING:
                        continue
                    if any(r in permanently_failed for r in spec.requires):
                        specs[i] = spec.replaced(
                            status=ExperimentStatus.FAILED,
                            owner=None,
                            heartbeat=None,
                        )
                        permanently_failed.add(spec.name)
                        failed_names.append(spec.name)
                        changed = True
        return failed_names

    def is_drained(self) -> bool:
        terminal = (ExperimentStatus.DONE, ExperimentStatus.FAILED)
        return all(s.status in terminal for s in self.snapshot())
