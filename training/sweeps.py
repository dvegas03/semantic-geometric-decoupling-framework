# Populates experiments/queue.yaml from the hypothesis grid (H1-H8).

from __future__ import annotations

import argparse
import fcntl
import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from training.queue import ExperimentSpec

_AXIS_LABELS: dict[str, str] = {
    "lora.rank": "r",
    "curriculum.schedule": "cur",
    "datagen.vocab_mode": "vocab",
}


def _axis_label(key: str) -> str:
    return _AXIS_LABELS.get(key, key.rsplit(".", 1)[-1])


def _format_coordinate(key: str, value: object) -> str:
    label = _axis_label(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{label}={value}"
    return f"{label}{value}"


@dataclass(frozen=True)
class SweepAxis:
    key: str
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("SweepAxis.key must be non-empty")
        if not self.values:
            raise ValueError(f"SweepAxis {self.key!r} needs at least one value")


class SweepGrid:
    def __init__(
        self,
        base_name: str,
        kind: str,
        axes: Sequence[SweepAxis],
        requires: tuple[str, ...] = (),
        max_retries: int = 2,
    ) -> None:
        if not base_name or not kind:
            raise ValueError("SweepGrid requires base_name and kind")
        if not axes:
            raise ValueError("SweepGrid needs at least one axis")
        self.base_name = base_name
        self.kind = kind
        self.axes = tuple(axes)
        self.requires = requires
        self.max_retries = max_retries

    def _name(self, point: Sequence[object]) -> str:
        coords = "-".join(
            _format_coordinate(axis.key, value) for axis, value in zip(self.axes, point)
        )
        return f"{self.base_name}-{coords}"

    def specs(self) -> list[ExperimentSpec]:
        out: list[ExperimentSpec] = []
        for point in itertools.product(*(axis.values for axis in self.axes)):
            config = {axis.key: value for axis, value in zip(self.axes, point)}
            out.append(
                ExperimentSpec(
                    name=self._name(point),
                    kind=self.kind,
                    config=config,
                    requires=self.requires,
                    max_retries=self.max_retries,
                )
            )
        return out


class QueuePopulator:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def merge(self, specs: Sequence[ExperimentSpec]) -> list[ExperimentSpec]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(yaml.safe_dump({"experiments": []}, sort_keys=False))
        with open(self.lock_path, "a") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                existing = self._read()
                known_names = {s.name for s in existing}
                added = [s for s in specs if s.name not in known_names]
                if added:
                    self._write(existing + added)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        return added

    def _read(self) -> list[ExperimentSpec]:
        raw = yaml.safe_load(self.path.read_text())
        entries = (raw or {}).get("experiments") or []
        return [ExperimentSpec.from_dict(e) for e in entries]

    def _write(self, specs: list[ExperimentSpec]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            yaml.safe_dump({"experiments": [s.to_dict() for s in specs]}, sort_keys=False)
        )
        os.replace(tmp, self.path)


def default_grid() -> SweepGrid:
    return SweepGrid(
        base_name="lora",
        kind="lora",
        axes=[
            SweepAxis(key="lora.rank", values=(8, 16, 32)),
            SweepAxis(key="curriculum.schedule", values=("curriculum", "uniform")),
            SweepAxis(key="datagen.vocab_mode", values=("wide", "narrow")),
        ],
        requires=("datagen-mini",),
    )


def _print_dry_run(grid: SweepGrid) -> None:
    specs = grid.specs()
    columns = [axis.key for axis in grid.axes]
    header = f"{'name':<34}" + "".join(f"{_axis_label(c):>12}" for c in columns)
    print(header)
    print("-" * len(header))
    for spec in specs:
        row = f"{spec.name:<34}" + "".join(f"{spec.config[c]!s:>12}" for c in columns)
        print(row)
    print(f"\n{len(specs)} arms (H1 lora.rank x H3 curriculum.schedule x H5 datagen.vocab_mode)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Populate experiments/queue.yaml from the H1 x H3 x H5 hypothesis grid."
    )
    parser.add_argument("--queue", default="experiments/queue.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the grid table; do not touch the queue file"
    )
    args = parser.parse_args(argv)

    grid = default_grid()
    if args.dry_run:
        _print_dry_run(grid)
        return 0

    populator = QueuePopulator(Path(args.queue))
    specs = grid.specs()
    added = populator.merge(specs)
    print(
        f"merged {len(added)} new entries into {args.queue} "
        f"({len(specs) - len(added)} already present)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
