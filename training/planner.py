# Generative experiment planners: turn ``ResultStore`` history into the next experiments, so a chain job's queue evolves instead of executing a fixed grid.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.queue import ExperimentQueue, ExperimentSpec, ExperimentStatus
from training.results import ResultStore, config_hash


@dataclass(frozen=True)
class ParamSpec:
    name: str
    values: tuple[Any, ...] | None = None
    low: float | None = None
    high: float | None = None
    log: bool = False
    is_int: bool = False

    def __post_init__(self) -> None:
        has_categorical = self.values is not None
        has_range = self.low is not None and self.high is not None
        if has_categorical == has_range:
            raise ValueError(
                f"ParamSpec {self.name!r} needs exactly one of `values` or `low`+`high`"
            )
        if has_range and self.log and self.low is not None and self.low <= 0.0:
            raise ValueError(f"ParamSpec {self.name!r}: log-scale sampling requires low > 0")

    def sample(self, rng: np.random.Generator) -> Any:
        if self.values is not None:
            return self.values[int(rng.integers(0, len(self.values)))]
        assert self.low is not None and self.high is not None
        if self.log:
            value = float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        else:
            value = float(rng.uniform(self.low, self.high))
        return int(round(value)) if self.is_int else value


@dataclass(frozen=True)
class SearchSpace:
    params: tuple[ParamSpec, ...]

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {p.name: p.sample(rng) for p in self.params}


class Planner(ABC):
    hypothesis: str

    @abstractmethod
    def propose(self, store: ResultStore, budget: int) -> list[ExperimentSpec]:
        pass


class SuccessiveHalvingPlanner(Planner):
    def __init__(
        self,
        hypothesis: str,
        kind: str,
        base_config: dict[str, Any],
        search_space: SearchSpace,
        rungs: tuple[int, ...] = (250, 1000, 4000),
        eta: int = 3,
        metric: str = "acc@0.5",
        maximize: bool = True,
        max_candidates: int = 200,
        seed: int = 0,
    ):
        if len(rungs) < 1:
            raise ValueError("rungs must be non-empty")
        if len(set(rungs)) != len(rungs):
            raise ValueError(f"rungs must be distinct, got {rungs}")
        if eta < 2:
            raise ValueError(f"eta must be >= 2, got {eta}")
        self.hypothesis = hypothesis
        self._kind = kind
        self._base_config = dict(base_config)
        self._space = search_space
        self._rungs = rungs
        self._eta = eta
        self._metric = metric
        self._maximize = maximize
        self._max_candidates = max_candidates
        self._rng = np.random.default_rng(seed)

    def propose(self, store: ResultStore, budget: int) -> list[ExperimentSpec]:
        if budget <= 0:
            return []
        proposals = self._promotions(store, budget)
        if len(proposals) >= budget:
            return proposals[:budget]
        if store.count(self.hypothesis) < self._max_candidates:
            proposals.extend(self._fresh_candidates(store, budget - len(proposals)))
        return proposals[:budget]

    def _rung_index(self, config: dict[str, Any]) -> int:
        steps = config.get("max_steps", self._rungs[0])
        return self._rungs.index(steps) if steps in self._rungs else 0

    def _promotions(self, store: ResultStore, budget: int) -> list[ExperimentSpec]:
        specs: list[ExperimentSpec] = []
        results = [r for r in store.all(self.hypothesis) if r.status == "done"]
        for rung_idx in range(len(self._rungs) - 1):
            at_rung = [r for r in results if self._rung_index(r.config) == rung_idx]
            if not at_rung:
                continue
            fallback = float("-inf") if self._maximize else float("inf")
            at_rung.sort(
                key=lambda r: r.metrics.get(self._metric, fallback), reverse=self._maximize
            )
            n_promote = max(1, len(at_rung) // self._eta)
            for result in at_rung[:n_promote]:
                promoted_cfg = dict(result.config)
                promoted_cfg["max_steps"] = self._rungs[rung_idx + 1]
                cfg_hash = config_hash(promoted_cfg)
                if store.seen(cfg_hash):
                    continue
                specs.append(self._spec_from_config(promoted_cfg))
                if len(specs) >= budget:
                    return specs
        return specs

    def _fresh_candidates(self, store: ResultStore, budget: int) -> list[ExperimentSpec]:
        specs: list[ExperimentSpec] = []
        attempts = 0
        max_attempts = max(budget * 20, 50)
        while len(specs) < budget and attempts < max_attempts:
            attempts += 1
            sampled = self._space.sample(self._rng)
            config = {**self._base_config, **sampled, "max_steps": self._rungs[0]}
            cfg_hash = config_hash(config)
            if store.seen(cfg_hash) or any(config_hash(s.config) == cfg_hash for s in specs):
                continue
            specs.append(self._spec_from_config(config))
        return specs

    def _spec_from_config(self, config: dict[str, Any]) -> ExperimentSpec:
        name = f"{self.hypothesis.lower()}-{self._kind}-{config_hash(config)[:10]}"
        return ExperimentSpec(name=name, kind=self._kind, config=config)


class MetaScheduler:
    def __init__(
        self,
        planners: Sequence[Planner],
        store: ResultStore,
        low_watermark: int = 2,
        refill_batch: int = 4,
    ):
        if not planners:
            raise ValueError("MetaScheduler requires at least one planner")
        self._planners = list(planners)
        self._store = store
        self._low_watermark = low_watermark
        self._refill_batch = refill_batch
        self._next_idx = 0

    def maybe_refill(self, queue: ExperimentQueue) -> int:
        pending = sum(1 for s in queue.snapshot() if s.status is ExperimentStatus.PENDING)
        if pending >= self._low_watermark:
            return 0

        n = len(self._planners)
        for offset in range(n):
            idx = (self._next_idx + offset) % n
            proposals = self._planners[idx].propose(self._store, self._refill_batch)
            if proposals:
                added = queue.extend(proposals)
                self._next_idx = (idx + 1) % n
                return added
        return 0
