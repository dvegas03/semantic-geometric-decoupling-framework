# Tests for the evolving-loop planner/scheduler.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from training.base import TrainerOutcome
from training.planner import (
    MetaScheduler,
    ParamSpec,
    Planner,
    SearchSpace,
    SuccessiveHalvingPlanner,
)
from training.queue import ExperimentQueue, ExperimentSpec, ExperimentStatus
from training.results import ExperimentResult, ResultStore, config_hash


def _write_queue(path: Path, experiments: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"experiments": experiments}, sort_keys=False))


@pytest.fixture
def store(tmp_path) -> ResultStore:
    return ResultStore(tmp_path / "results.db")


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        params=(
            ParamSpec(name="rank", values=(4, 8, 16)),
            ParamSpec(name="learning_rate", low=1e-5, high=1e-3, log=True),
        )
    )


class TestParamSpec:
    def test_categorical_samples_from_values(self):
        spec = ParamSpec(name="rank", values=(4, 8, 16))
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert spec.sample(rng) in (4, 8, 16)

    def test_log_uniform_stays_in_range(self):
        spec = ParamSpec(name="lr", low=1e-5, high=1e-3, log=True)
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = spec.sample(rng)
            assert 1e-5 <= v <= 1e-3

    def test_rejects_both_categorical_and_range(self):
        with pytest.raises(ValueError):
            ParamSpec(name="x", values=(1, 2), low=0.0, high=1.0)

    def test_rejects_neither(self):
        with pytest.raises(ValueError):
            ParamSpec(name="x")


class TestSuccessiveHalvingPlanner:
    def test_fresh_candidates_are_deduped_against_store(self, store, space):
        planner = SuccessiveHalvingPlanner(
            hypothesis="H1",
            kind="lora",
            base_config={},
            search_space=space,
            rungs=(10, 40),
            seed=0,
        )
        first_batch = planner.propose(store, budget=5)
        assert len(first_batch) == 5
        for spec in first_batch:
            store.record(
                ExperimentResult(
                    experiment=spec.name,
                    hypothesis="H1",
                    kind="lora",
                    config=spec.config,
                    metrics={"acc@0.5": 0.1},
                    status="done",
                )
            )
        second_batch = planner.propose(store, budget=5)
        first_configs = {frozenset(s.config.items()) for s in first_batch}
        fresh_in_second = [s for s in second_batch if s.config.get("max_steps") == 10]
        assert all(frozenset(s.config.items()) not in first_configs for s in fresh_in_second)

    def test_promotes_top_fraction_to_next_rung(self, store, space):
        planner = SuccessiveHalvingPlanner(
            hypothesis="H1",
            kind="lora",
            base_config={},
            search_space=space,
            rungs=(10, 40, 160),
            eta=3,
            metric="acc@0.5",
            maximize=True,
            seed=0,
        )
        configs = [{"rank": r, "learning_rate": 1e-4, "max_steps": 10} for r in range(6)]
        for i, cfg in enumerate(configs):
            store.record(
                ExperimentResult(
                    experiment=f"arm-{i}",
                    hypothesis="H1",
                    kind="lora",
                    config=cfg,
                    metrics={"acc@0.5": float(i) / 10.0},
                    status="done",
                )
            )
        proposals = planner.propose(store, budget=10)
        promoted = [s for s in proposals if s.config.get("max_steps") == 40]
        assert len(promoted) >= 1
        assert len(promoted) == 2
        promoted_ranks = {s.config["rank"] for s in promoted}
        assert promoted_ranks == {5, 4}

    def test_never_proposes_a_seen_config_hash(self, store, space):
        planner = SuccessiveHalvingPlanner(
            hypothesis="H1",
            kind="lora",
            base_config={},
            search_space=space,
            rungs=(10,),
            seed=1,
        )
        seen_hashes: set[str] = set()
        for _ in range(4):
            batch = planner.propose(store, budget=5)
            for spec in batch:
                h = config_hash(spec.config)
                assert h not in seen_hashes, "planner repeated a config across polls"
                seen_hashes.add(h)
                store.record(
                    ExperimentResult(
                        experiment=spec.name,
                        hypothesis="H1",
                        kind="lora",
                        config=spec.config,
                        metrics={"acc@0.5": 0.5},
                        status="done",
                    )
                )

    def test_converges_to_empty_once_max_candidates_reached(self, store, space):
        planner = SuccessiveHalvingPlanner(
            hypothesis="H1",
            kind="lora",
            base_config={},
            search_space=space,
            rungs=(10,),
            max_candidates=3,
            seed=2,
        )
        for i in range(3):
            store.record(
                ExperimentResult(
                    experiment=f"a{i}",
                    hypothesis="H1",
                    kind="lora",
                    config={"rank": i, "learning_rate": 1e-4, "max_steps": 10},
                    metrics={"acc@0.5": 0.1},
                    status="done",
                )
            )
        assert planner.propose(store, budget=5) == []

    def test_rejects_duplicate_rungs(self, space):
        with pytest.raises(ValueError):
            SuccessiveHalvingPlanner(
                hypothesis="H1",
                kind="lora",
                base_config={},
                search_space=space,
                rungs=(10, 10),
            )


class TestExperimentQueueExtend:
    def test_extend_appends_new_pending_entries(self, tmp_path):
        path = tmp_path / "queue.yaml"
        _write_queue(path, [{"name": "a", "kind": "datagen", "config": {}, "status": "done"}])
        queue = ExperimentQueue(path)
        added = queue.extend([ExperimentSpec(name="b", kind="lora", config={"rank": 8})])
        assert added == 1
        snap = queue.snapshot()
        assert {s.name for s in snap} == {"a", "b"}
        b = next(s for s in snap if s.name == "b")
        assert b.status is ExperimentStatus.PENDING

    def test_extend_is_idempotent_by_name(self, tmp_path):
        path = tmp_path / "queue.yaml"
        _write_queue(path, [{"name": "a", "kind": "datagen", "config": {}, "status": "done"}])
        queue = ExperimentQueue(path)
        queue.extend([ExperimentSpec(name="b", kind="lora", config={})])
        added_again = queue.extend([ExperimentSpec(name="b", kind="lora", config={"different": 1})])
        assert added_again == 0
        assert len(queue.snapshot()) == 2


class TestMetaScheduler:
    def test_refill_skipped_when_above_watermark(self, tmp_path, store, space):
        path = tmp_path / "queue.yaml"
        _write_queue(
            path,
            [
                {"name": "a", "kind": "datagen", "config": {}, "status": "pending"},
                {"name": "b", "kind": "lora", "config": {}, "status": "pending"},
            ],
        )
        queue = ExperimentQueue(path)
        planner = SuccessiveHalvingPlanner(
            hypothesis="H1", kind="lora", base_config={}, search_space=space, rungs=(10,)
        )
        scheduler = MetaScheduler([planner], store, low_watermark=2, refill_batch=3)
        assert scheduler.maybe_refill(queue) == 0
        assert len(queue.snapshot()) == 2

    def test_refill_adds_new_pending_when_below_watermark(self, tmp_path, store, space):
        path = tmp_path / "queue.yaml"
        _write_queue(path, [{"name": "a", "kind": "datagen", "config": {}, "status": "done"}])
        queue = ExperimentQueue(path)
        planner = SuccessiveHalvingPlanner(
            hypothesis="H1", kind="lora", base_config={}, search_space=space, rungs=(10,), seed=3
        )
        scheduler = MetaScheduler([planner], store, low_watermark=2, refill_batch=3)
        added = scheduler.maybe_refill(queue)
        assert added == 3
        assert len(queue.snapshot()) == 4

    def test_round_robins_across_planners(self, tmp_path, store, space):
        path = tmp_path / "queue.yaml"
        _write_queue(path, [])
        queue = ExperimentQueue(path)
        p1 = SuccessiveHalvingPlanner(
            hypothesis="H1",
            kind="lora",
            base_config={},
            search_space=space,
            rungs=(10,),
            max_candidates=1,
            seed=4,
        )
        p2 = SuccessiveHalvingPlanner(
            hypothesis="H2",
            kind="lora",
            base_config={},
            search_space=space,
            rungs=(10,),
            seed=5,
        )
        scheduler = MetaScheduler([p1, p2], store, low_watermark=1, refill_batch=1)
        scheduler.maybe_refill(queue)
        added_names = {s.name for s in queue.snapshot()}
        assert any(n.startswith("h1-") for n in added_names)
        spec = queue.snapshot()[0]
        store.record(
            ExperimentResult(
                experiment=spec.name,
                hypothesis="H1",
                kind="lora",
                config=spec.config,
                metrics={"acc@0.5": 0.5},
                status="done",
            )
        )
        queue.claim_next("test-owner")
        queue.release(spec.name, "test-owner", outcome=TrainerOutcome.COMPLETED)
        scheduler.maybe_refill(queue)
        added_names_2 = {s.name for s in queue.snapshot()}
        assert any(n.startswith("h2-") for n in added_names_2)

    def test_rejects_empty_planners(self, store):
        with pytest.raises(ValueError):
            MetaScheduler([], store)


class TestFakePlannerContract:
    def test_custom_planner_subclass_works(self, tmp_path, store):
        class AlwaysEmptyPlanner(Planner):
            hypothesis = "H9"

            def propose(self, store, budget):
                return []

        path = tmp_path / "queue.yaml"
        _write_queue(path, [{"name": "a", "kind": "datagen", "config": {}, "status": "pending"}])
        queue = ExperimentQueue(path)
        scheduler = MetaScheduler([AlwaysEmptyPlanner()], store, low_watermark=5)
        assert scheduler.maybe_refill(queue) == 0
