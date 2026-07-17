# Tests for the queue-draining runner with fake trainers.

from __future__ import annotations

import threading
import time
from pathlib import Path

import yaml

from training.base import Trainer, TrainerOutcome, TrainerRegistry
from training.canary import CanaryCheck, CanarySuite
from training.planner import MetaScheduler, Planner
from training.profile import Profile
from training.queue import ExperimentQueue, ExperimentSpec, ExperimentStatus
from training.results import ExperimentResult, ResultStore
from training.runner import EXIT_CANARY_FAILED, EXIT_DRAINED, EXIT_OK, Runner


class SleepTrainer(Trainer):
    fail_times: int = 0
    _fail_counts = {}

    def run(self) -> TrainerOutcome:
        key = self.spec.name
        fails_left = int(self.spec.config.get("fail_times", 0))
        seen = SleepTrainer._fail_counts.get(key, 0)
        if seen < fails_left:
            SleepTrainer._fail_counts[key] = seen + 1
            raise RuntimeError(f"intentional fail #{seen + 1}")

        steps = int(self.spec.config.get("steps", 5))
        start = 0
        latest = self.checkpoints.latest()
        if latest is not None:
            start = int((latest / "step.txt").read_text())

        for step in range(start, steps):
            if self.stop_event.is_set():
                self.checkpoints.save(step, lambda d, s=step: (d / "step.txt").write_text(str(s)))
                return TrainerOutcome.PREEMPTED
            time.sleep(float(self.spec.config.get("sleep", 0.01)))
            self.checkpoints.save(
                step + 1, lambda d, s=step + 1: (d / "step.txt").write_text(str(s))
            )
        return TrainerOutcome.COMPLETED


def _profile(tmp_path: Path) -> Profile:
    raw = {
        "name": "test",
        "device": "cpu",
        "dataloader_workers": 0,
        "checkpoint_interval_s": 1,
        "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
        "paths": {
            "data_root": str(tmp_path / "data"),
            "ckpt_root": str(tmp_path / "ckpt"),
            "mirror_root": str(tmp_path / "mirror"),
        },
        "corpus": {
            "frames": 4,
            "categories": 4,
            "holdout_category_fraction": 0.25,
            "shard_size": 2,
            "image_hw": [32, 32],
        },
        "lora": {
            "rank": 4,
            "batch_size": 1,
            "grad_accum": 1,
            "max_steps": 2,
            "learning_rate": 1e-4,
        },
    }
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(raw))
    return Profile.load(str(path))


def _registry() -> TrainerRegistry:
    reg = TrainerRegistry()
    reg.register("sleep", lambda: SleepTrainer)
    return reg


class TestRunner:
    def test_drains_queue_and_exits_3(self, tmp_path):
        SleepTrainer._fail_counts.clear()
        queue_path = tmp_path / "q.yaml"
        queue_path.write_text(
            yaml.safe_dump(
                {
                    "experiments": [
                        {
                            "name": "one",
                            "kind": "sleep",
                            "config": {"steps": 2, "sleep": 0.001},
                        },
                        {
                            "name": "two",
                            "kind": "sleep",
                            "config": {"steps": 2, "sleep": 0.001},
                        },
                    ]
                }
            )
        )
        runner = Runner(
            ExperimentQueue(queue_path),
            _profile(tmp_path),
            _registry(),
            poll_s=0.05,
        )
        code = runner.run()
        assert code == EXIT_DRAINED
        assert ExperimentQueue(queue_path).is_drained()
        statuses = {s.name: s.status for s in ExperimentQueue(queue_path).snapshot()}
        assert statuses == {
            "one": ExperimentStatus.DONE,
            "two": ExperimentStatus.DONE,
        }

    def test_sigterm_preempts_then_entry_is_pending(self, tmp_path):
        SleepTrainer._fail_counts.clear()
        queue_path = tmp_path / "q.yaml"
        queue_path.write_text(
            yaml.safe_dump(
                {
                    "experiments": [
                        {
                            "name": "slow",
                            "kind": "sleep",
                            "config": {"steps": 50, "sleep": 0.05},
                        }
                    ]
                }
            )
        )
        runner = Runner(
            ExperimentQueue(queue_path),
            _profile(tmp_path),
            _registry(),
            poll_s=0.05,
        )

        def stop_soon():
            time.sleep(0.12)
            runner.stop_event.set()

        threading.Thread(target=stop_soon, daemon=True).start()
        code = runner.run()
        assert code == EXIT_OK
        snap = ExperimentQueue(queue_path).snapshot()[0]
        assert snap.status is ExperimentStatus.PENDING
        assert snap.retries == 0

    def test_crashing_trainer_consumes_retry_then_queue_continues(self, tmp_path):
        SleepTrainer._fail_counts.clear()
        queue_path = tmp_path / "q.yaml"
        queue_path.write_text(
            yaml.safe_dump(
                {
                    "experiments": [
                        {
                            "name": "flaky",
                            "kind": "sleep",
                            "config": {"steps": 1, "fail_times": 1, "sleep": 0.001},
                            "max_retries": 2,
                        },
                        {
                            "name": "ok",
                            "kind": "sleep",
                            "config": {"steps": 1, "sleep": 0.001},
                        },
                    ]
                }
            )
        )
        runner = Runner(
            ExperimentQueue(queue_path),
            _profile(tmp_path),
            _registry(),
            poll_s=0.05,
        )
        code = runner.run()
        assert code == EXIT_DRAINED
        statuses = {s.name: s for s in ExperimentQueue(queue_path).snapshot()}
        assert statuses["flaky"].status is ExperimentStatus.DONE
        assert statuses["flaky"].retries == 1
        assert statuses["ok"].status is ExperimentStatus.DONE

    def test_stuck_queue_with_unmet_requires_is_not_reported_drained(self, tmp_path):
        queue_path = tmp_path / "q.yaml"
        queue_path.write_text(
            yaml.safe_dump(
                {
                    "experiments": [
                        {
                            "name": "blocked",
                            "kind": "sleep",
                            "requires": ["missing"],
                            "config": {},
                        }
                    ]
                }
            )
        )
        q = ExperimentQueue(queue_path)
        assert q.claim_next("o") is None
        assert not q.is_drained()

        runner = Runner(q, _profile(tmp_path), _registry(), poll_s=0.05)
        runner.stop_event.set()
        code = runner.run()
        assert code == EXIT_OK
        assert not ExperimentQueue(queue_path).is_drained()


class ScoringTrainer(Trainer):
    def run(self) -> TrainerOutcome:
        store = ResultStore(self.profile.paths.ckpt_root / "results.db")
        store.record(
            ExperimentResult(
                experiment=self.spec.name,
                hypothesis=str(self.spec.config.get("hypothesis", "H-TEST")),
                kind="scoring",
                config=self.spec.config,
                metrics={"score": float(self.spec.config.get("score", 0.0))},
                status="done",
            )
        )
        return TrainerOutcome.COMPLETED


class _FixedBatchPlanner(Planner):
    hypothesis = "H-TEST"

    def __init__(self, n_batches: int, batch_size: int):
        self._remaining = n_batches
        self._batch_size = batch_size
        self._next_id = 0

    def propose(self, store: ResultStore, budget: int) -> list[ExperimentSpec]:
        if self._remaining <= 0:
            return []
        self._remaining -= 1
        specs = []
        for _ in range(min(self._batch_size, budget)):
            specs.append(
                ExperimentSpec(
                    name=f"evolve-{self._next_id}",
                    kind="scoring",
                    config={"score": float(self._next_id), "hypothesis": self.hypothesis},
                )
            )
            self._next_id += 1
        return specs


class TestEvolvingLoop:
    def _registry_with_scoring(self) -> TrainerRegistry:
        reg = TrainerRegistry()
        reg.register("scoring", lambda: ScoringTrainer)
        return reg

    def test_scheduler_refills_until_planner_converges_then_drains(self, tmp_path):
        queue_path = tmp_path / "q.yaml"
        queue_path.write_text(yaml.safe_dump({"experiments": []}))
        profile = _profile(tmp_path)
        store = ResultStore(profile.paths.ckpt_root / "results.db")
        planner = _FixedBatchPlanner(n_batches=5, batch_size=4)
        scheduler = MetaScheduler([planner], store, low_watermark=2, refill_batch=4)

        runner = Runner(
            ExperimentQueue(queue_path),
            profile,
            self._registry_with_scoring(),
            poll_s=0.01,
            scheduler=scheduler,
        )
        code = runner.run()

        assert code == EXIT_DRAINED
        all_results = store.all("H-TEST")
        assert len(all_results) == 20
        assert len({r.config_hash for r in all_results}) == 20
        assert all(r.status == "done" for r in all_results)

    def test_canary_failure_halts_before_any_trainer_runs(self, tmp_path):
        queue_path = tmp_path / "q.yaml"
        queue_path.write_text(
            yaml.safe_dump(
                {"experiments": [{"name": "one", "kind": "sleep", "config": {"steps": 1}}]}
            )
        )
        failing_canary = CanarySuite(
            (CanaryCheck(name="broken", run=lambda: False, description="simulated failure"),)
        )
        runner = Runner(
            ExperimentQueue(queue_path),
            _profile(tmp_path),
            _registry(),
            poll_s=0.05,
            canary=failing_canary,
        )
        code = runner.run()
        assert code == EXIT_CANARY_FAILED
        snap = ExperimentQueue(queue_path).snapshot()[0]
        assert snap.status is ExperimentStatus.PENDING

    def test_passing_canary_does_not_block_normal_drain(self, tmp_path):
        SleepTrainer._fail_counts.clear()
        queue_path = tmp_path / "q.yaml"
        queue_path.write_text(
            yaml.safe_dump(
                {"experiments": [{"name": "one", "kind": "sleep", "config": {"steps": 1}}]}
            )
        )
        passing_canary = CanarySuite(
            (CanaryCheck(name="ok", run=lambda: True, description="always fine"),)
        )
        runner = Runner(
            ExperimentQueue(queue_path),
            _profile(tmp_path),
            _registry(),
            poll_s=0.05,
            canary=passing_canary,
        )
        code = runner.run()
        assert code == EXIT_DRAINED
        assert ExperimentQueue(queue_path).is_drained()
