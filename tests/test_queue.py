# Tests for the file-locked experiment queue.

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
import yaml

from training.base import TrainerOutcome
from training.queue import ExperimentQueue, ExperimentStatus, QueueError


def _write_queue(path: Path, experiments) -> None:
    path.write_text(yaml.safe_dump({"experiments": experiments}, sort_keys=False))


def _claim_worker(path_str: str, owner: str, out_q) -> None:
    q = ExperimentQueue(Path(path_str), claim_ttl_s=60.0)
    claimed = q.claim_next(owner)
    out_q.put(None if claimed is None else claimed.name)


@pytest.fixture
def tmp_queue(tmp_path):
    path = tmp_path / "queue.yaml"
    _write_queue(
        path,
        [
            {"name": "a", "kind": "datagen", "config": {}},
            {"name": "b", "kind": "lora", "requires": ["a"], "config": {}},
            {"name": "c", "kind": "eval", "requires": ["a", "b"], "config": {}},
            {"name": "d", "kind": "pointnet", "config": {}},
        ],
    )
    return path


class TestExperimentQueue:
    def test_claim_marks_running_and_sets_owner(self, tmp_queue):
        q = ExperimentQueue(tmp_queue, claim_ttl_s=60.0)
        claimed = q.claim_next("owner-1")
        assert claimed is not None
        assert claimed.name == "a"
        assert claimed.status is ExperimentStatus.RUNNING
        assert claimed.owner == "owner-1"
        assert claimed.heartbeat is not None

    def test_requires_gate_blocks_until_dependency_done(self, tmp_queue):
        q = ExperimentQueue(tmp_queue)
        a = q.claim_next("o")
        assert a.name == "a"
        nxt = q.claim_next("o2")
        assert nxt is not None
        assert nxt.name == "d"
        q.release("a", "o", TrainerOutcome.COMPLETED)
        b = q.claim_next("o3")
        assert b is not None
        assert b.name == "b"

    def test_stale_heartbeat_is_reclaimable(self, tmp_queue):
        clock = {"t": 1000.0}

        def now():
            return clock["t"]

        q = ExperimentQueue(tmp_queue, claim_ttl_s=10.0, clock=now)
        first = q.claim_next("old")
        assert first.name == "a"
        clock["t"] = 1020.0
        reclaimed = q.claim_next("new")
        assert reclaimed is not None
        assert reclaimed.name == "a"
        assert reclaimed.owner == "new"

    def test_release_preempted_returns_to_pending_without_retry_cost(self, tmp_queue):
        q = ExperimentQueue(tmp_queue)
        a = q.claim_next("o")
        q.release(a.name, "o", TrainerOutcome.PREEMPTED)
        snap = {s.name: s for s in q.snapshot()}
        assert snap["a"].status is ExperimentStatus.PENDING
        assert snap["a"].retries == 0
        assert snap["a"].owner is None

    def test_release_failed_exhausts_retries_then_fails_permanently(self, tmp_queue):
        q = ExperimentQueue(tmp_queue)
        for i in range(3):
            claimed = q.claim_next(f"o{i}")
            assert claimed.name == "a"
            q.release("a", f"o{i}", TrainerOutcome.FAILED)
        snap = {s.name: s for s in q.snapshot()}
        assert snap["a"].status is ExperimentStatus.FAILED
        assert snap["a"].retries == 3

    def test_heartbeat_wrong_owner_raises(self, tmp_queue):
        q = ExperimentQueue(tmp_queue)
        q.claim_next("owner")
        with pytest.raises(QueueError):
            q.heartbeat("a", "intruder")

    def test_is_drained(self, tmp_queue):
        q = ExperimentQueue(tmp_queue)
        assert not q.is_drained()
        specs = q.snapshot()
        _write_queue(
            tmp_queue,
            [{**s.to_dict(), "status": "done", "owner": None, "heartbeat": None} for s in specs],
        )
        assert ExperimentQueue(tmp_queue).is_drained()

    def test_fail_unsatisfiable_marks_dependents_failed(self, tmp_queue):
        q = ExperimentQueue(tmp_queue)
        for i in range(3):
            claimed = q.claim_next(f"o{i}")
            assert claimed.name == "a"
            q.release("a", f"o{i}", TrainerOutcome.FAILED)
        blocked = q.fail_unsatisfiable()
        assert "b" in blocked
        assert "c" in blocked
        snap = {s.name: s for s in q.snapshot()}
        assert snap["b"].status is ExperimentStatus.FAILED
        assert snap["c"].status is ExperimentStatus.FAILED
        assert snap["d"].status is ExperimentStatus.PENDING
        assert q.is_drained() is False
        nxt = q.claim_next("o-d")
        assert nxt is not None
        assert nxt.name == "d"

    def test_concurrent_claims_never_double_assign(self, tmp_queue):
        _write_queue(
            tmp_queue,
            [{"name": f"e{i}", "kind": "datagen", "config": {}} for i in range(4)],
        )

        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        procs = [
            ctx.Process(target=_claim_worker, args=(str(tmp_queue), f"p{i}", out_q))
            for i in range(8)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0

        names = [out_q.get(timeout=5) for _ in range(8)]
        claimed_names = [n for n in names if n is not None]
        assert len(claimed_names) == 4
        assert len(set(claimed_names)) == 4
