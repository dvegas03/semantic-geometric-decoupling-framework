# Tests for the hypothesis-grid sweep generator and queue populator.

from __future__ import annotations

import yaml

from training.queue import ExperimentQueue, ExperimentStatus
from training.sweeps import QueuePopulator, SweepAxis, SweepGrid, default_grid


def _small_grid() -> SweepGrid:
    return SweepGrid(
        base_name="lora",
        kind="lora",
        axes=[
            SweepAxis(key="lora.rank", values=(8, 16)),
            SweepAxis(key="curriculum.schedule", values=("curriculum", "uniform")),
            SweepAxis(key="datagen.vocab_mode", values=("wide", "narrow")),
        ],
        requires=("datagen-mini",),
    )


class TestSweepGrid:
    def test_specs_cover_full_cartesian_product(self):
        grid = _small_grid()
        specs = grid.specs()
        assert len(specs) == 2 * 2 * 2

    def test_names_are_deterministic_across_runs(self):
        grid = _small_grid()
        names_a = [s.name for s in grid.specs()]
        names_b = [s.name for s in grid.specs()]
        assert names_a == names_b

    def test_names_are_unique_and_encode_coordinates(self):
        grid = _small_grid()
        names = [s.name for s in grid.specs()]
        assert len(names) == len(set(names))
        assert "lora-r16-cur=uniform-vocab=wide" in names
        assert "lora-r8-cur=curriculum-vocab=narrow" in names

    def test_specs_inherit_kind_and_requires(self):
        grid = _small_grid()
        for spec in grid.specs():
            assert spec.kind == "lora"
            assert spec.requires == ("datagen-mini",)

    def test_default_grid_is_twelve_arms(self):
        assert len(default_grid().specs()) == 12


class TestQueuePopulator:
    def _seed_queue(self, tmp_path):
        path = tmp_path / "queue.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "experiments": [
                        {"name": "datagen-mini", "kind": "datagen", "config": {}, "status": "done"},
                    ]
                },
                sort_keys=False,
            )
        )
        return path

    def test_merge_appends_all_new_specs(self, tmp_path):
        path = self._seed_queue(tmp_path)
        grid = _small_grid()
        populator = QueuePopulator(path)

        added = populator.merge(grid.specs())
        assert len(added) == len(grid.specs())

        snapshot = ExperimentQueue(path).snapshot()
        assert len(snapshot) == 1 + len(grid.specs())
        names = {s.name for s in snapshot}
        assert all(s.name in names for s in grid.specs())

    def test_merge_is_idempotent(self, tmp_path):
        path = self._seed_queue(tmp_path)
        grid = _small_grid()
        populator = QueuePopulator(path)

        first = populator.merge(grid.specs())
        second = populator.merge(grid.specs())

        assert len(first) == len(grid.specs())
        assert second == []

        snapshot = ExperimentQueue(path).snapshot()
        assert len(snapshot) == 1 + len(grid.specs())

    def test_merge_preserves_status_of_existing_entries(self, tmp_path):
        path = self._seed_queue(tmp_path)
        grid = _small_grid()
        populator = QueuePopulator(path)
        populator.merge(grid.specs())

        queue = ExperimentQueue(path)
        claimed = queue.claim_next("worker-1")
        assert claimed is not None
        assert claimed.status is ExperimentStatus.RUNNING

        populator.merge(grid.specs())
        snapshot = {s.name: s for s in ExperimentQueue(path).snapshot()}
        assert snapshot[claimed.name].status is ExperimentStatus.RUNNING
        assert snapshot[claimed.name].owner == "worker-1"

    def test_merge_creates_queue_file_when_missing(self, tmp_path):
        path = tmp_path / "nested" / "queue.yaml"
        grid = _small_grid()
        populator = QueuePopulator(path)

        added = populator.merge(grid.specs())
        assert len(added) == len(grid.specs())
        assert path.exists()
