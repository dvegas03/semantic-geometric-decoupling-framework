# Tests for atomic checkpoint contract.

from __future__ import annotations

from pathlib import Path

import pytest

from training.checkpoint import CheckpointManager


class TestCheckpointManager:
    def test_save_then_latest_roundtrip(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ck")
        path = mgr.save(3, lambda d: (d / "x.txt").write_text("hi"))
        assert path.name == "step_000000003"
        latest = mgr.latest()
        assert latest == path
        assert (latest / "x.txt").read_text() == "hi"
        assert (tmp_path / "ck" / "LATEST").read_text().strip() == "step_000000003"

    def test_hard_kill_leaves_no_partial_checkpoint(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ck", keep=2)
        mgr.save(1, lambda d: (d / "ok.txt").write_text("v1"))

        def boom(d: Path) -> None:
            (d / "partial.bin").write_bytes(b"xxx")
            raise RuntimeError("simulated SIGKILL mid-write")

        with pytest.raises(RuntimeError):
            mgr.save(2, boom)

        latest = mgr.latest()
        assert latest is not None
        assert latest.name == "step_000000001"
        assert (latest / "ok.txt").read_text() == "v1"
        mgr.save(3, lambda d: (d / "ok.txt").write_text("v3"))
        assert not list((tmp_path / "ck").glob("step_*.tmp"))

    def test_pointer_always_names_a_complete_dir(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ck")
        mgr.save(5, lambda d: (d / "a").write_text("1"))
        pointed = (tmp_path / "ck" / "LATEST").read_text().strip()
        assert (tmp_path / "ck" / pointed).is_dir()
        assert not pointed.endswith(".tmp")

    def test_prune_keeps_newest_and_pointed(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ck", keep=2)
        for step in range(1, 6):
            mgr.save(step, lambda d, s=step: (d / "s").write_text(str(s)))
        complete = sorted(p.name for p in (tmp_path / "ck").glob("step_*") if p.is_dir())
        assert len(complete) == 2
        assert "step_000000005" in complete
        assert mgr.latest().name == "step_000000005"

    def test_keep_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError):
            CheckpointManager(tmp_path, keep=0)
