# Tests for mini-corpus datagen primitives.

from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest
import yaml

from training.checkpoint import CheckpointManager
from training.datagen import (
    CategorySplit,
    CorruptionParams,
    DatagenJob,
    FrameComposer,
    PrimitiveAssetLibrary,
    QueryGenerator,
    QuestDepthCorrupter,
    RayCastRenderer,
    SceneSampler,
)
from training.profile import Profile
from training.queue import ExperimentSpec


@pytest.fixture
def tiny_profile(tmp_path) -> Profile:
    raw = {
        "name": "tiny",
        "device": "cpu",
        "dataloader_workers": 0,
        "checkpoint_interval_s": 1,
        "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
        "paths": {
            "data_root": str(tmp_path / "corpus"),
            "ckpt_root": str(tmp_path / "ckpt"),
            "mirror_root": str(tmp_path / "mirror"),
        },
        "corpus": {
            "frames": 8,
            "categories": 8,
            "holdout_category_fraction": 0.25,
            "shard_size": 4,
            "image_hw": [64, 80],
        },
        "lora": {
            "rank": 4,
            "batch_size": 1,
            "grad_accum": 1,
            "max_steps": 2,
            "learning_rate": 1e-4,
        },
    }
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(raw))
    return Profile.load(str(path))


class TestQuestDepthCorrupter:
    def test_void_ratio_within_tolerance(self):
        h, w = 120, 160
        clean = np.full((h, w), 1.5, dtype=np.float32)
        clean[40:80, 50:110] = 1.2
        params = CorruptionParams(void_ratio=0.5, noise_std_1m=0.0, edge_bleed_px=3, seed=0)
        out = QuestDepthCorrupter().corrupt(clean, params)
        void_frac = float((out == 0).mean())
        assert abs(void_frac - 0.5) <= 0.02

    def test_noise_grows_with_depth(self):
        clean = np.zeros((40, 40), dtype=np.float32)
        clean[:, :20] = 1.0
        clean[:, 20:] = 3.0
        params = CorruptionParams(void_ratio=0.0, noise_std_1m=0.05, edge_bleed_px=1, seed=1)
        out = QuestDepthCorrupter().corrupt(clean, params)
        left = out[:, 5:15]
        right = out[:, 25:35]
        left_valid = left[left > 0]
        right_valid = right[right > 0]
        left_std = float(np.std(left_valid - 1.0)) if left_valid.size else 0.0
        right_std = float(np.std(right_valid - 3.0)) if right_valid.size else 0.0
        assert right_std > left_std


class TestCategorySplit:
    def test_deterministic_exact_count_and_order_independent(self):
        cats = [f"c{i}" for i in range(20)]
        a = CategorySplit(cats, 0.2)
        b = CategorySplit(list(reversed(cats)), 0.2)
        assert a.heldout == b.heldout
        assert len(a.heldout) == round(0.2 * 20)
        assert CategorySplit(cats, 0.2).heldout == a.heldout

    def test_exact_count_for_large_n(self):
        cats = [f"c{i}" for i in range(100)]
        split = CategorySplit(cats, 0.2)
        assert len(split.heldout) == 20


class TestFrameComposer:
    def test_gt_bbox_matches_instance_map(self):
        library = PrimitiveAssetLibrary(n_categories=8)
        split = CategorySplit(library.categories(), 0.25)
        composer = FrameComposer(
            library=library,
            split=split,
            sampler=SceneSampler(library, max_objects=4),
            renderer=RayCastRenderer((64, 80)),
            corrupter=QuestDepthCorrupter(),
            queries=QueryGenerator(),
        )
        rng = np.random.default_rng(0)
        record = composer.compose(0, rng)
        ys, xs = np.where(record.instance_map > 0)
        if ys.size == 0:
            pytest.skip("no visible instances in this random scene")
        assert record.bbox_2d.width > 0 and record.bbox_2d.height > 0
        crop = record.instance_map[record.bbox_2d.as_slices()]
        assert crop.size > 0
        assert (crop > 0).any()


class TestDatagenJob:
    def test_shard_regeneration_after_preemption_is_bit_identical(self, tiny_profile):
        spec = ExperimentSpec(name="dg", kind="datagen", config={})
        ck = CheckpointManager(tiny_profile.paths.ckpt_root / "dg")
        stop = threading.Event()

        job = DatagenJob(spec, tiny_profile, ck, stop)
        original = job._write_shard
        calls = {"n": 0}

        def wrapped(composer, out_dir, shard_idx):
            original(composer, out_dir, shard_idx)
            calls["n"] += 1
            if calls["n"] >= 1:
                stop.set()

        job._write_shard = wrapped  # type: ignore[method-assign]
        outcome = job.run()
        assert outcome.value == "preempted"
        shard0 = tiny_profile.paths.data_root / "shard-00000.tar"
        assert shard0.exists()
        digest1 = hashlib.sha256(shard0.read_bytes()).hexdigest()

        stop2 = threading.Event()
        DatagenJob(spec, tiny_profile, ck, stop2)
        shard0.unlink()
        import shutil

        shutil.rmtree(tiny_profile.paths.ckpt_root / "dg")
        ck2 = CheckpointManager(tiny_profile.paths.ckpt_root / "dg")
        job3 = DatagenJob(spec, tiny_profile, ck2, threading.Event())
        assert job3.run().value == "completed"
        digest2 = hashlib.sha256(shard0.read_bytes()).hexdigest()
        assert digest1 == digest2

    def test_datagen_writes_expected_shards(self, tiny_profile):
        spec = ExperimentSpec(name="dg2", kind="datagen", config={})
        ck = CheckpointManager(tiny_profile.paths.ckpt_root / "dg2")
        job = DatagenJob(spec, tiny_profile, ck, threading.Event())
        assert job.run().value == "completed"
        shards = sorted(tiny_profile.paths.data_root.glob("shard-*.tar"))
        assert len(shards) == 2
