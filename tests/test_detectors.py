# Tests for detector factory / adapter resolver (no model weights).

from __future__ import annotations

from pathlib import Path

import pytest

from training.detectors import (
    AdapterNotFoundError,
    AdapterResolver,
    DetectorFactory,
    GtDetector,
)
from training.profile import Profile


@pytest.fixture
def tiny_profile(tmp_path: Path) -> Profile:
    data = tmp_path / "data"
    ckpt = tmp_path / "ckpts"
    mirror = tmp_path / "mirror"
    data.mkdir()
    ckpt.mkdir()
    mirror.mkdir()
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        f"""
name: t
device: cpu
dataloader_workers: 0
checkpoint_interval_s: 60
vlm_model: Qwen/Qwen2-VL-2B-Instruct
paths:
  data_root: {data}
  ckpt_root: {ckpt}
  mirror_root: {mirror}
corpus:
  frames: 8
  categories: 4
  holdout_category_fraction: 0.25
  shard_size: 4
  image_hw: [48, 64]
lora:
  rank: 4
  batch_size: 1
  grad_accum: 1
  max_steps: 2
  learning_rate: 1.0e-4
"""
    )
    return Profile.load(str(yaml_path))


class TestDetectorFactory:
    def test_gt_dispatch(self, tiny_profile: Profile):
        det = DetectorFactory(tiny_profile).create({"detector": "gt"})
        assert isinstance(det, GtDetector)
        assert det.name == "gt"

    def test_unknown_raises(self, tiny_profile: Profile):
        with pytest.raises(ValueError, match="unknown detector"):
            DetectorFactory(tiny_profile).create({"detector": "nope"})

    def test_adapter_requires_adapter_from(self, tiny_profile: Profile):
        with pytest.raises(ValueError, match="adapter_from"):
            DetectorFactory(tiny_profile).create({"detector": "vlm+adapter"})


class TestAdapterResolver:
    def test_missing_pointer(self, tmp_path: Path):
        with pytest.raises(AdapterNotFoundError, match="no LATEST"):
            AdapterResolver(tmp_path).resolve("lora-missing")

    def test_missing_adapter_dir(self, tmp_path: Path):
        exp = tmp_path / "lora-x"
        step = exp / "step_000000001"
        step.mkdir(parents=True)
        (exp / "LATEST").write_text("step_000000001\n")
        with pytest.raises(AdapterNotFoundError, match="no adapter dir"):
            AdapterResolver(tmp_path).resolve("lora-x")

    def test_resolves(self, tmp_path: Path):
        exp = tmp_path / "lora-x"
        adapter = exp / "best" / "adapter"
        adapter.mkdir(parents=True)
        (exp / "LATEST").write_text("best\n")
        assert AdapterResolver(tmp_path).resolve("lora-x") == adapter
