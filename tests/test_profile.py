# Tests for Profile loading and path expansion.

from __future__ import annotations

import pytest
import yaml

from training.profile import CorpusProfile, Profile


class TestProfile:
    def test_load_local_by_name(self):
        profile = Profile.load("local")
        assert profile.name == "local"
        assert profile.vlm_model == "Qwen/Qwen2-VL-2B-Instruct"
        assert profile.corpus.frames == 3000
        assert profile.corpus.image_hw == (480, 640)
        assert profile.lora.batch_size == 1

    def test_load_vista_schema(self):
        profile = Profile.load("vista")
        assert profile.name == "vista"
        assert "7B" in profile.vlm_model
        assert profile.corpus.frames == 400000
        assert profile.checkpoint_interval_s == 900

    def test_var_default_expansion(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PILOT_ROOT", raising=False)
        profile = Profile.load("local")
        assert "data/pilot" in str(profile.paths.data_root)

        monkeypatch.setenv("PILOT_ROOT", str(tmp_path / "custom"))
        profile2 = Profile.load("local")
        assert profile2.paths.data_root == tmp_path / "custom" / "corpus"
        assert profile2.paths.ckpt_root == tmp_path / "custom" / "ckpts"

    def test_explicit_yaml_path(self, tmp_path):
        raw = {
            "name": "tiny",
            "device": "cpu",
            "dataloader_workers": 0,
            "checkpoint_interval_s": 10,
            "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
            "paths": {
                "data_root": str(tmp_path / "data"),
                "ckpt_root": str(tmp_path / "ckpt"),
                "mirror_root": str(tmp_path / "mirror"),
            },
            "corpus": {
                "frames": 8,
                "categories": 4,
                "holdout_category_fraction": 0.25,
                "shard_size": 4,
                "image_hw": [64, 64],
            },
            "lora": {
                "rank": 4,
                "batch_size": 1,
                "grad_accum": 1,
                "max_steps": 2,
                "learning_rate": 1e-4,
            },
        }
        path = tmp_path / "tiny.yaml"
        path.write_text(yaml.safe_dump(raw))
        profile = Profile.load(str(path))
        assert profile.name == "tiny"
        assert profile.corpus.frames == 8

    def test_holdout_fraction_validation(self):
        with pytest.raises(ValueError):
            CorpusProfile(
                frames=10,
                categories=5,
                holdout_category_fraction=0.0,
                shard_size=2,
                image_hw=(32, 32),
            )
        with pytest.raises(ValueError):
            CorpusProfile(
                frames=0,
                categories=5,
                holdout_category_fraction=0.2,
                shard_size=2,
                image_hw=(32, 32),
            )
