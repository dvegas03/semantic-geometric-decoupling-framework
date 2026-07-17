# EvalJob -> ResultStore wiring (progress reporting).

from __future__ import annotations

import threading

import yaml

from training.checkpoint import CheckpointManager
from training.eval import EvalJob
from training.profile import Profile
from training.queue import ExperimentSpec
from training.results import ResultStore


def _profile(tmp_path) -> Profile:
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
            "frames": 4,
            "categories": 4,
            "holdout_category_fraction": 0.25,
            "shard_size": 2,
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


def _job(tmp_path, config: dict) -> EvalJob:
    profile = _profile(tmp_path)
    spec = ExperimentSpec(name="eval-x", kind="eval", config=config)
    ck = CheckpointManager(profile.paths.ckpt_root / "eval-x")
    return EvalJob(spec, profile, ck, threading.Event())


class TestInferHypothesis:
    def test_sunrgbd_source_is_h7(self):
        assert EvalJob._infer_hypothesis({"source": "sunrgbd", "detector": "gt"}) == "H7"

    def test_gt_detector_is_h2(self):
        assert EvalJob._infer_hypothesis({"detector": "gt"}) == "H2"

    def test_vlm_detector_is_h1(self):
        assert EvalJob._infer_hypothesis({"detector": "vlm"}) == "H1"
        assert EvalJob._infer_hypothesis({"detector": "vlm+adapter"}) == "H1"

    def test_explicit_hypothesis_overrides_inference(self, tmp_path):
        job = _job(tmp_path, {"detector": "gt", "hypothesis": "H4"})
        aggregates = {
            "gt|synthetic|full_sev>=0.0": {
                "detector": "gt",
                "grounding_acc@0.5": 0.9,
                "median_iou_3d": 0.6,
                "n": 10,
            }
        }
        job._record_result(job.spec.config, aggregates)
        store = ResultStore(job.profile.paths.ckpt_root / "results.db")
        results = store.all("H4")
        assert len(results) == 1
        assert results[0].experiment == "eval-x"


class TestRecordResult:
    def test_flattens_aggregates_into_metrics(self, tmp_path):
        job = _job(tmp_path, {"detector": "gt", "lifting": "both"})
        aggregates = {
            "gt|synthetic|full_sev>=0.0": {
                "detector": "gt",
                "grounding_acc@0.5": 1.0,
                "median_iou_2d": 1.0,
                "median_z_err_cm": 3.68,
                "median_iou_3d": 0.46,
                "n": 580,
            },
            "gt|synthetic|naive_sev>=0.0": {
                "detector": "gt",
                "grounding_acc@0.5": 1.0,
                "median_iou_2d": 1.0,
                "median_z_err_cm": 3.26,
                "median_iou_3d": 0.30,
                "n": 580,
            },
        }
        job._record_result(job.spec.config, aggregates)
        store = ResultStore(job.profile.paths.ckpt_root / "results.db")
        result = store.all("H2")[0]
        assert result.metrics["gt|synthetic|full_sev>=0.0__median_iou_3d"] == 0.46
        assert result.metrics["gt|synthetic|naive_sev>=0.0__median_z_err_cm"] == 3.26

    def test_never_raises_on_store_failure(self, tmp_path, monkeypatch):
        job = _job(tmp_path, {"detector": "gt"})

        import training.results as results_module

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated disk failure")

        monkeypatch.setattr(results_module, "ResultStore", _boom)
        job._record_result(job.spec.config, {})
