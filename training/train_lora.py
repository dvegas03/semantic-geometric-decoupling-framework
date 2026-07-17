# Qwen2-VL LoRA grounding fine-tune (2B locally, 7B on Vista via profile).

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from engine_grounder.perception.vlm_agent import _BBOX_PROMPT_TEMPLATE
from engine_grounder.spatial.lifting import BBox2D
from training.base import Trainer, TrainerOutcome

log = logging.getLogger("training.train_lora")


class MetricsLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields: object) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fields) + "\n")


@dataclass
class LossDivergenceGuard:
    ema_alpha: float = 0.9
    patience: int = 25
    rel_tol: float = 0.02
    warmup_steps: int = 40
    max_rollbacks: int = 3

    ema: float | None = None
    best_ema: float | None = None
    best_step: int = 0
    worsening: int = 0
    rollbacks: int = 0

    def observe(self, step: int, loss: float) -> str:
        if self.ema is None:
            self.ema = float(loss)
        else:
            a = self.ema_alpha
            self.ema = a * self.ema + (1.0 - a) * float(loss)

        assert self.ema is not None
        if self.best_ema is None or self.ema < self.best_ema:
            self.best_ema = self.ema
            self.best_step = step
            self.worsening = 0
            return "new_best"

        if step < self.warmup_steps:
            return "ok"

        ceiling = self.best_ema * (1.0 + self.rel_tol)
        if self.ema > ceiling:
            self.worsening += 1
        else:
            self.worsening = 0

        if self.worsening < self.patience:
            return "ok"
        if self.rollbacks >= self.max_rollbacks:
            return "exhausted"
        self.rollbacks += 1
        self.worsening = 0
        self.ema = self.best_ema
        return "rollback"

    def reset_after_rollback(self) -> None:
        self.worsening = 0
        self.ema = self.best_ema


class GroundingIterableDataset:
    def __init__(
        self,
        shard_dir: Path,
        seed: int = 0,
        split: str = "train",
        skip_samples: int = 0,
    ):
        self.shard_dir = Path(shard_dir)
        self.seed = seed
        self.split = split
        self.skip_samples = skip_samples
        self.shards = sorted(self.shard_dir.glob("shard-*.tar"))
        if not self.shards:
            raise FileNotFoundError(f"no shard-*.tar under {self.shard_dir}")

    def __iter__(self) -> Iterator[tuple[Image.Image, str, BBox2D]]:
        import io

        import webdataset as wds

        epoch = 0
        seen = 0
        while True:
            order = np.random.default_rng(self.seed + epoch).permutation(len(self.shards))
            for idx in order:
                shard = self.shards[int(idx)]
                dataset = wds.WebDataset(str(shard), shardshuffle=False).decode()
                for sample in dataset:
                    meta = sample["meta.json"]
                    if isinstance(meta, (bytes, bytearray, str)):
                        meta = json.loads(meta)
                    if meta.get("split", "train") != self.split:
                        continue
                    if seen < self.skip_samples:
                        seen += 1
                        continue
                    seen += 1
                    rgb = sample["rgb.png"]
                    if isinstance(rgb, Image.Image):
                        image = rgb.convert("RGB")
                    else:
                        image = Image.open(io.BytesIO(rgb)).convert("RGB")
                    bbox = BBox2D(*[int(v) for v in meta["bbox_2d"]])
                    yield image, str(meta["query_text"]), bbox
            epoch += 1


class QwenGroundingCollator:
    def __init__(self, processor: Any, image_hw: tuple[int, int]):
        self.processor = processor
        self.image_hw = image_hw

    def __call__(self, batch: list[tuple[Image.Image, str, BBox2D]]) -> Any:
        texts, images, prompt_lens = [], [], []
        for rgb, query, bbox in batch:
            answer = json.dumps({"bbox_2d": [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]})
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rgb},
                        {
                            "type": "text",
                            "text": _BBOX_PROMPT_TEMPLATE.format(
                                query=query,
                                width=self.image_hw[1],
                                height=self.image_hw[0],
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            prompt_only = self.processor.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True
            )
            prompt_lens.append(len(self.processor.tokenizer(prompt_only).input_ids))
            images.append(rgb)

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = inputs.input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[inputs.input_ids == pad_id] = -100
        for row, plen in enumerate(prompt_lens):
            labels[row, :plen] = -100
        inputs["labels"] = labels
        return inputs


class LoRATrainer(Trainer):
    def run(self) -> TrainerOutcome:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model

        run_start = time.monotonic()
        cfg = self.profile.lora
        max_steps = self._config_override("max_steps", cfg.max_steps, int)
        lora_rank = self._config_override("rank", cfg.rank, int)
        learning_rate = self._config_override("learning_rate", cfg.learning_rate, float)

        base_model, processor, local_model_dir = self._load_base_model()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        latest = self.checkpoints.latest()
        adapter_dir = None if latest is None else latest / "adapter"
        model: Any
        if adapter_dir is not None and adapter_dir.exists():
            model = PeftModel.from_pretrained(
                base_model,
                str(adapter_dir),
                is_trainable=True,
                local_files_only=True,
            )
            log.info("resumed from step … (adapter loaded from %s)", adapter_dir)
        else:
            model = get_peft_model(
                base_model,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=2 * lora_rank,
                    lora_dropout=0.05,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                ),
            )
        for peft_cfg in model.peft_config.values():
            peft_cfg.base_model_name_or_path = local_model_dir
        self._disable_use_cache(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

        step, samples_seen = self._try_resume(model, optimizer, scheduler, latest)
        guard = LossDivergenceGuard(
            ema_alpha=cfg.loss_ema,
            patience=cfg.loss_patience,
            rel_tol=cfg.loss_rel_tol,
            warmup_steps=cfg.loss_warmup_steps,
            max_rollbacks=cfg.loss_max_rollbacks,
        )
        best_dir = self.checkpoints.root / "best"
        if best_dir.is_dir() and (best_dir / "train_state.pt").exists():
            try:
                best_state = torch.load(
                    best_dir / "train_state.pt", map_location="cpu", weights_only=False
                )
                guard.best_ema = best_state.get("best_ema")
                guard.best_step = int(best_state.get("step", 0))
                guard.rollbacks = int(best_state.get("rollbacks", 0))
                if guard.best_ema is not None:
                    guard.ema = float(guard.best_ema)
            except Exception:
                log.exception("could not restore divergence guard state from best/")

        loader_iter = iter(self._build_loader(processor, skip_samples=samples_seen))
        metrics = MetricsLog(self.checkpoints.root / "metrics.jsonl")
        last_save = time.monotonic()
        last_best_flush = -(10**9)
        device = next(model.parameters()).device
        from training.telemetry import GpuTelemetryProbe

        probe = GpuTelemetryProbe(device)

        while step < max_steps:
            if self.stop_event.is_set():
                self._save(model, optimizer, scheduler, step, samples_seen, guard)
                return TrainerOutcome.PREEMPTED

            batch = next(loader_iter)
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            loss = model(**batch, use_cache=False).loss / cfg.grad_accum
            loss.backward()
            samples_seen += cfg.batch_size
            if samples_seen % (cfg.batch_size * cfg.grad_accum) == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_val = float(loss.detach().item()) * cfg.grad_accum
                action = guard.observe(step, loss_val)
                metrics.log(
                    step=step,
                    loss=loss_val,
                    lr=scheduler.get_last_lr()[0],
                    grad_norm=grad_norm,
                    ema_loss=guard.ema,
                    best_ema=guard.best_ema,
                    event=action if action != "ok" else None,
                    **probe.sample(),
                )
                if step % 10 == 0:
                    log.info(
                        "lora step %d/%d loss=%.4f ema=%.4f best=%.4f",
                        step,
                        max_steps,
                        loss_val,
                        guard.ema if guard.ema is not None else loss_val,
                        guard.best_ema if guard.best_ema is not None else loss_val,
                    )

                if action == "new_best":
                    if step - last_best_flush >= 10:
                        self._save_best(model, optimizer, scheduler, step, samples_seen, guard)
                        last_best_flush = step
                elif action == "rollback":
                    step, samples_seen = self._rollback(
                        model, optimizer, scheduler, guard, cfg.loss_lr_decay
                    )
                    loader_iter = iter(self._build_loader(processor, skip_samples=samples_seen))
                    metrics.log(
                        step=step,
                        event="rollback",
                        rollbacks=guard.rollbacks,
                        lr=scheduler.get_last_lr()[0],
                        best_ema=guard.best_ema,
                    )
                    last_save = time.monotonic()
                    last_best_flush = step
                    continue
                elif action == "exhausted":
                    log.warning(
                        "loss still rising after %d rollbacks — restoring best and finishing",
                        guard.rollbacks,
                    )
                    step, samples_seen = self._rollback(
                        model, optimizer, scheduler, guard, lr_decay=1.0
                    )
                    self._save(model, optimizer, scheduler, step, samples_seen, guard)
                    if best_dir.is_dir():
                        self.checkpoints.point_to("best")
                    self._record_result(lora_rank, learning_rate, max_steps, guard, run_start)
                    return TrainerOutcome.COMPLETED

            if time.monotonic() - last_save > self.profile.checkpoint_interval_s:
                self._save(model, optimizer, scheduler, step, samples_seen, guard)
                last_save = time.monotonic()

        if guard.best_step > last_best_flush and guard.best_step == step:
            self._save_best(model, optimizer, scheduler, step, samples_seen, guard)
        self._save(model, optimizer, scheduler, step, samples_seen, guard)
        if best_dir.is_dir():
            self.checkpoints.point_to("best")
            log.info("pointed LATEST at best/ (step %d) for eval", guard.best_step)
        self._record_result(lora_rank, learning_rate, max_steps, guard, run_start)
        return TrainerOutcome.COMPLETED

    def _config_override(self, key: str, default: Any, cast_fn: Any) -> Any:
        return cast_fn(self.spec.config[key]) if key in self.spec.config else default

    def _record_result(
        self,
        rank: int,
        learning_rate: float,
        max_steps: int,
        guard: LossDivergenceGuard,
        run_start: float,
    ) -> None:
        try:
            from training.results import ExperimentResult, ResultStore

            store = ResultStore(self.profile.paths.ckpt_root / "results.db")
            store.record(
                ExperimentResult(
                    experiment=self.spec.name,
                    hypothesis=str(self.spec.config.get("hypothesis", "H8")),
                    kind="lora",
                    config={
                        "rank": rank,
                        "learning_rate": learning_rate,
                        "max_steps": max_steps,
                    },
                    metrics={
                        "ema_loss": float(guard.best_ema) if guard.best_ema is not None else 0.0,
                    },
                    status="done",
                    wall_seconds=time.monotonic() - run_start,
                )
            )
        except Exception:
            log.exception("failed to record result into ResultStore (non-fatal)")

    def _load_base_model(self):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model_name = self.profile.vlm_model
        device = self.profile.device if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
            local_files_only=False,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
            local_files_only=False,
        )
        if device != "cuda":
            model = model.to(device)
        self._disable_use_cache(model)
        model.gradient_checkpointing_enable()
        model.train()
        from transformers.utils import cached_file

        config_path = cached_file(model_name, "config.json", local_files_only=True)
        if config_path is None:
            config_path = cached_file(model_name, "config.json", local_files_only=False)
        local_model_dir = str(Path(config_path).parent)
        return model, processor, local_model_dir

    @staticmethod
    def _disable_use_cache(model) -> None:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            with contextlib.suppress(Exception):
                cfg.use_cache = False
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                with contextlib.suppress(Exception):
                    text_cfg.use_cache = False
        gen = getattr(model, "generation_config", None)
        if gen is not None:
            with contextlib.suppress(Exception):
                gen.use_cache = False

    def _try_resume(self, model, optimizer, scheduler, latest: Path | None) -> tuple[int, int]:
        import torch

        if latest is None or not (latest / "train_state.pt").exists():
            return 0, 0

        path = latest / "train_state.pt"
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = int(state["step"])
            samples_seen = int(state["samples_seen"])
            rng = state.get("rng", {})
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "cuda" in rng and torch.cuda.is_available() and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
        except Exception:
            log.exception(
                "failed to restore train_state from %s — continuing with adapter weights only",
                path,
            )
            return 0, 0
        log.info("resumed from step %d (samples_seen=%d)", step, samples_seen)
        return step, samples_seen

    def _build_loader(self, processor, skip_samples: int = 0):
        from torch.utils.data import DataLoader, IterableDataset

        class _TorchIterable(IterableDataset):
            def __init__(self, inner: GroundingIterableDataset):
                self.inner = inner

            def __iter__(self):
                return iter(self.inner)

        dataset = GroundingIterableDataset(
            self.profile.paths.data_root,
            seed=0,
            split="train",
            skip_samples=skip_samples,
        )
        collator = QwenGroundingCollator(
            processor,
            (int(self.profile.corpus.image_hw[0]), int(self.profile.corpus.image_hw[1])),
        )
        return DataLoader(
            _TorchIterable(dataset),
            batch_size=self.profile.lora.batch_size,
            collate_fn=collator,
            num_workers=0,
        )

    def _train_state_dict(
        self,
        optimizer,
        scheduler,
        step: int,
        samples_seen: int,
        guard: LossDivergenceGuard | None = None,
    ) -> dict[str, object]:
        import torch

        state: dict[str, object] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "samples_seen": samples_seen,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                "numpy": np.random.get_state(),
            },
        }
        if guard is not None:
            state["best_ema"] = guard.best_ema
            state["rollbacks"] = guard.rollbacks
        return state

    def _save(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        samples_seen: int,
        guard: LossDivergenceGuard | None = None,
    ) -> None:
        import torch

        def write(d: Path) -> None:
            model.save_pretrained(d / "adapter", save_embedding_layers=False)
            torch.save(
                self._train_state_dict(optimizer, scheduler, step, samples_seen, guard),
                d / "train_state.pt",
            )

        self.checkpoints.save(max(step, 0), write)
        log.info("checkpointed at step %d", step)

    def _save_best(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        samples_seen: int,
        guard: LossDivergenceGuard,
    ) -> None:
        import torch

        root = self.checkpoints.root
        final = root / "best"
        tmp = root / "best.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        model.save_pretrained(tmp / "adapter", save_embedding_layers=False)
        torch.save(
            self._train_state_dict(optimizer, scheduler, step, samples_seen, guard),
            tmp / "train_state.pt",
        )
        if final.exists():
            stale = root / "best.stale"
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
            os.replace(final, stale)
            try:
                os.replace(tmp, final)
            finally:
                shutil.rmtree(stale, ignore_errors=True)
        else:
            os.replace(tmp, final)
        log.debug(
            "new best ema=%.4f at step %d — saved best/",
            guard.best_ema if guard.best_ema is not None else float("nan"),
            step,
        )

    def _load_adapter_into(self, model, adapter_dir: Path) -> None:
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        weights = load_peft_weights(str(adapter_dir))
        set_peft_model_state_dict(model, weights)

    def _rollback(
        self,
        model,
        optimizer,
        scheduler,
        guard: LossDivergenceGuard,
        lr_decay: float,
    ) -> tuple[int, int]:
        import torch

        best = self.checkpoints.root / "best"
        if not best.is_dir() or not (best / "train_state.pt").exists():
            log.error("rollback requested but best/ missing — continuing without restore")
            guard.reset_after_rollback()
            return 0, 0

        self._load_adapter_into(model, best / "adapter")
        state = torch.load(best / "train_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * lr_decay
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [lr * lr_decay for lr in scheduler.base_lrs]
        step = int(state["step"])
        samples_seen = int(state["samples_seen"])
        rng = state.get("rng", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available() and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        guard.reset_after_rollback()
        self.checkpoints.point_to("best")
        log.warning(
            "loss diverged — rollback #%d to step %d (ema≈%.4f), lr*=%.2f → %.2e",
            guard.rollbacks,
            step,
            guard.best_ema if guard.best_ema is not None else float("nan"),
            lr_decay,
            optimizer.param_groups[0]["lr"],
        )
        return step, samples_seen
