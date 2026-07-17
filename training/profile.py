# Execution profiles: one YAML per tier (local 4070 vs. Vista GH200).

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from re import Match

import yaml

_PROFILE_DIR = Path(__file__).resolve().parent.parent / "configs" / "profiles"


def _expand(raw: str) -> Path:

    def sub(match: Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        return os.environ.get(var, default if default is not None else "")

    expanded = re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", sub, raw)
    return Path(os.path.expandvars(expanded)).expanduser()


@dataclass(frozen=True)
class PathsProfile:
    data_root: Path
    ckpt_root: Path
    mirror_root: Path

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> PathsProfile:
        return cls(
            data_root=_expand(raw["data_root"]),
            ckpt_root=_expand(raw["ckpt_root"]),
            mirror_root=_expand(raw["mirror_root"]),
        )


@dataclass(frozen=True)
class CorpusProfile:
    frames: int
    categories: int
    holdout_category_fraction: float
    shard_size: int
    image_hw: tuple[int, int]

    def __post_init__(self) -> None:
        if not 0.0 < self.holdout_category_fraction < 1.0:
            raise ValueError(f"holdout fraction must be in (0, 1), got {self}")
        if self.frames <= 0 or self.shard_size <= 0:
            raise ValueError(f"frames and shard_size must be positive, got {self}")


@dataclass(frozen=True)
class LoRAProfile:
    rank: int
    batch_size: int
    grad_accum: int
    max_steps: int
    learning_rate: float
    loss_ema: float = 0.9
    loss_patience: int = 25
    loss_rel_tol: float = 0.02
    loss_warmup_steps: int = 40
    loss_lr_decay: float = 0.5
    loss_max_rollbacks: int = 3


@dataclass(frozen=True)
class Profile:
    name: str
    device: str
    dataloader_workers: int
    checkpoint_interval_s: float
    vlm_model: str
    paths: PathsProfile
    corpus: CorpusProfile
    lora: LoRAProfile

    @classmethod
    def load(cls, name_or_path: str) -> Profile:
        path = Path(name_or_path)
        if path.suffix != ".yaml":
            path = _PROFILE_DIR / f"{name_or_path}.yaml"
        raw = yaml.safe_load(path.read_text())
        corpus_raw = dict(raw["corpus"])
        corpus_raw["image_hw"] = tuple(corpus_raw["image_hw"])
        return cls(
            name=raw["name"],
            device=raw["device"],
            dataloader_workers=int(raw["dataloader_workers"]),
            checkpoint_interval_s=float(raw["checkpoint_interval_s"]),
            vlm_model=raw["vlm_model"],
            paths=PathsProfile.from_dict(raw["paths"]),
            corpus=CorpusProfile(**corpus_raw),
            lora=LoRAProfile(**raw["lora"]),
        )
