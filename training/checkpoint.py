# Atomic checkpoint directories with a LATEST pointer and background mirror.

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

_POINTER = "LATEST"


class CheckpointManager:
    def __init__(self, root: Path, keep: int = 2):
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        self.root = Path(root)
        self.keep = keep
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, writer: Callable[[Path], None]) -> Path:
        final = self.root / f"step_{step:09d}"
        tmp = self.root / f"step_{step:09d}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        writer(tmp)
        if final.exists():
            stale = self.root / f"step_{step:09d}.stale"
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
            os.replace(final, stale)
            try:
                os.replace(tmp, final)
            finally:
                shutil.rmtree(stale, ignore_errors=True)
        else:
            os.replace(tmp, final)
        self._write_pointer(final.name)
        self._prune()
        return final

    def _write_pointer(self, name: str) -> None:
        tmp = self.root / (_POINTER + ".tmp")
        tmp.write_text(name + "\n")
        os.replace(tmp, self.root / _POINTER)

    def latest(self) -> Path | None:
        pointer = self.root / _POINTER
        if not pointer.exists():
            return None
        target = self.root / pointer.read_text().strip()
        return target if target.is_dir() else None

    def point_to(self, name: str) -> Path:
        target = self.root / name
        if not target.is_dir():
            raise FileNotFoundError(f"checkpoint {name!r} not found under {self.root}")
        self._write_pointer(name)
        return target

    def _prune(self) -> None:
        for orphan in self.root.glob("step_*.tmp"):
            shutil.rmtree(orphan, ignore_errors=True)
        complete = sorted(d for d in self.root.glob("step_*") if d.is_dir())
        pointed = self.latest()
        for stale in complete[: -self.keep]:
            if stale != pointed:
                shutil.rmtree(stale, ignore_errors=True)


class CheckpointMirror:
    def __init__(
        self,
        manager: CheckpointManager,
        dest_root: Path,
        interval_s: float = 3600.0,
    ):
        self.manager = manager
        self.dest_root = Path(dest_root)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.dest_root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="ckpt-mirror", daemon=True)
        self._thread.start()

    def stop(self, final_sync: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        if final_sync:
            self.sync_once()

    def sync_once(self) -> None:
        latest = self.manager.latest()
        if latest is None:
            return
        dest = self.dest_root / self.manager.root.name / latest.name
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                str(latest) + "/",
                str(dest),
            ],
            check=False,
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sync_once()
