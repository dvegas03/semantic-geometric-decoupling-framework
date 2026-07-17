# Queue-draining runner: the single process a chain job supervises.

from __future__ import annotations

import argparse
import logging
import signal
import threading
import traceback
from pathlib import Path

from training.base import TrainerOutcome, TrainerRegistry
from training.canary import CanarySuite, SystemHalt
from training.checkpoint import CheckpointManager, CheckpointMirror
from training.planner import MetaScheduler
from training.profile import Profile
from training.queue import ExperimentQueue, ExperimentStatus, default_owner_id

log = logging.getLogger("training.runner")

EXIT_OK = 0
EXIT_DRAINED = 3
EXIT_CANARY_FAILED = 4


class _Heartbeat:
    def __init__(self, queue: ExperimentQueue, name: str, owner: str, interval_s: float):
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            args=(queue, name, owner, interval_s),
            name=f"heartbeat-{name}",
            daemon=True,
        )

    def _loop(self, queue, name, owner, interval_s):
        while not self._stop.wait(interval_s):
            queue.heartbeat(name, owner)

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)


class Runner:
    def __init__(
        self,
        queue: ExperimentQueue,
        profile: Profile,
        registry: TrainerRegistry,
        poll_s: float = 30.0,
        canary: CanarySuite | None = None,
        scheduler: MetaScheduler | None = None,
    ):
        self.queue = queue
        self.profile = profile
        self.registry = registry
        self.poll_s = poll_s
        self.canary = canary
        self.scheduler = scheduler
        self.owner = default_owner_id()
        self.stop_event = threading.Event()

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        log.warning("received signal %s — requesting checkpoint-and-stop", signum)
        self.stop_event.set()

    def run(self) -> int:
        if self.canary is not None:
            try:
                self.canary.assert_healthy()
            except SystemHalt:
                log.exception("canary preflight failed — refusing to start")
                return EXIT_CANARY_FAILED

        while not self.stop_event.is_set():
            spec = self.queue.claim_next(self.owner)
            if spec is None:
                blocked = self.queue.fail_unsatisfiable()
                if blocked:
                    log.error(
                        "failed unsatisfiable deps (required experiment permanently failed): %s",
                        blocked,
                    )
                if self.scheduler is not None:
                    added = self.scheduler.maybe_refill(self.queue)
                    if added:
                        log.info("evolving loop: planner proposed %d new experiment(s)", added)
                        continue
                if self.queue.is_drained():
                    log.info("queue drained — chain may stop")
                    return EXIT_DRAINED
                snap = self.queue.snapshot()
                pending = [s.name for s in snap if s.status is ExperimentStatus.PENDING]
                if pending:
                    log.warning(
                        "queue stuck-but-not-drained; pending=%s (check requires typos)",
                        pending,
                    )
                log.info(
                    "nothing runnable (deps pending or claims live); poll in %ss",
                    self.poll_s,
                )
                self.stop_event.wait(self.poll_s)
                continue

            log.info("claimed %s (kind=%s, retries=%d)", spec.name, spec.kind, spec.retries)
            checkpoints = CheckpointManager(self.profile.paths.ckpt_root / spec.name)
            mirror = CheckpointMirror(checkpoints, self.profile.paths.mirror_root)
            mirror.start()
            try:
                trainer = self.registry.create(spec, self.profile, checkpoints, self.stop_event)
                with _Heartbeat(self.queue, spec.name, self.owner, self.queue.claim_ttl_s / 3):
                    outcome = trainer.run()
            except Exception:
                log.error("trainer %s crashed:\n%s", spec.name, traceback.format_exc())
                outcome = TrainerOutcome.FAILED
            finally:
                mirror.stop(final_sync=True)
            self.queue.release(spec.name, self.owner, outcome)
            log.info("released %s as %s", spec.name, outcome.value)
        return EXIT_OK


def _default_evolving_scheduler(profile: Profile) -> MetaScheduler:
    from training.planner import ParamSpec, SearchSpace, SuccessiveHalvingPlanner
    from training.results import ResultStore

    store = ResultStore(profile.paths.ckpt_root / "results.db")
    space = SearchSpace(
        params=(
            ParamSpec(name="rank", values=(4, 8, 16, 32)),
            ParamSpec(name="learning_rate", low=5e-5, high=5e-4, log=True),
        )
    )
    planner = SuccessiveHalvingPlanner(
        hypothesis="H8",
        kind="lora",
        base_config={},
        search_space=space,
        rungs=(250, 1000, 4000),
        eta=3,
        metric="ema_loss",
        maximize=False,
    )
    return MetaScheduler([planner], store, low_watermark=2, refill_batch=3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Drain the experiment queue.")
    parser.add_argument("--profile", required=True, help="profile name or YAML path")
    parser.add_argument("--queue", default="experiments/queue.yaml")
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument(
        "--evolve",
        action="store_true",
        help=(
            "Never drain: run the WS0/§11.4 canary preflight, then keep the "
            "queue refilled from a LoRA hyperparameter planner instead of "
            "exiting once the static queue is done."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    profile = Profile.load(args.profile)
    runner = Runner(
        queue=ExperimentQueue(Path(args.queue)),
        profile=profile,
        registry=TrainerRegistry.default(),
        poll_s=args.poll_s,
        canary=CanarySuite.default() if args.evolve else None,
        scheduler=_default_evolving_scheduler(profile) if args.evolve else None,
    )
    runner.install_signal_handlers()
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
