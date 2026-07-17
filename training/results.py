# Append-only experiment ledger — the evolving loop's long-term memory.

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentResult:
    experiment: str
    hypothesis: str
    kind: str
    config: dict[str, Any]
    metrics: dict[str, float]
    status: str = "done"
    wall_seconds: float = 0.0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.status not in ("done", "failed", "pruned"):
            raise ValueError(f"invalid status {self.status!r}")

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)


class ResultStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                kind TEXT NOT NULL,
                config_json TEXT NOT NULL,
                config_hash TEXT NOT NULL UNIQUE,
                metrics_json TEXT NOT NULL,
                status TEXT NOT NULL,
                wall_seconds REAL NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_hypothesis ON results(hypothesis)")
        conn.commit()

    def record(self, result: ExperimentResult) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO results
                (experiment, hypothesis, kind, config_json, config_hash,
                 metrics_json, status, wall_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(config_hash) DO UPDATE SET
                experiment=excluded.experiment, hypothesis=excluded.hypothesis,
                kind=excluded.kind, metrics_json=excluded.metrics_json,
                status=excluded.status, wall_seconds=excluded.wall_seconds,
                created_at=excluded.created_at
            """,
            (
                result.experiment,
                result.hypothesis,
                result.kind,
                json.dumps(result.config, sort_keys=True),
                result.config_hash,
                json.dumps(result.metrics),
                result.status,
                result.wall_seconds,
                result.created_at,
            ),
        )
        conn.commit()

    def seen(self, cfg_hash: str) -> bool:
        cur = self._conn().execute(
            "SELECT 1 FROM results WHERE config_hash = ? LIMIT 1", (cfg_hash,)
        )
        return cur.fetchone() is not None

    def count(self, hypothesis: str | None = None) -> int:
        if hypothesis is None:
            cur = self._conn().execute("SELECT COUNT(*) FROM results")
        else:
            cur = self._conn().execute(
                "SELECT COUNT(*) FROM results WHERE hypothesis = ?", (hypothesis,)
            )
        return int(cur.fetchone()[0])

    def best(self, hypothesis: str, metric: str, maximize: bool = True) -> ExperimentResult | None:
        frontier = self.frontier(hypothesis, metric, k=1, maximize=maximize)
        return frontier[0] if frontier else None

    def frontier(
        self, hypothesis: str, metric: str, k: int, maximize: bool = True
    ) -> list[ExperimentResult]:
        rows = self._rows_for(hypothesis)
        scored = [(r, r.metrics[metric]) for r in rows if metric in r.metrics]
        scored.sort(key=lambda rs: rs[1], reverse=maximize)
        return [r for r, _ in scored[:k]]

    def all(self, hypothesis: str | None = None) -> list[ExperimentResult]:
        if hypothesis is None:
            cur = self._conn().execute("SELECT * FROM results ORDER BY created_at")
        else:
            cur = self._conn().execute(
                "SELECT * FROM results WHERE hypothesis = ? ORDER BY created_at", (hypothesis,)
            )
        return [self._row_to_result(row) for row in cur.fetchall()]

    def _rows_for(self, hypothesis: str) -> list[ExperimentResult]:
        return self.all(hypothesis)

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> ExperimentResult:
        return ExperimentResult(
            experiment=row["experiment"],
            hypothesis=row["hypothesis"],
            kind=row["kind"],
            config=json.loads(row["config_json"]),
            metrics=json.loads(row["metrics_json"]),
            status=row["status"],
            wall_seconds=row["wall_seconds"],
            created_at=row["created_at"],
        )
