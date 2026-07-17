# Tests for the ResultStore experiment ledger.

from __future__ import annotations

from training.results import ExperimentResult, ResultStore, config_hash


def _result(name: str, hyp: str, metric_value: float, cfg: dict | None = None) -> ExperimentResult:
    return ExperimentResult(
        experiment=name,
        hypothesis=hyp,
        kind="lora",
        config=cfg or {"rank": 8, "name": name},
        metrics={"acc@0.5": metric_value},
        status="done",
        wall_seconds=12.0,
    )


class TestConfigHash:
    def test_order_independent(self):
        a = config_hash({"rank": 8, "lr": 1e-4})
        b = config_hash({"lr": 1e-4, "rank": 8})
        assert a == b

    def test_different_configs_differ(self):
        assert config_hash({"rank": 8}) != config_hash({"rank": 16})


class TestResultStore:
    def test_record_and_seen(self, tmp_path):
        store = ResultStore(tmp_path / "results.db")
        r = _result("exp-a", "H1", 0.5)
        assert not store.seen(r.config_hash)
        store.record(r)
        assert store.seen(r.config_hash)

    def test_never_repeats_a_config(self, tmp_path):
        store = ResultStore(tmp_path / "results.db")
        cfg = {"rank": 8, "lr": 1e-4}
        store.record(_result("exp-a", "H1", 0.5, cfg))
        assert store.seen(config_hash(cfg))
        assert store.count("H1") == 1
        store.record(_result("exp-a-resumed", "H1", 0.55, cfg))
        assert store.count("H1") == 1

    def test_best_and_frontier_rank_by_metric(self, tmp_path):
        store = ResultStore(tmp_path / "results.db")
        store.record(_result("a", "H1", 0.10, {"rank": 4}))
        store.record(_result("b", "H1", 0.55, {"rank": 8}))
        store.record(_result("c", "H1", 0.30, {"rank": 16}))

        best = store.best("H1", "acc@0.5")
        assert best is not None
        assert best.experiment == "b"

        top2 = store.frontier("H1", "acc@0.5", k=2)
        assert [r.experiment for r in top2] == ["b", "c"]

    def test_frontier_excludes_missing_metric_rows(self, tmp_path):
        store = ResultStore(tmp_path / "results.db")
        store.record(
            ExperimentResult(
                experiment="no-metric",
                hypothesis="H1",
                kind="lora",
                config={"rank": 99},
                metrics={},
                status="failed",
            )
        )
        store.record(_result("has-metric", "H1", 0.4, {"rank": 4}))
        frontier = store.frontier("H1", "acc@0.5", k=5)
        assert [r.experiment for r in frontier] == ["has-metric"]

    def test_hypotheses_are_isolated(self, tmp_path):
        store = ResultStore(tmp_path / "results.db")
        store.record(_result("h1-a", "H1", 0.9, {"rank": 1}))
        store.record(_result("h2-a", "H2", 0.1, {"rank": 2}))
        assert store.count("H1") == 1
        assert store.count("H2") == 1
        assert store.count() == 2

    def test_persists_across_reopen(self, tmp_path):
        db = tmp_path / "results.db"
        ResultStore(db).record(_result("a", "H1", 0.5, {"rank": 1}))
        reopened = ResultStore(db)
        assert reopened.count("H1") == 1
        assert reopened.best("H1", "acc@0.5").experiment == "a"
