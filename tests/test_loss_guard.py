# Unit tests for LoRA loss-divergence rollback guard.

from __future__ import annotations

from training.train_lora import LossDivergenceGuard


class TestLossDivergenceGuard:
    def test_tracks_new_best(self):
        g = LossDivergenceGuard(ema_alpha=0.5, warmup_steps=0, patience=3, rel_tol=0.05)
        assert g.observe(1, 10.0) == "new_best"
        assert g.observe(2, 8.0) == "new_best"
        assert g.best_step == 2

    def test_rollback_after_sustained_rise(self):
        g = LossDivergenceGuard(
            ema_alpha=0.0,
            warmup_steps=0,
            patience=3,
            rel_tol=0.05,
            max_rollbacks=2,
        )
        assert g.observe(1, 1.0) == "new_best"
        assert g.observe(2, 1.2) == "ok"
        assert g.observe(3, 1.2) == "ok"
        assert g.observe(4, 1.2) == "rollback"
        assert g.rollbacks == 1

    def test_exhausted_after_max_rollbacks(self):
        g = LossDivergenceGuard(
            ema_alpha=0.0,
            warmup_steps=0,
            patience=1,
            rel_tol=0.01,
            max_rollbacks=1,
        )
        assert g.observe(1, 1.0) == "new_best"
        assert g.observe(2, 2.0) == "rollback"
        g.reset_after_rollback()
        assert g.observe(3, 2.0) == "exhausted"

    def test_warmup_skips_rollback(self):
        g = LossDivergenceGuard(
            ema_alpha=0.0,
            warmup_steps=10,
            patience=1,
            rel_tol=0.01,
        )
        assert g.observe(1, 1.0) == "new_best"
        assert g.observe(2, 5.0) == "ok"
