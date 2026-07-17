# Tests for the CanarySuite preflight.

from __future__ import annotations

import pytest

from training.canary import CanaryCheck, CanarySuite, SystemHalt


class TestCanarySuite:
    def test_all_checks_passing_does_not_raise(self):
        suite = CanarySuite(
            (
                CanaryCheck(name="a", run=lambda: True, description="always ok"),
                CanaryCheck(name="b", run=lambda: True, description="always ok"),
            )
        )
        suite.assert_healthy()

    def test_one_failing_check_halts(self):
        suite = CanarySuite(
            (
                CanaryCheck(name="ok", run=lambda: True, description="fine"),
                CanaryCheck(name="broken", run=lambda: False, description="broken metric"),
            )
        )
        with pytest.raises(SystemHalt, match="broken"):
            suite.assert_healthy()

    def test_check_raising_an_exception_counts_as_failure(self):
        def _boom() -> bool:
            raise RuntimeError("simulated crash")

        suite = CanarySuite((CanaryCheck(name="crashy", run=_boom, description="crashes"),))
        with pytest.raises(SystemHalt, match="crashy"):
            suite.assert_healthy()

    def test_empty_checks_rejected(self):
        with pytest.raises(ValueError):
            CanarySuite(())

    def test_default_suite_passes_on_the_fixed_pipeline(self):
        CanarySuite.default().assert_healthy()
