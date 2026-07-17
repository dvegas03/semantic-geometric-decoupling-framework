# Eval aggregate metrics including grounding_acc@0.5 (H1).

from __future__ import annotations

import pytest

from training.eval import EvalReport, EvalRow, GroundingEvaluator


def _row(**kwargs) -> EvalRow:
    base = {
        "detector": "gt",
        "category": "box",
        "split": "heldout",
        "source": "synthetic",
        "severity": 0.6,
        "lifter": "full",
        "iou_2d": 0.7,
        "z_err_cm": 5.0,
        "iou_3d": 0.4,
        "success": True,
    }
    base.update(kwargs)
    return EvalRow(**base)  # type: ignore[arg-type]


class TestEvalAggregate:
    def test_grounding_acc_at_half(self):
        rows = [
            _row(iou_2d=0.8, success=True),
            _row(iou_2d=0.3, success=True),
            _row(iou_2d=0.0, success=False),
        ]
        agg = GroundingEvaluator._aggregate(rows)
        key = "gt|synthetic|full_sev>=0.0"
        assert key in agg
        assert agg[key]["grounding_acc@0.5"] == pytest.approx(1 / 3)
        assert agg[key]["detector"] == "gt"
        assert agg[key]["n"] == 3

    def test_markdown_includes_detector_and_acc(self):
        rows = [_row()]
        report = EvalReport(rows=rows, aggregates=GroundingEvaluator._aggregate(rows))
        md = report.to_markdown()
        assert "detector" in md
        assert "acc@0.5" in md
        assert "gt" in md
