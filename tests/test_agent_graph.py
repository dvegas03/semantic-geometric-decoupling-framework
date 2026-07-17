# Integration tests for the LangGraph grounding agent (fake VLM, no weights).

import numpy as np
import pytest

pytest.importorskip("langgraph")

from engine_grounder.agent.graph import GroundingAgent
from engine_grounder.spatial.lifting import BBox2D

INTRINSICS = (500.0, 500.0, 160.0, 120.0)


class FakeVLM:
    def __init__(self, bbox):
        self.bbox = bbox
        self.calls = []

    def detect(self, image, object_query):
        self.calls.append(object_query)
        return self.bbox


def make_frame(z: float = 2.0):
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), z, dtype=np.float32)
    return rgb, depth


class TestGroundingAgent:
    def test_happy_path(self):
        rgb, depth = make_frame(z=2.0)
        agent = GroundingAgent(vlm=FakeVLM(BBox2D(100, 80, 220, 160)))
        response = agent.ground(rgb, depth, INTRINSICS, "the red mug")
        assert response["success"] is True
        assert abs(response["z_est"] - 2.0) < 0.05
        assert len(response["vertices_3d"]) == 8

    def test_vlm_failure_short_circuits(self):
        rgb, depth = make_frame()
        agent = GroundingAgent(vlm=FakeVLM(None))
        response = agent.ground(rgb, depth, INTRINSICS, "the invisible object")
        assert response == {
            "success": False,
            "error": "vlm_detection_failed",
            "query": "the invisible object",
        }

    def test_lift_failure_reported(self):
        rgb, _ = make_frame()
        depth = np.zeros((240, 320), dtype=np.float32)
        agent = GroundingAgent(vlm=FakeVLM(BBox2D(100, 80, 220, 160)))
        response = agent.ground(rgb, depth, INTRINSICS, "the mug")
        assert response["success"] is False
        assert response["error"] == "geometric_lift_failed"
