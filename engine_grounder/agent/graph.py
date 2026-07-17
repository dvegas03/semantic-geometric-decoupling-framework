# LangGraph orchestration: semantic detection decoupled from geometric lifting.

from __future__ import annotations

from engine_grounder.agent.state import GroundingState
from engine_grounder.perception.vlm_agent import VLMAgent
from engine_grounder.spatial.lifting import BBoxLifter


class GroundingAgent:
    def __init__(self, vlm: VLMAgent, lifter: BBoxLifter | None = None):
        self.vlm = vlm
        self.lifter = lifter if lifter is not None else BBoxLifter()
        self._graph = self._build()

    def _vlm_detect(self, state: GroundingState) -> GroundingState:
        bbox = self.vlm.detect(state["rgb"], state["user_query"])
        if bbox is None:
            return {"bbox_2d": None, "error": "vlm_detection_failed"}
        return {"bbox_2d": bbox}

    def _geometric_lift(self, state: GroundingState) -> GroundingState:
        bbox = state["bbox_2d"]
        assert bbox is not None
        result = self.lifter.lift(state["depth"], state["intrinsics"], bbox)
        if result is None:
            return {"lift": None, "error": "geometric_lift_failed"}
        return {"lift": result}

    def _format_response(self, state: GroundingState) -> GroundingState:
        error = state.get("error")
        if error is not None:
            return {
                "response": {
                    "success": False,
                    "error": error,
                    "query": state["user_query"],
                }
            }
        lift = state["lift"]
        assert lift is not None
        return {
            "response": {
                "success": True,
                "query": state["user_query"],
                "bbox_2d": [
                    lift.bbox.x_min,
                    lift.bbox.y_min,
                    lift.bbox.x_max,
                    lift.bbox.y_max,
                ],
                "vertices_3d": lift.vertices.tolist(),
                "z_est": lift.z_est,
                "z_min": lift.z_min,
                "z_max": lift.z_max,
                "inlier_ratio": lift.inlier_ratio,
            }
        }

    def _build(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(GroundingState)
        graph.add_node("vlm_detect", self._vlm_detect)
        graph.add_node("geometric_lift", self._geometric_lift)
        graph.add_node("format_response", self._format_response)

        graph.add_edge(START, "vlm_detect")
        graph.add_conditional_edges(
            "vlm_detect",
            lambda s: "format_response" if s.get("error") else "geometric_lift",
            {"geometric_lift": "geometric_lift", "format_response": "format_response"},
        )
        graph.add_edge("geometric_lift", "format_response")
        graph.add_edge("format_response", END)
        return graph.compile()

    def ground(self, rgb, depth, intrinsics, user_query: str) -> dict:
        final = self._graph.invoke(
            {
                "rgb": rgb,
                "depth": depth,
                "intrinsics": intrinsics,
                "user_query": user_query,
            }
        )
        response: dict = final["response"]
        return response
