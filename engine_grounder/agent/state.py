# Shared state schema for the grounding agent graph.

from __future__ import annotations

import numpy as np
from typing_extensions import TypedDict

from engine_grounder.spatial.lifting import BBox2D, LiftResult


class GroundingState(TypedDict, total=False):
    user_query: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: tuple[float, float, float, float]

    bbox_2d: BBox2D | None

    lift: LiftResult | None

    response: dict

    error: str | None
