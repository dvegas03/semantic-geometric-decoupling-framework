# Public surface for shape encoding/description and the VLM agent.

from engine_grounder.perception.shape_descriptor import ShapeDescriptor
from engine_grounder.perception.shape_encoder import PointNetEncoder, ShapeEncoder
from engine_grounder.perception.vlm_agent import VLMAgent

__all__ = ["ShapeEncoder", "PointNetEncoder", "ShapeDescriptor", "VLMAgent"]
