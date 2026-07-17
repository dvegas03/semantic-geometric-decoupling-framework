# Unit tests for VLMAgent response parsing — no model weights required.

import json

from engine_grounder.perception.vlm_agent import VLMAgent
from engine_grounder.spatial.lifting import BBox2D


class TestExtractBBox:
    def test_plain_json(self):
        assert VLMAgent._extract_bbox('{"bbox_2d": [10, 20, 110, 220]}') == (
            10,
            20,
            110,
            220,
        )

    def test_fenced_json_with_prose(self):
        response = 'The hose is here:\n```json\n{"bbox_2d": [5, 5, 50, 60]}\n```'
        assert VLMAgent._extract_bbox(response) == (5, 5, 50, 60)

    def test_qwen_native_grounding_tokens(self):
        assert VLMAgent._extract_bbox("<|box_start|>(12,34),(56,78)<|box_end|>") == (12, 34, 56, 78)

    def test_inverted_box_rejected(self):
        assert VLMAgent._extract_bbox('{"bbox_2d": [100, 20, 10, 220]}') is None

    def test_garbage_rejected(self):
        assert VLMAgent._extract_bbox("I cannot find that object.") is None


class TestToPixelBBox:
    def test_fractional_coords_scaled(self):
        box = VLMAgent._to_pixel_bbox((0.25, 0.25, 0.75, 0.75), 320, 240)
        assert box == BBox2D(80, 60, 240, 180)

    def test_permille_coords_scaled(self):
        box = VLMAgent._to_pixel_bbox((250.0, 250.0, 750.0, 750.0), 320, 240)
        assert box == BBox2D(80, 60, 240, 180)

    def test_pixel_coords_passthrough(self):
        box = VLMAgent._to_pixel_bbox((80.0, 60.0, 240.0, 180.0), 320, 240)
        assert box == BBox2D(80, 60, 240, 180)


def test_collator_answer_roundtrips_through_agent_parser():
    bbox = BBox2D(258, 121, 321, 188)
    answer = json.dumps({"bbox_2d": [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]})
    coords = VLMAgent._extract_bbox(answer)
    assert coords is not None
    parsed = VLMAgent._to_pixel_bbox(coords, width=640, height=480)
    assert parsed is not None
    assert (parsed.x_min, parsed.y_min, parsed.x_max, parsed.y_max) == (258, 121, 321, 188)
