# HuggingFace Qwen2-VL wrapper with spatial-context fusion interface.

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from engine_grounder.spatial.lifting import BBox2D

_BBOX_PROMPT_TEMPLATE = (
    "Locate the object described as: '{query}'. "
    "Respond with ONLY a JSON object of the form "
    '{{"bbox_2d": [x_min, y_min, x_max, y_max]}} '
    "using pixel coordinates in the {width}x{height} image."
)


class VLMAgent:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.processor: Any = None

        self._embedding: np.ndarray | None = None
        self._shape_text: str | None = None
        self._z_est: float | None = None

    def set_spatial_context(self, embedding: np.ndarray, shape_text: str, z_est: float) -> None:
        self._embedding = embedding
        self._shape_text = shape_text
        self._z_est = z_est

    def has_spatial_context(self) -> bool:
        return self._embedding is not None

    def load_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device = self._resolve_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float16
        if device == "cpu":
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=dtype
        )
        self.model.to(device)  # type: ignore[arg-type]
        self.model.eval()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def query(self, image: np.ndarray, prompt: str) -> str:
        self.load_model()
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        if self.has_spatial_context():
            prompt = (
                f"Geometric context: {self._shape_text} "
                f"Estimated distance: {self._z_est:.2f} m.\n{prompt}"
            )

        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        decoded: str = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded

    def detect(self, image: np.ndarray, object_query: str) -> BBox2D | None:
        height, width = image.shape[:2]
        prompt = _BBOX_PROMPT_TEMPLATE.format(query=object_query, width=width, height=height)
        response = self.query(image, prompt)
        coords = self._extract_bbox(response)
        if coords is None:
            return None
        return self._to_pixel_bbox(coords, width, height)

    @staticmethod
    def _extract_bbox(response: str) -> tuple[float, float, float, float] | None:
        native = re.search(r"\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)", response)
        if native:
            x1, y1, x2, y2 = (float(g) for g in native.groups())
            return (x1, y1, x2, y2)

        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        coords = payload.get("bbox_2d") or payload.get("bbox")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(c) for c in coords)
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _to_pixel_bbox(
        coords: tuple[float, float, float, float], width: int, height: int
    ) -> BBox2D | None:
        x1, y1, x2, y2 = coords
        if max(coords) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        elif x2 > width or y2 > height:
            x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
            y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
        try:
            return BBox2D(
                x_min=int(round(x1)),
                y_min=int(round(y1)),
                x_max=int(round(x2)),
                y_max=int(round(y2)),
            ).clamped(width, height)
        except ValueError:
            return None
