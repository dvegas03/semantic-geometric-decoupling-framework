# GPU telemetry probe — appends device stats to each trainer's metrics stream.

from __future__ import annotations

import torch


class GpuTelemetryProbe:
    def __init__(self, device: torch.device):
        self._enabled = device.type == "cuda" and torch.cuda.is_available()
        if self._enabled:
            torch.cuda.reset_peak_memory_stats(device)
        self._device = device

    def sample(self) -> dict[str, float]:
        if not self._enabled:
            return {}
        return {
            "vram_alloc_gb": torch.cuda.memory_allocated(self._device) / 2**30,
            "vram_peak_gb": torch.cuda.max_memory_allocated(self._device) / 2**30,
            "vram_reserved_gb": torch.cuda.memory_reserved(self._device) / 2**30,
        }
