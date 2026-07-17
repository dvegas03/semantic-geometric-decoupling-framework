# Minimal PointNet encoder for 3-D point clouds.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _PointNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        per_point = self.mlp(x)
        global_feat = per_point.max(dim=2).values
        return per_point, global_feat


class PointNetEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.backbone = _PointNetBackbone()
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        per_point, glob = self.backbone(x)
        embedding = self.head(glob)
        return embedding, per_point


class ShapeEncoder:
    def __init__(
        self,
        embed_dim: int = 256,
        device: str | None = None,
        weights_path: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = PointNetEncoder(embed_dim=embed_dim).to(self.device).eval()
        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

    @torch.no_grad()
    def _forward(self, points: np.ndarray):
        pts = torch.from_numpy(points.astype(np.float32))
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)
        x = pts.permute(0, 2, 1).to(self.device)
        emb, pp = self.model(x)
        return emb, pp

    def encode(self, points: np.ndarray) -> np.ndarray:
        emb, _ = self._forward(points)
        return emb.squeeze(0).cpu().numpy()

    def encode_per_point(self, points: np.ndarray) -> np.ndarray:
        _, pp = self._forward(points)
        return pp.squeeze(0).permute(1, 0).cpu().numpy()
