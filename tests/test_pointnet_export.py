# Round-trip test for the V7 contract: best-model export loads into ShapeEncoder.

from __future__ import annotations

import torch

from engine_grounder.perception.shape_encoder import PointNetEncoder
from training.train_pointnet import BestModelTracker, PointNetClassifier


def test_export_loads_into_shape_encoder(tmp_path):
    model = PointNetClassifier(num_classes=40)
    tracker = BestModelTracker()
    tracker.observe(model, acc=0.5)
    path = tracker.export(tmp_path)
    encoder = PointNetEncoder(embed_dim=256)
    encoder.load_state_dict(torch.load(path, weights_only=True))
    emb, _ = encoder(torch.randn(2, 3, 1024))
    assert emb.shape == (2, 256)
