# Quantitative metric functions for the Semantic-Geometric Decoupling Framework pipeline.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engine_grounder.spatial.lifting import BBox2D


def z_error_cm(z_est: float, z_true: float) -> float:
    return abs(z_est - z_true) * 100.0


def depth_rmse(
    depth_pred: np.ndarray,
    depth_true: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None:
        mask = (depth_pred > 0) & (depth_true > 0)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((depth_pred[mask] - depth_true[mask]) ** 2)))


def depth_mae(
    depth_pred: np.ndarray,
    depth_true: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None:
        mask = (depth_pred > 0) & (depth_true > 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(depth_pred[mask] - depth_true[mask])))


def depth_bias(
    depth_pred: np.ndarray,
    depth_true: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None:
        mask = (depth_pred > 0) & (depth_true > 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(depth_pred[mask] - depth_true[mask]))


def rmse_improvement_ratio(rmse_before: float, rmse_after: float) -> float:
    if rmse_after == 0 or np.isnan(rmse_after):
        return float("nan")
    return rmse_before / rmse_after


def outlier_metrics(
    predicted_mask: np.ndarray,
    true_outlier_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    pred = predicted_mask.astype(bool) & valid_mask.astype(bool)
    true = true_outlier_mask.astype(bool) & valid_mask.astype(bool)

    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true & valid_mask.astype(bool)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def inlier_retention_rate(
    predicted_outlier_mask: np.ndarray,
    true_outlier_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    true_inlier = valid_mask.astype(bool) & ~true_outlier_mask.astype(bool)
    if not true_inlier.any():
        return float("nan")
    false_flags = predicted_outlier_mask.astype(bool) & true_inlier
    return float(1.0 - false_flags.sum() / true_inlier.sum())


def sigma_recovery_rel_error(
    sigma_hat: float,
    sigma_true: float,
) -> float:
    if sigma_true <= 0:
        return float("nan")
    return abs(sigma_hat - sigma_true) / sigma_true


def chamfer_hausdorff(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    n_sample: int = 5_000,
    seed: int = 0,
) -> tuple[float, float]:
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)
    if len(pts_a) > n_sample:
        pts_a = pts_a[rng.choice(len(pts_a), n_sample, replace=False)]
    if len(pts_b) > n_sample:
        pts_b = pts_b[rng.choice(len(pts_b), n_sample, replace=False)]

    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan"), float("nan")

    d_a2b, _ = KDTree(pts_b).query(pts_a, k=1, workers=-1)
    d_b2a, _ = KDTree(pts_a).query(pts_b, k=1, workers=-1)

    chamfer = float(0.5 * (d_a2b.mean() + d_b2a.mean()))
    hausdorff = float(max(d_a2b.max(), d_b2a.max()))
    return chamfer, hausdorff


def bbox_iou_2d(a: BBox2D, b: BBox2D) -> float:
    ix = max(0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    iy = max(0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def aabb_iou_3d(vertices_a: np.ndarray, vertices_b: np.ndarray) -> float:
    a_min, a_max = vertices_a.min(axis=0), vertices_a.max(axis=0)
    b_min, b_max = vertices_b.min(axis=0), vertices_b.max(axis=0)
    inter = float(np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0.0, None).prod())
    vol_a = float(np.clip(a_max - a_min, 0.0, None).prod())
    vol_b = float(np.clip(b_max - b_min, 0.0, None).prod())
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def point_cloud_coverage(
    pts_reconstructed: np.ndarray,
    pts_reference: np.ndarray,
    radius: float = 0.05,
) -> float:
    from scipy.spatial import KDTree

    if len(pts_reconstructed) == 0 or len(pts_reference) == 0:
        return 0.0

    d, _ = KDTree(pts_reconstructed).query(pts_reference, k=1, workers=-1)
    return float((d <= radius).mean())
