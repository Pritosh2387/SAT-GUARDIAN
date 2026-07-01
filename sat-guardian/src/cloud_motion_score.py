"""
SAT-GUARDIAN: Cloud Motion Consistency Score
==============================================
Quantifies how well the predicted cloud motion (from optical flow) aligns
with the ERA5 reanalysis wind vectors.

Score = cosine_similarity(predicted_motion, ERA5_motion) × 100
      → Range: 0 (perfectly opposed) to 100 (perfectly aligned)

Implementation
--------------
1. Sample motion vectors at regular spatial grid points.
2. Compute per-pixel cosine similarity between flow and wind vectors.
3. Average across all valid (non-zero-wind) grid points.
4. Scale to 0–100.
"""

import logging
from typing import Optional, Tuple, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def compute_cloud_motion_score(
    predicted_flow:   np.ndarray,
    wind_flow_px:     np.ndarray,
    spatial_sample:   int   = 8,
    min_wind_mag:     float = 0.01,
    cloud_mask:       Optional[np.ndarray] = None,
) -> float:
    """
    Compute the Cloud Motion Consistency Score (CMCS).

    CMCS = mean(cosine_similarity(u_pred, u_wind)) × 100

    Parameters
    ----------
    predicted_flow : np.ndarray  (H, W, 2) – physics-constrained flow
    wind_flow_px   : np.ndarray  (H, W, 2) – ERA5 wind in pixel units
    spatial_sample : int  sample every N pixels in each spatial dimension
    min_wind_mag   : float  ignore grid points where wind magnitude is below this
    cloud_mask     : optional bool (H, W) – restrict to cloud regions

    Returns
    -------
    float  score in [0, 100]
    """
    H, W = predicted_flow.shape[:2]

    # --- Spatial subsampling ---
    ys = np.arange(0, H, spatial_sample)
    xs = np.arange(0, W, spatial_sample)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    pred_u = predicted_flow[yy, xx, 0].ravel()   # (N,)
    pred_v = predicted_flow[yy, xx, 1].ravel()
    wind_u = wind_flow_px[yy,  xx, 0].ravel()
    wind_v = wind_flow_px[yy,  xx, 1].ravel()

    # --- Cloud mask filtering ---
    if cloud_mask is not None:
        valid_mask = cloud_mask[yy, xx].ravel().astype(bool)
    else:
        valid_mask = np.ones(len(pred_u), dtype=bool)

    # Exclude near-zero wind points (undefined direction)
    wind_mag = np.sqrt(wind_u**2 + wind_v**2)
    valid_mask = valid_mask & (wind_mag > min_wind_mag)

    if valid_mask.sum() == 0:
        logger.warning("No valid wind vectors found; returning score=0")
        return 0.0

    pred_vecs = np.stack([pred_u[valid_mask], pred_v[valid_mask]], axis=1)   # (N, 2)
    wind_vecs = np.stack([wind_u[valid_mask], wind_v[valid_mask]], axis=1)

    # --- Per-vector cosine similarity ---
    scores = _batch_cosine_similarity(pred_vecs, wind_vecs)

    # Map [-1, 1] → [0, 100]
    score = float(np.mean(scores) + 1.0) / 2.0 * 100.0

    logger.info(
        "Cloud Motion Score: %.2f/100 (valid_pts=%d, mean_cos=%.4f)",
        score, valid_mask.sum(), float(np.mean(scores)),
    )
    return round(score, 2)


def _batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Per-row cosine similarity between arrays a and b of shape (N, 2).

    Returns np.ndarray of shape (N,) with values in [-1, 1].
    """
    dot   = np.sum(a * b, axis=1)
    mag_a = np.linalg.norm(a, axis=1) + 1e-8
    mag_b = np.linalg.norm(b, axis=1) + 1e-8
    return np.clip(dot / (mag_a * mag_b), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Extended spatial analysis
# ---------------------------------------------------------------------------

def compute_spatial_consistency_map(
    predicted_flow: np.ndarray,
    wind_flow_px:   np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel cosine similarity between predicted flow and wind flow.
    Useful for visualising which spatial regions are physics-consistent.

    Returns
    -------
    np.ndarray  (H, W) float32 in [0, 1]  (0 = fully inconsistent, 1 = aligned)
    """
    pred_u = predicted_flow[..., 0]
    pred_v = predicted_flow[..., 1]
    wind_u = wind_flow_px[..., 0]
    wind_v = wind_flow_px[..., 1]

    dot   = pred_u * wind_u + pred_v * wind_v
    mag_p = np.sqrt(pred_u**2 + pred_v**2) + 1e-8
    mag_w = np.sqrt(wind_u**2 + wind_v**2) + 1e-8

    cos_sim = np.clip(dot / (mag_p * mag_w), -1.0, 1.0)
    # Normalise to [0, 1]
    spatial_map = (cos_sim + 1.0) / 2.0
    return spatial_map.astype(np.float32)


def score_report(
    predicted_flow: np.ndarray,
    wind_flow_px:   np.ndarray,
    cloud_mask:     Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Generate a comprehensive cloud motion consistency report.

    Returns
    -------
    dict with overall_score, spatial_map statistics, and interpretation
    """
    overall_score = compute_cloud_motion_score(
        predicted_flow, wind_flow_px, cloud_mask=cloud_mask
    )
    spatial_map = compute_spatial_consistency_map(predicted_flow, wind_flow_px)

    if cloud_mask is not None:
        cloud_score = compute_cloud_motion_score(
            predicted_flow, wind_flow_px, cloud_mask=cloud_mask
        )
    else:
        cloud_score = overall_score

    interpretation = (
        "Excellent – motion highly consistent with ERA5 wind" if overall_score >= 80
        else "Good – moderate physics consistency"         if overall_score >= 60
        else "Fair – some disagreement with wind patterns" if overall_score >= 40
        else "Poor – motion inconsistent with ERA5 wind"
    )

    return {
        "overall_score":      overall_score,
        "cloud_region_score": cloud_score,
        "spatial_mean_sim":   float(spatial_map.mean()),
        "spatial_std_sim":    float(spatial_map.std()),
        "interpretation":     interpretation,
    }
