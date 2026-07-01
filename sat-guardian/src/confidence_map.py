"""
SAT-GUARDIAN: Confidence Map Generator
========================================
Generates per-pixel confidence values in [0, 1] that reflect how well
the predicted frame aligns with physics-guided expectations.

Confidence is derived from the discrepancy between:
  - Pure optical flow vectors (data-driven signal)
  - ERA5 wind-guided flow vectors (physics prior)

Low discrepancy → high confidence (physics and imagery agree)
High discrepancy → low confidence (uncertain region)

Additionally supports:
  - Occlusion detection via forward-backward flow consistency
  - Spatial smoothing of confidence maps
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core confidence computation
# ---------------------------------------------------------------------------

def compute_flow_discrepancy(
    optical_flow: np.ndarray,
    wind_flow_px: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel L2 discrepancy between optical flow and wind-guided flow.

    Parameters
    ----------
    optical_flow : np.ndarray  (H, W, 2) – raw Farneback flow
    wind_flow_px : np.ndarray  (H, W, 2) – ERA5 wind in pixel units

    Returns
    -------
    np.ndarray  (H, W) float32 – discrepancy magnitude (un-normalised)
    """
    diff = optical_flow - wind_flow_px
    discrepancy = np.linalg.norm(diff, axis=-1)  # (H, W)
    return discrepancy.astype(np.float32)


def compute_confidence_map(
    optical_flow:  np.ndarray,
    wind_flow_px:  np.ndarray,
    flow_bwd:      Optional[np.ndarray] = None,
    flow_fwd:      Optional[np.ndarray] = None,
    sigma:         float = 2.0,
    occlusion_thr: float = 1.0,
) -> np.ndarray:
    """
    Generate a per-pixel confidence map in [0, 1].

    Confidence = exp(-discrepancy / scale)
    Optionally penalised by forward-backward flow inconsistency (occlusion).

    Parameters
    ----------
    optical_flow  : (H, W, 2) raw optical flow
    wind_flow_px  : (H, W, 2) ERA5 wind-guided pixel flow
    flow_bwd      : (H, W, 2) backward flow (optional, for occlusion check)
    flow_fwd      : (H, W, 2) forward flow (optional, for occlusion check)
    sigma         : float  Gaussian smoothing sigma for spatial coherence
    occlusion_thr : float  forward-backward consistency threshold in pixels

    Returns
    -------
    np.ndarray  (H, W) float32 in [0, 1]
    """
    # --- Flow discrepancy-based confidence ---
    discrepancy = compute_flow_discrepancy(optical_flow, wind_flow_px)

    # Normalise discrepancy by its 95th-percentile for robustness
    scale = np.percentile(discrepancy, 95) + 1e-6
    confidence = np.exp(-discrepancy / scale)

    # --- Occlusion mask (forward-backward consistency) ---
    if flow_fwd is not None and flow_bwd is not None:
        occ_mask = _forward_backward_consistency(flow_fwd, flow_bwd, occlusion_thr)
        # Attenuate confidence in occluded regions
        confidence = confidence * (0.3 + 0.7 * (~occ_mask).astype(np.float32))

    # --- Spatial smoothing for perceptual coherence ---
    if sigma > 0:
        import scipy.ndimage as ndi
        confidence = ndi.gaussian_filter(confidence.astype(np.float64), sigma=sigma).astype(np.float32)

    # Ensure strict [0, 1] range
    confidence = np.clip(confidence, 0.0, 1.0)

    logger.info(
        "Confidence map: mean=%.3f | std=%.3f | min=%.3f | max=%.3f",
        confidence.mean(), confidence.std(), confidence.min(), confidence.max(),
    )
    return confidence


def _forward_backward_consistency(
    flow_fwd: np.ndarray,
    flow_bwd: np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Detect occluded pixels using forward-backward flow consistency check.

    A pixel p is considered occluded if:
        ||flow_fwd(p) + flow_bwd(p + flow_fwd(p))|| > threshold

    Returns
    -------
    np.ndarray  bool mask, True = likely occluded
    """
    H, W = flow_fwd.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    # Forward warp the backward flow
    map_x = np.clip(grid_x + flow_fwd[..., 0], 0, W - 1).astype(np.float32)
    map_y = np.clip(grid_y + flow_fwd[..., 1], 0, H - 1).astype(np.float32)

    bwd_u_at_fwd = cv2.remap(flow_bwd[..., 0], map_x, map_y,
                             cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
    bwd_v_at_fwd = cv2.remap(flow_bwd[..., 1], map_x, map_y,
                             cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)

    fb_diff = np.stack([
        flow_fwd[..., 0] + bwd_u_at_fwd,
        flow_fwd[..., 1] + bwd_v_at_fwd,
    ], axis=-1)

    consistency_err = np.linalg.norm(fb_diff, axis=-1)
    return consistency_err > threshold


# ---------------------------------------------------------------------------
# Aggregated statistics
# ---------------------------------------------------------------------------

def confidence_stats(confidence: np.ndarray) -> dict:
    """Return a dict of descriptive statistics for a confidence map."""
    return {
        "mean":          float(confidence.mean()),
        "std":           float(confidence.std()),
        "min":           float(confidence.min()),
        "max":           float(confidence.max()),
        "pct_high":      float((confidence > 0.8).mean()),   # fraction of high-conf pixels
        "pct_medium":    float(((confidence > 0.5) & (confidence <= 0.8)).mean()),
        "pct_low":       float((confidence <= 0.5).mean()),
    }


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_confidence_map(
    confidence: np.ndarray,
    path: str,
    colormap: int = cv2.COLORMAP_RdYlGn if hasattr(cv2, "COLORMAP_RdYlGn") else cv2.COLORMAP_JET,
) -> None:
    """
    Save a confidence map as a colour-coded PNG image.

    Parameters
    ----------
    confidence : (H, W) float32 in [0, 1]
    path       : output file path (.png recommended)
    colormap   : OpenCV colormap index
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Confidence")
    ax.set_title("Pixel Confidence Map")
    ax.axis("off")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confidence map saved: %s", path)
