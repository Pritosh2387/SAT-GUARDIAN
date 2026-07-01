"""
SAT-GUARDIAN: Image Quality Metrics
=====================================
Implements SSIM, PSNR, MSE, and an FSIM approximation for evaluating
the quality of interpolated satellite frames against ground truth.
"""

import logging
from typing import Dict, Optional, List, Tuple

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error between two frames (lower is better)."""
    return float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))


def compute_psnr(
    pred:   np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio in dB (higher is better).

    Parameters
    ----------
    data_range : float  maximum possible value (1.0 for [0,1] normalised frames)
    """
    mse = compute_mse(pred, target)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(
    pred:        np.ndarray,
    target:      np.ndarray,
    data_range:  float = 1.0,
    win_size:    int   = 11,
) -> float:
    """
    Structural Similarity Index (SSIM), value in [-1, 1] (1 = identical).
    Uses scikit-image implementation for robustness.
    """
    win_size = min(win_size, min(pred.shape[:2]) - 1)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    try:
        score = skimage_ssim(
            pred.astype(np.float64),
            target.astype(np.float64),
            data_range=data_range,
            win_size=win_size,
        )
    except Exception as e:
        logger.warning("SSIM computation failed (%s); returning 0", e)
        score = 0.0
    return float(score)


def compute_fsim(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Feature Similarity Index (FSIM) approximation.

    A simplified version based on gradient magnitude similarity and
    phase congruency approximation (Sobel-based).

    Reference:
        Zhang et al. "FSIM: A Feature Similarity Index for Image Quality
        Assessment", IEEE TIP 2011. (This is an approximation, not the
        full phase congruency implementation.)

    Returns value in [0, 1] (higher is better).
    """
    import cv2

    pred_u8   = (np.clip(pred,   0, 1) * 255).astype(np.uint8)
    target_u8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)

    # Gradient magnitudes (phase-proxy)
    def _grad_mag(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    gm_pred   = _grad_mag(pred_u8).astype(np.float64)
    gm_target = _grad_mag(target_u8).astype(np.float64)

    T1 = 0.85  # gradient similarity constant
    T2 = 160.0 # luminance similarity constant (pixel scale 0-255)

    # Gradient similarity
    gs = (2.0 * gm_pred * gm_target + T1) / (gm_pred**2 + gm_target**2 + T1)

    # Luminance similarity
    lum_pred   = pred_u8.astype(np.float64)
    lum_target = target_u8.astype(np.float64)
    ls = (2.0 * lum_pred * lum_target + T2) / (lum_pred**2 + lum_target**2 + T2)

    # Phase congruency proxy = gradient magnitude (normalised)
    pc = (gm_pred + gm_target) / (np.max(gm_pred + gm_target) + 1e-8)

    fsim = np.sum(gs * ls * pc) / (np.sum(pc) + 1e-8)
    return float(np.clip(fsim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------

def evaluate_frame(
    pred:       np.ndarray,
    target:     np.ndarray,
    label:      str = "frame",
    data_range: float = 1.0,
) -> Dict[str, float]:
    """
    Compute all metrics for a single predicted frame against ground truth.

    Returns
    -------
    dict with keys: mse, psnr, ssim, fsim
    """
    result = {
        "label": label,
        "mse":   compute_mse(pred, target),
        "psnr":  compute_psnr(pred, target, data_range),
        "ssim":  compute_ssim(pred, target, data_range),
        "fsim":  compute_fsim(pred, target),
    }
    logger.info(
        "[%s] MSE=%.5f | PSNR=%.2fdB | SSIM=%.4f | FSIM=%.4f",
        label, result["mse"], result["psnr"], result["ssim"], result["fsim"],
    )
    return result


def evaluate_all(
    frames: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    data_range: float = 1.0,
) -> List[Dict]:
    """
    Batch evaluation across multiple frames.

    Parameters
    ----------
    frames  : dict of {label: predicted_array}
    targets : dict of {label: ground_truth_array}

    Returns
    -------
    list of metric dicts
    """
    results = []
    for label in frames:
        if label not in targets:
            continue
        result = evaluate_frame(frames[label], targets[label], label, data_range)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

def temporal_consistency_score(frames: List[np.ndarray]) -> float:
    """
    Measure temporal smoothness across a sequence of frames.
    Computed as 1 - mean(MSE between consecutive frames).
    Higher is smoother (more temporally consistent).

    Parameters
    ----------
    frames : list of (H, W) float32 arrays in chronological order

    Returns
    -------
    float in [0, 1]
    """
    if len(frames) < 2:
        return 1.0
    inter_frame_mses = [compute_mse(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    score = 1.0 - float(np.mean(inter_frame_mses))
    return float(np.clip(score, 0.0, 1.0))
