"""
SAT-GUARDIAN: Preprocessing
============================
Frame normalisation, resizing, histogram equalisation, cloud masking,
and ERA5 wind field pre-processing utilities.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame normalisation
# ---------------------------------------------------------------------------

def normalize_frame(frame: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalise a 2-D satellite frame.

    Parameters
    ----------
    frame : np.ndarray  shape (H, W)
    method : str
        "minmax"  → [0, 1]
        "zscore"  → zero-mean unit-variance
        "clahe"   → Contrast-Limited Adaptive Histogram Equalisation (8-bit)

    Returns
    -------
    np.ndarray  float32 in [0, 1] (or z-score range for "zscore")
    """
    frame = frame.astype(np.float32)

    if method == "minmax":
        vmin, vmax = frame.min(), frame.max()
        return (frame - vmin) / (vmax - vmin + 1e-8)

    elif method == "zscore":
        return (frame - frame.mean()) / (frame.std() + 1e-8)

    elif method == "clahe":
        img_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_uint8).astype(np.float32) / 255.0
        return enhanced

    else:
        raise ValueError(f"Unknown normalisation method: {method}")


# ---------------------------------------------------------------------------
# Spatial resizing
# ---------------------------------------------------------------------------

def resize_frame(
    frame: np.ndarray,
    target: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize (H, W) frame to target (H', W')."""
    H, W = target
    return cv2.resize(frame, (W, H), interpolation=interpolation).astype(np.float32)


def resize_wind(
    u: np.ndarray,
    v: np.ndarray,
    target: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize wind components to match target spatial dimensions."""
    H, W = target
    u_r = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    v_r = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return u_r, v_r


# ---------------------------------------------------------------------------
# Cloud masking
# ---------------------------------------------------------------------------

def compute_cloud_mask(
    frame: np.ndarray,
    threshold: float = 0.65,
    morphology_kernel: int = 5,
) -> np.ndarray:
    """
    Generate a binary cloud mask from a normalised brightness-temperature frame.
    High brightness-temperature values typically correspond to cold cloud tops.

    Parameters
    ----------
    frame : np.ndarray   shape (H, W), values in [0, 1]
    threshold : float    pixels above this are classified as cloud
    morphology_kernel : int  size of morphological cleaning kernel

    Returns
    -------
    np.ndarray  bool mask, True where cloud is present
    """
    mask = frame > threshold
    kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
    mask_uint = mask.astype(np.uint8)
    mask_uint = cv2.morphologyEx(mask_uint, cv2.MORPH_CLOSE, kernel)
    mask_uint = cv2.morphologyEx(mask_uint, cv2.MORPH_OPEN,  kernel)
    return mask_uint.astype(bool)


# ---------------------------------------------------------------------------
# Wind field normalisation / scaling
# ---------------------------------------------------------------------------

def normalize_wind_to_pixels(
    u: np.ndarray,
    v: np.ndarray,
    frame_height: int,
    frame_width: int,
    time_delta_s: float = 1800.0,
    pixel_size_m: float = 4000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ERA5 wind speed (m/s) to pixel-displacement per frame interval.

    Parameters
    ----------
    u, v : np.ndarray   wind components in m/s
    frame_height, frame_width : int   spatial dimensions
    time_delta_s : float   seconds between T0 and T1 (INSAT-3DS: 30 min = 1800 s)
    pixel_size_m : float  metres per pixel (INSAT-3DS: ~4 km)

    Returns
    -------
    (u_px, v_px) pixel displacements
    """
    scale = time_delta_s / pixel_size_m
    u_px = u * scale
    v_px = v * scale
    # Clamp to ±(frame_size / 2) for stability
    u_px = np.clip(u_px, -frame_width  / 2, frame_width  / 2)
    v_px = np.clip(v_px, -frame_height / 2, frame_height / 2)
    return u_px.astype(np.float32), v_px.astype(np.float32)


# ---------------------------------------------------------------------------
# Frame pair preparation for model input
# ---------------------------------------------------------------------------

def prepare_frame_pair(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    target_shape: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize and normalise a frame pair so they are model-ready.

    Returns
    -------
    Tuple of two float32 arrays in [0, 1], each of shape target_shape.
    """
    t0 = resize_frame(normalize_frame(frame_t0), target_shape)
    t1 = resize_frame(normalize_frame(frame_t1), target_shape)
    return t0, t1


# ---------------------------------------------------------------------------
# Patch extraction (for training data augmentation)
# ---------------------------------------------------------------------------

def extract_patches(
    frame: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
) -> np.ndarray:
    """
    Extract overlapping patches from a 2-D frame for training.

    Returns
    -------
    np.ndarray  shape (N, patch_size, patch_size)
    """
    H, W = frame.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patches.append(frame[y:y + patch_size, x:x + patch_size])
    return np.stack(patches, axis=0).astype(np.float32)


def augment_frame(frame: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Apply random photometric augmentations to a single frame (for training).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random horizontal / vertical flip
    if rng.random() > 0.5:
        frame = np.fliplr(frame)
    if rng.random() > 0.5:
        frame = np.flipud(frame)

    # Brightness jitter
    delta = rng.uniform(-0.05, 0.05)
    frame = np.clip(frame + delta, 0, 1)

    return frame.astype(np.float32)
