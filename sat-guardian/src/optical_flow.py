"""
SAT-GUARDIAN: Physics-Constrained Optical Flow
===============================================
Combines Farneback optical flow with ERA5 wind vector guidance to produce
a physics-aware motion field for satellite frame interpolation.

Blend formula
-------------
    final_flow = α * optical_flow + (1 - α) * wind_vectors_px

where α = optical_weight (default 0.70) and wind vectors are first
converted from m/s to pixel-displacement units.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Optional, Dict

from preprocessing import normalize_wind_to_pixels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core flow computation
# ---------------------------------------------------------------------------

def compute_farneback_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> np.ndarray:
    """
    Compute dense optical flow between two grayscale frames using Farneback.

    Parameters
    ----------
    frame1, frame2 : np.ndarray
        Grayscale float32 arrays in [0, 1], shape (H, W).

    Returns
    -------
    np.ndarray  shape (H, W, 2) – flow[y, x] = (dx, dy) in pixels
    """
    # OpenCV expects uint8
    f1 = (np.clip(frame1, 0, 1) * 255).astype(np.uint8)
    f2 = (np.clip(frame2, 0, 1) * 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        f1, f2,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )
    logger.debug("Farneback flow: mean_mag=%.3f px", np.linalg.norm(flow, axis=2).mean())
    return flow.astype(np.float32)


def compute_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    wind_u: Optional[np.ndarray] = None,
    wind_v: Optional[np.ndarray] = None,
    optical_weight: float = 0.70,
    physics_weight: float = 0.30,
    time_delta_s: float = 1800.0,
    pixel_size_m: float = 4000.0,
    flow_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute physics-constrained optical flow.

    Combines Farneback dense optical flow with ERA5 wind-guided displacement
    vectors according to the blending formula:

        final_flow = optical_weight * optical_flow
                   + physics_weight * wind_flow_px

    Parameters
    ----------
    frame1, frame2 : np.ndarray  shape (H, W) in [0, 1]
    wind_u, wind_v : np.ndarray | None  ERA5 wind components in m/s (H, W)
    optical_weight : float   weight for pure optical flow (default 0.70)
    physics_weight : float   weight for ERA5-guided flow (default 0.30)
    time_delta_s : float     seconds between frames
    pixel_size_m : float     metres per pixel
    flow_params : dict | None  override Farneback parameters

    Returns
    -------
    (final_flow, optical_flow, wind_flow_px)
        final_flow   : np.ndarray (H, W, 2) – blended motion field
        optical_flow : np.ndarray (H, W, 2) – raw Farneback flow
        wind_flow_px : np.ndarray (H, W, 2) – ERA5-derived pixel displacement
    """
    H, W = frame1.shape

    # --- Farneback optical flow ---
    params = {
        "pyr_scale": 0.5, "levels": 3, "winsize": 15,
        "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0,
    }
    if flow_params:
        params.update(flow_params)

    optical_flow = compute_farneback_flow(frame1, frame2, **params)

    # --- ERA5 wind → pixel displacement ---
    if wind_u is not None and wind_v is not None:
        u_px, v_px = normalize_wind_to_pixels(
            wind_u, wind_v, H, W, time_delta_s, pixel_size_m
        )
    else:
        logger.warning("No ERA5 wind provided; using zero wind flow")
        u_px = np.zeros((H, W), dtype=np.float32)
        v_px = np.zeros((H, W), dtype=np.float32)

    wind_flow_px = np.stack([u_px, v_px], axis=-1)  # (H, W, 2)

    # --- Blend ---
    final_flow = optical_weight * optical_flow + physics_weight * wind_flow_px

    logger.info(
        "Flow computed: optical_mag=%.3f | wind_mag=%.3f | final_mag=%.3f",
        np.linalg.norm(optical_flow, axis=2).mean(),
        np.linalg.norm(wind_flow_px, axis=2).mean(),
        np.linalg.norm(final_flow,   axis=2).mean(),
    )
    return final_flow, optical_flow, wind_flow_px


# ---------------------------------------------------------------------------
# Flow utilities
# ---------------------------------------------------------------------------

def flow_to_magnitude_angle(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose an (H, W, 2) flow array into magnitude and angle maps.

    Returns
    -------
    (magnitude, angle)  each shape (H, W)
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag.astype(np.float32), ang.astype(np.float32)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W, 2) flow array to an HSV-coloured RGB visualisation.

    Returns
    -------
    np.ndarray  uint8 RGB image of shape (H, W, 3)
    """
    mag, ang = flow_to_magnitude_angle(flow)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp a frame forward by the given flow field using bilinear remapping.

    Parameters
    ----------
    frame : np.ndarray  shape (H, W)
    flow  : np.ndarray  shape (H, W, 2) – (dx, dy) in pixels

    Returns
    -------
    np.ndarray  warped frame, shape (H, W), float32 in [0, 1]
    """
    H, W = frame.shape
    # Build remap maps
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(
        frame.astype(np.float32),
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.clip(warped, 0, 1).astype(np.float32)


def smooth_flow(flow: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Apply Gaussian smoothing to each channel of the flow field."""
    import scipy.ndimage as ndi
    fx = ndi.gaussian_filter(flow[..., 0], sigma=sigma)
    fy = ndi.gaussian_filter(flow[..., 1], sigma=sigma)
    return np.stack([fx, fy], axis=-1).astype(np.float32)
