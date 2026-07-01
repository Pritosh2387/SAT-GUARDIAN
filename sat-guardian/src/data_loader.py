"""
SAT-GUARDIAN: Data Loader
=========================
Handles loading of INSAT-3DS satellite frames (.nc files or numpy arrays)
and ERA5 wind field data. Falls back to synthetic sample data for demos.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NetCDF / xarray helpers
# ---------------------------------------------------------------------------

def load_insat_frame(
    source: Union[str, Path, np.ndarray],
    variable: str = "brightness_temperature",
    time_idx: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load a single INSAT-3DS satellite frame.

    Parameters
    ----------
    source : str | Path | np.ndarray
        Path to a .nc file **or** a pre-loaded numpy array.
    variable : str
        NetCDF variable name to extract (default: brightness_temperature).
    time_idx : int
        Time index to extract when multiple time steps exist in the file.
    normalize : bool
        If True, normalise the frame to [0, 1].

    Returns
    -------
    np.ndarray
        2-D float32 array of shape (H, W), normalised to [0, 1].
    """
    if isinstance(source, np.ndarray):
        frame = source.astype(np.float32)
    else:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"INSAT frame not found: {source}")

        try:
            import xarray as xr
            ds = xr.open_dataset(source)
            if variable not in ds:
                variable = list(ds.data_vars)[0]
                logger.warning("Variable not found; using '%s'", variable)
            data = ds[variable].values
            ds.close()
        except Exception as exc:
            logger.error("xarray load failed (%s); trying netCDF4 fallback", exc)
            from netCDF4 import Dataset  # noqa: F401
            with Dataset(source, "r") as ds:
                data = ds.variables[variable][:]

        # Handle (T, H, W) or (H, W) shapes
        if data.ndim == 3:
            frame = data[time_idx].astype(np.float32)
        elif data.ndim == 2:
            frame = data.astype(np.float32)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

    if normalize:
        vmin, vmax = frame.min(), frame.max()
        if vmax > vmin:
            frame = (frame - vmin) / (vmax - vmin)
        else:
            frame = np.zeros_like(frame)

    logger.info("Loaded frame: shape=%s, min=%.3f, max=%.3f", frame.shape, frame.min(), frame.max())
    return frame


def load_era5_wind(
    source: Union[str, Path, np.ndarray, None],
    u_variable: str = "u",
    v_variable: str = "v",
    time_idx: int = 0,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ERA5 wind (u, v) components and optionally resize to target_shape.

    Parameters
    ----------
    source : str | Path | np.ndarray | None
        Path to a .nc file containing u/v wind components, OR a pre-loaded
        array of shape (2, H, W) where [0]=u, [1]=v, OR None to generate
        synthetic wind fields for demo purposes.
    u_variable, v_variable : str
        NetCDF variable names for zonal and meridional wind.
    time_idx : int
        Time index when the file holds multiple time steps.
    target_shape : (H, W) | None
        If provided, the wind arrays are interpolated to this spatial size.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (u, v) – each float32 array of shape (H, W) in m/s.
    """
    import cv2

    if source is None:
        logger.info("No ERA5 source provided – generating synthetic wind field")
        H, W = target_shape if target_shape else (256, 256)
        u, v = _synthetic_wind(H, W)
    elif isinstance(source, np.ndarray):
        if source.ndim == 3 and source.shape[0] == 2:
            u, v = source[0].astype(np.float32), source[1].astype(np.float32)
        else:
            raise ValueError("numpy ERA5 source must have shape (2, H, W)")
    else:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"ERA5 file not found: {source}")

        try:
            import xarray as xr
            ds = xr.open_dataset(source)
            u_data = ds[u_variable].values
            v_data = ds[v_variable].values
            ds.close()
        except Exception as exc:
            logger.error("xarray load failed (%s); trying netCDF4 fallback", exc)
            from netCDF4 import Dataset
            with Dataset(source, "r") as ds:
                u_data = ds.variables[u_variable][:]
                v_data = ds.variables[v_variable][:]

        u = (u_data[time_idx] if u_data.ndim == 3 else u_data).astype(np.float32)
        v = (v_data[time_idx] if v_data.ndim == 3 else v_data).astype(np.float32)

    # Spatial resizing
    if target_shape is not None and (u.shape != target_shape):
        H, W = target_shape
        u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)

    logger.info("Loaded ERA5 wind: shape=%s, u=[%.2f,%.2f], v=[%.2f,%.2f]",
                u.shape, u.min(), u.max(), v.min(), v.max())
    return u, v


# ---------------------------------------------------------------------------
# Synthetic data generators (for demos / tests)
# ---------------------------------------------------------------------------

def _synthetic_wind(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a smooth, physically plausible synthetic wind field.
    Uses a combination of sinusoidal patterns to mimic zonal/meridional flow.
    """
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    u = 5.0 * np.sin(2 * np.pi * y_idx / H) + 2.0 * np.cos(4 * np.pi * x_idx / W)
    v = 3.0 * np.cos(2 * np.pi * x_idx / W) + 1.5 * np.sin(3 * np.pi * y_idx / H)
    return u.astype(np.float32), v.astype(np.float32)


def generate_sample_frames(
    H: int = 256,
    W: int = 256,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate a pair of synthetic satellite frames (T0, T1) with realistic
    cloud-like structures and an associated ERA5 wind field for demo use.

    Parameters
    ----------
    H, W : int
        Frame spatial dimensions.
    seed : int
        Random seed for reproducibility.
    save_dir : str | None
        If provided, saves arrays as .npy files under this directory.

    Returns
    -------
    dict with keys: "frame_t0", "frame_t1", "wind_u", "wind_v"
    """
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    def _cloud_field(rng, H, W, n_blobs=8):
        field = np.zeros((H, W), dtype=np.float32)
        for _ in range(n_blobs):
            cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
            sigma = rng.uniform(15, 50)
            amp = rng.uniform(0.3, 1.0)
            y_idx, x_idx = np.mgrid[0:H, 0:W]
            blob = amp * np.exp(-((y_idx - cy)**2 + (x_idx - cx)**2) / (2 * sigma**2))
            field += blob
        field = gaussian_filter(field, sigma=3)
        field = np.clip(field, 0, 1)
        return field

    frame_t0 = _cloud_field(rng, H, W, n_blobs=10)

    # T1 is T0 shifted by wind + small structural evolution
    u, v = _synthetic_wind(H, W)
    shift_y = int(np.mean(v) * 2)
    shift_x = int(np.mean(u) * 2)

    from scipy.ndimage import shift as ndimage_shift
    frame_t1 = ndimage_shift(frame_t0, shift=(shift_y, shift_x), mode="wrap")
    # Add slight evolution noise
    frame_t1 = np.clip(frame_t1 + rng.normal(0, 0.02, (H, W)), 0, 1).astype(np.float32)

    result = {
        "frame_t0": frame_t0,
        "frame_t1": frame_t1,
        "wind_u": u,
        "wind_v": v,
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in result.items():
            np.save(save_dir / f"{name}.npy", arr)
        logger.info("Sample data saved to %s", save_dir)

    return result


def load_sample_data(
    sample_dir: str = "data/sample",
    H: int = 256,
    W: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Load sample data from disk if it exists, otherwise generate and save it.
    """
    sample_dir = Path(sample_dir)
    required = ["frame_t0.npy", "frame_t1.npy", "wind_u.npy", "wind_v.npy"]

    if all((sample_dir / f).exists() for f in required):
        logger.info("Loading existing sample data from %s", sample_dir)
        return {
            "frame_t0": np.load(sample_dir / "frame_t0.npy"),
            "frame_t1": np.load(sample_dir / "frame_t1.npy"),
            "wind_u":   np.load(sample_dir / "wind_u.npy"),
            "wind_v":   np.load(sample_dir / "wind_v.npy"),
        }

    logger.info("Sample data not found – generating synthetic data")
    return generate_sample_frames(H=H, W=W, save_dir=str(sample_dir))
