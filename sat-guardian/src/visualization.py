"""
SAT-GUARDIAN: Visualization Module
====================================
Generates publication-quality plots, animations, and diagnostic figures:

1.  Side-by-side satellite frame comparison
2.  Animated GIF of the interpolated sequence
3.  Optical flow RGB overlay
4.  Confidence heatmap
5.  Cloud motion consistency spatial map
6.  Metrics bar charts
7.  Full pipeline summary dashboard
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import imageio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom colour palette
# ---------------------------------------------------------------------------

_CONFIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "conf", [(0, "#d62728"), (0.5, "#ffeb3b"), (1, "#2ca02c")]
)


# ---------------------------------------------------------------------------
# 1. Frame comparison plot
# ---------------------------------------------------------------------------

def plot_frame_comparison(
    frame_t0:   np.ndarray,
    frame_025:  np.ndarray,
    frame_050:  np.ndarray,
    frame_075:  np.ndarray,
    frame_t1:   np.ndarray,
    save_path:  Optional[str] = None,
    dpi:        int = 150,
) -> plt.Figure:
    """
    Side-by-side plot: T0 | T0.25 | T0.50 | T0.75 | T1

    Parameters
    ----------
    frame_* : (H, W) float32 arrays in [0, 1]
    save_path : if provided, saves figure to this path

    Returns
    -------
    matplotlib Figure
    """
    frames = [frame_t0, frame_025, frame_050, frame_075, frame_t1]
    labels = ["T₀\n(Input)", "T₀.₂₅\n(Generated)", "T₀.₅₀\n(Generated)",
              "T₀.₇₅\n(Generated)", "T₁\n(Input)"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor("#0f1117")

    for ax, frame, label in zip(axes, frames, labels):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        # Highlight generated frames
        if "Generated" in label:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#00d4ff")
                spine.set_linewidth(2)

    fig.suptitle("SAT-GUARDIAN: Interpolated Satellite Sequence",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 2. GIF animation
# ---------------------------------------------------------------------------

def create_gif_animation(
    frames:    List[np.ndarray],
    save_path: str,
    fps:       int  = 4,
    labels:    Optional[List[str]] = None,
    loop:      int  = 0,
) -> None:
    """
    Create an animated GIF from a list of (H, W) satellite frames.

    Parameters
    ----------
    frames    : list of float32 arrays
    save_path : output .gif path
    fps       : frames per second
    labels    : optional text labels to overlay on each frame
    loop      : 0 = loop forever
    """
    if labels is None:
        labels = [f"Frame {i}" for i in range(len(frames))]

    gif_frames = []
    for frame, label in zip(frames, labels):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, color="white", fontsize=13, fontweight="bold")
        ax.axis("off")
        plt.tight_layout(pad=0.5)

        # Backend-agnostic figure-to-array conversion
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        from PIL import Image
        img_array = np.array(Image.open(buf).convert("RGB"))
        gif_frames.append(img_array)
        buf.close()
        plt.close(fig)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / fps
    imageio.mimsave(save_path, gif_frames, duration=duration, loop=loop)
    logger.info("GIF saved: %s (%d frames, %d fps)", save_path, len(gif_frames), fps)


# ---------------------------------------------------------------------------
# 3. Optical flow overlay
# ---------------------------------------------------------------------------

def plot_flow_overlay(
    frame:     np.ndarray,
    flow:      np.ndarray,
    save_path: Optional[str] = None,
    step:      int = 16,
    dpi:       int = 150,
) -> plt.Figure:
    """
    Quiver plot of optical flow vectors overlaid on a satellite frame.

    Parameters
    ----------
    frame : (H, W) satellite frame
    flow  : (H, W, 2) flow field
    step  : spatial sampling step for quiver arrows
    """
    H, W = frame.shape
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    u = flow[yy, xx, 0]
    v = flow[yy, xx, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    # Left: frame + arrows
    axes[0].imshow(frame, cmap="gray", vmin=0, vmax=1)
    axes[0].quiver(xx, yy, u, -v, color="#00d4ff", scale=50, alpha=0.7,
                   headwidth=3, headlength=4)
    axes[0].set_title("Motion Field (Quiver)", color="white", fontsize=11)
    axes[0].axis("off")

    # Right: HSV flow visualisation
    from optical_flow import flow_to_rgb
    flow_rgb = flow_to_rgb(flow)
    axes[1].imshow(flow_rgb)
    axes[1].set_title("Flow Magnitude/Direction (HSV)", color="white", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 4. Confidence heatmap
# ---------------------------------------------------------------------------

def plot_confidence_heatmap(
    confidence:   np.ndarray,
    frame:        Optional[np.ndarray] = None,
    save_path:    Optional[str] = None,
    dpi:          int = 150,
) -> plt.Figure:
    """
    Visualise confidence map as a heatmap, optionally alongside the frame.
    """
    if frame is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0f1117")
        axes[0].imshow(frame, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Generated Frame", color="white", fontsize=11)
        axes[0].axis("off")
        ax_conf = axes[1]
    else:
        fig, ax_conf = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0f1117")

    im = ax_conf.imshow(confidence, cmap=_CONFIDENCE_CMAP, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)
    cbar.set_label("Confidence", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax_conf.set_title("Pixel Confidence Map", color="white", fontsize=11)
    ax_conf.axis("off")

    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 5. Spatial consistency map
# ---------------------------------------------------------------------------

def plot_spatial_consistency(
    spatial_map: np.ndarray,
    save_path:   Optional[str] = None,
    dpi:         int = 150,
) -> plt.Figure:
    """Plot the per-pixel cloud motion consistency map."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")
    im = ax.imshow(spatial_map, cmap=_CONFIDENCE_CMAP, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title("Cloud Motion Consistency (Spatial)", color="white", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 6. Metrics bar chart
# ---------------------------------------------------------------------------

def plot_metrics_chart(
    metrics_list: List[Dict],
    save_path:    Optional[str] = None,
    dpi:          int = 150,
) -> plt.Figure:
    """
    Bar chart comparing SSIM, PSNR (normalised), and FSIM across frames.

    Parameters
    ----------
    metrics_list : list of dicts from metrics.evaluate_frame()
    """
    labels  = [m.get("label", f"Frame {i}") for i, m in enumerate(metrics_list)]
    ssims   = [m.get("ssim",  0) for m in metrics_list]
    psnrs   = [m.get("psnr",  0) for m in metrics_list]
    fsims   = [m.get("fsim",  0) for m in metrics_list]
    mses    = [m.get("mse",   0) for m in metrics_list]

    # Normalise PSNR to [0, 1] for joint display (cap at 60 dB)
    psnrs_norm = [min(p / 60.0, 1.0) for p in psnrs]

    x   = np.arange(len(labels))
    w   = 0.22
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Quality metrics
    bars = [
        ax1.bar(x - w, ssims,      w, label="SSIM",       color="#00d4ff", alpha=0.85),
        ax1.bar(x,     psnrs_norm, w, label="PSNR (norm)", color="#ff6b6b", alpha=0.85),
        ax1.bar(x + w, fsims,      w, label="FSIM",        color="#95e06c", alpha=0.85),
    ]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="white", fontsize=9)
    ax1.set_ylabel("Score", color="white")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Quality Metrics per Frame", color="white", fontweight="bold")
    ax1.legend(facecolor="#1a1d2e", labelcolor="white", edgecolor="#444")
    ax1.yaxis.label.set_color("white")

    # MSE
    ax2.bar(labels, mses, color="#f7b731", alpha=0.85)
    ax2.set_ylabel("MSE (lower is better)", color="white")
    ax2.set_title("Mean Squared Error", color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.yaxis.label.set_color("white")

    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 7. Full dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(
    frame_t0:    np.ndarray,
    frame_025:   np.ndarray,
    frame_050:   np.ndarray,
    frame_075:   np.ndarray,
    frame_t1:    np.ndarray,
    confidence:  np.ndarray,
    flow:        np.ndarray,
    cmcs:        float,
    metrics_list: Optional[List[Dict]] = None,
    save_path:   Optional[str] = None,
    dpi:         int = 150,
) -> plt.Figure:
    """
    Full-pipeline summary dashboard with all key outputs in one figure.
    """
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)

    frames_row = [frame_t0, frame_025, frame_050, frame_075, frame_t1]
    labels_row = ["T₀ (Input)", "T₀.₂₅", "T₀.₅₀", "T₀.₇₅", "T₁ (Input)"]

    # Row 1: frame sequence
    for col, (fr, lb) in enumerate(zip(frames_row, labels_row)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(fr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(lb, color="white", fontsize=9, fontweight="bold")
        ax.axis("off")
        if "₀.₂₅" in lb or "₀.₅₀" in lb or "₀.₇₅" in lb:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#00d4ff")
                spine.set_linewidth(1.5)

    # Row 2a: confidence map
    ax_conf = fig.add_subplot(gs[1, :2])
    im = ax_conf.imshow(confidence, cmap=_CONFIDENCE_CMAP, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_conf, fraction=0.04, pad=0.04).set_label("Confidence", color="white")
    ax_conf.set_title("Pixel Confidence (T₀.₅₀)", color="white", fontsize=9)
    ax_conf.axis("off")

    # Row 2b: flow HSV
    from optical_flow import flow_to_rgb
    ax_flow = fig.add_subplot(gs[1, 2:4])
    ax_flow.imshow(flow_to_rgb(flow))
    ax_flow.set_title("Physics-Constrained Flow Field", color="white", fontsize=9)
    ax_flow.axis("off")

    # Row 2c: CMCS gauge
    ax_gauge = fig.add_subplot(gs[1, 4])
    ax_gauge.set_facecolor("#1a1d2e")
    _draw_gauge(ax_gauge, cmcs)

    # Row 3: metrics if provided
    if metrics_list:
        ax_metrics = fig.add_subplot(gs[2, :])
        ax_metrics.set_facecolor("#1a1d2e")
        _draw_metrics_table(ax_metrics, metrics_list)

    fig.suptitle(
        "SAT-GUARDIAN — Physics-Aware Satellite Frame Interpolation",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )

    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# Gauge and table helpers
# ---------------------------------------------------------------------------

def _draw_gauge(ax: plt.Axes, score: float) -> None:
    """Draw a simple half-pie gauge for the cloud motion score."""
    import matplotlib.patches as mpatches

    score_clamp = max(0.0, min(100.0, score))
    colours = ["#d62728", "#ff7f0e", "#ffeb3b", "#2ca02c"]
    thresholds = [0, 40, 60, 80, 100]

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.1, 1.3)
    ax.axis("off")

    theta = np.linspace(0, np.pi, 200)
    for i in range(len(colours)):
        start = thresholds[i] / 100 * np.pi
        end   = thresholds[i+1] / 100 * np.pi
        th_s  = np.linspace(start, end, 50)
        for t1, t2 in zip(th_s[:-1], th_s[1:]):
            ax.fill_between(
                [0.7*np.cos(t1), 0.9*np.cos(t1), 0.9*np.cos(t2), 0.7*np.cos(t2)],
                [0.7*np.sin(t1), 0.9*np.sin(t1), 0.9*np.sin(t2), 0.7*np.sin(t2)],
                color=colours[i], alpha=0.9,
            )

    # Needle
    angle = (1.0 - score_clamp / 100.0) * np.pi
    ax.annotate("", xy=(0.65*np.cos(angle), 0.65*np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
    ax.text(0, -0.08, f"{score_clamp:.1f}/100", ha="center", va="top",
            color="white", fontsize=12, fontweight="bold")
    ax.set_title("Cloud Motion\nConsistency Score", color="white", fontsize=8)


def _draw_metrics_table(ax: plt.Axes, metrics_list: List[Dict]) -> None:
    """Render metrics as a styled table inside an axes."""
    col_labels = ["Frame", "MSE ↓", "PSNR ↑ (dB)", "SSIM ↑", "FSIM ↑"]
    rows = []
    for m in metrics_list:
        rows.append([
            m.get("label", "?"),
            f'{m.get("mse",  0):.5f}',
            f'{m.get("psnr", 0):.2f}',
            f'{m.get("ssim", 0):.4f}',
            f'{m.get("fsim", 0):.4f}',
        ])

    ax.axis("off")
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        if r == 0:
            cell.set_facecolor("#00d4ff")
            cell.get_text().set_color("black")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#1e2235")
            cell.get_text().set_color("white")
        else:
            cell.set_facecolor("#161929")
            cell.get_text().set_color("white")


# ---------------------------------------------------------------------------
# File I/O helper
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    logger.info("Figure saved: %s", path)


def save_frame(frame: np.ndarray, path: str) -> None:
    """Save a single grayscale frame as a PNG."""
    import cv2
    img = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    logger.info("Frame saved: %s", path)


def plt_close_all() -> None:
    """Close all open matplotlib figures to free memory."""
    plt.close("all")
