"""
SAT-GUARDIAN: Inference Engine
================================
High-level inference API that ties all components together.
Can be used programmatically or via the demo / evaluate scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import yaml

# Ensure src/ is on path when running scripts
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import load_insat_frame, load_era5_wind, load_sample_data
from preprocessing import prepare_frame_pair, compute_cloud_mask
from optical_flow import compute_optical_flow
from three_frame_generator import generate_all_frames
from confidence_map import compute_confidence_map, confidence_stats, save_confidence_map
from cloud_motion_score import compute_cloud_motion_score, score_report, compute_spatial_consistency_map
from metrics import evaluate_frame, temporal_consistency_score
from visualization import (
    plot_frame_comparison,
    create_gif_animation,
    plot_flow_overlay,
    plot_confidence_heatmap,
    plot_metrics_chart,
    plot_dashboard,
    save_frame,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

class SATGuardianInference:
    """
    End-to-end inference pipeline for SAT-GUARDIAN.

    Usage
    -----
    >>> engine = SATGuardianInference("configs/config.yaml")
    >>> results = engine.run(frame_t0, frame_t1, wind_u, wind_v)
    >>> engine.save_all(results, output_dir="outputs")
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        model_path:  Optional[str] = None,
        strategy:    str = "flow",
    ):
        self.config   = load_config(config_path)
        self.strategy = strategy
        self.model    = None

        if model_path and Path(model_path).exists():
            from interpolation_model import load_checkpoint, get_device
            device = get_device(self.config["training"].get("device", "auto"))
            self.model = load_checkpoint(model_path, device)
            self.strategy = "model"
            logger.info("Model loaded from %s | strategy=model", model_path)
        else:
            logger.info("Running in flow-only mode (no model checkpoint)")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        frame_t0: np.ndarray,
        frame_t1: np.ndarray,
        wind_u:   Optional[np.ndarray] = None,
        wind_v:   Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Execute the full inference pipeline.

        Parameters
        ----------
        frame_t0, frame_t1 : (H, W) float32 normalised frames
        wind_u, wind_v     : (H, W) ERA5 wind components in m/s

        Returns
        -------
        dict containing:
            - generated frames: frame_025, frame_050, frame_075
            - flows: flow_fwd, flow_bwd, optical_flow_fwd, wind_flow_fwd
            - confidence_050: pixel confidence map for T0.50
            - cloud_motion_report: CMCS report dict
            - metrics: list of metric dicts (if ground truth is supplied)
        """
        flow_cfg = self.config.get("optical_flow", {})
        frame_shape = self.config["data"]["frame_height"], self.config["data"]["frame_width"]

        # Ensure consistent spatial dimensions
        from preprocessing import resize_frame, resize_wind
        frame_t0 = resize_frame(frame_t0, frame_shape)
        frame_t1 = resize_frame(frame_t1, frame_shape)
        if wind_u is not None:
            wind_u, wind_v = resize_wind(wind_u, wind_v, frame_shape)

        # Generate frames
        logger.info("=== SAT-GUARDIAN Inference Start ===")
        gen = generate_all_frames(
            frame_t0, frame_t1, wind_u, wind_v,
            model          = self.model,
            strategy       = self.strategy,
            optical_weight = flow_cfg.get("optical_weight", 0.70),
            physics_weight = flow_cfg.get("physics_weight", 0.30),
        )

        # Confidence map for T0.50 (most important frame)
        conf_cfg = self.config.get("confidence", {})
        confidence_050 = compute_confidence_map(
            gen["optical_flow_fwd"],
            gen["wind_flow_fwd"],
            flow_bwd = gen["flow_bwd"],
            flow_fwd = gen["flow_fwd"],
            sigma    = conf_cfg.get("sigma", 2.0),
        )

        # Cloud motion consistency report
        cm_cfg = self.config.get("cloud_motion", {})
        cloud_mask = compute_cloud_mask(frame_t0)
        cm_report = score_report(gen["flow_fwd"], gen["wind_flow_fwd"], cloud_mask)

        spatial_map = compute_spatial_consistency_map(
            gen["flow_fwd"], gen["wind_flow_fwd"]
        )

        logger.info(
            "CMCS = %.1f/100 | %s",
            cm_report["overall_score"], cm_report["interpretation"],
        )

        return {
            **gen,
            "frame_t0":          frame_t0,
            "frame_t1":          frame_t1,
            "wind_u":            wind_u,
            "wind_v":            wind_v,
            "confidence_050":    confidence_050,
            "cloud_motion":      cm_report,
            "spatial_map":       spatial_map,
            "cloud_mask":        cloud_mask,
        }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    def save_all(
        self,
        results:       Dict,
        output_dir:    str = "outputs",
        metrics_list:  Optional[List[Dict]] = None,
        gif_fps:       int = 4,
    ) -> None:
        """
        Save all generated artefacts to disk.

        Parameters
        ----------
        results      : dict from run()
        output_dir   : root output directory
        metrics_list : optional list of metric dicts for dashboard
        """
        out = Path(output_dir)
        frames_dir = out / "generated_frames"
        conf_dir   = out / "confidence_maps"
        anim_dir   = out / "animations"

        for d in [frames_dir, conf_dir, anim_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Individual frames
        save_frame(results["frame_t0"],  str(frames_dir / "frame_t0.png"))
        save_frame(results["frame_025"], str(frames_dir / "frame_t025.png"))
        save_frame(results["frame_050"], str(frames_dir / "frame_t050.png"))
        save_frame(results["frame_075"], str(frames_dir / "frame_t075.png"))
        save_frame(results["frame_t1"],  str(frames_dir / "frame_t1.png"))

        # Side-by-side comparison
        plot_frame_comparison(
            results["frame_t0"], results["frame_025"],
            results["frame_050"], results["frame_075"],
            results["frame_t1"],
            save_path=str(out / "frame_comparison.png"),
        )
        plt_close_all()

        # Confidence map
        plot_confidence_heatmap(
            results["confidence_050"],
            frame     = results["frame_050"],
            save_path = str(conf_dir / "confidence_t050.png"),
        )
        plt_close_all()

        save_confidence_map(
            results["confidence_050"],
            path=str(conf_dir / "confidence_t050_colourmap.png"),
        )

        # Flow overlay
        plot_flow_overlay(
            results["frame_t0"],
            results["flow_fwd"],
            save_path=str(out / "flow_overlay.png"),
        )
        plt_close_all()

        # GIF animation
        frame_sequence = [
            results["frame_t0"],
            results["frame_025"],
            results["frame_050"],
            results["frame_075"],
            results["frame_t1"],
        ]
        gif_labels = ["T₀ (Input)", "T₀.₂₅ (Gen)", "T₀.₅₀ (Gen)",
                      "T₀.₇₅ (Gen)", "T₁ (Input)"]
        create_gif_animation(
            frame_sequence,
            save_path = str(anim_dir / "interpolation.gif"),
            fps       = gif_fps,
            labels    = gif_labels,
        )

        # Metrics chart
        if metrics_list:
            plot_metrics_chart(
                metrics_list,
                save_path=str(out / "metrics_chart.png"),
            )
            plt_close_all()

        # Full dashboard
        plot_dashboard(
            results["frame_t0"], results["frame_025"],
            results["frame_050"], results["frame_075"], results["frame_t1"],
            results["confidence_050"],
            results["flow_fwd"],
            results["cloud_motion"]["overall_score"],
            metrics_list = metrics_list,
            save_path    = str(out / "dashboard.png"),
        )
        plt_close_all()

        logger.info("=== All outputs saved to: %s ===", output_dir)


def plt_close_all():
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# Convenience: run from sample data
# ---------------------------------------------------------------------------

def run_on_sample(
    config_path: str = "configs/config.yaml",
    output_dir:  str = "outputs",
    model_path:  Optional[str] = None,
) -> Dict:
    """
    End-to-end demo run using synthetic sample data.

    Returns the full results dict.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config(config_path)
    H   = cfg["data"]["frame_height"]
    W   = cfg["data"]["frame_width"]

    # Load or generate sample data
    sample = load_sample_data(
        sample_dir = cfg["data"]["sample_dir"],
        H=H, W=W,
    )

    engine  = SATGuardianInference(config_path, model_path)
    results = engine.run(
        sample["frame_t0"], sample["frame_t1"],
        sample["wind_u"],   sample["wind_v"],
    )

    # Temporal consistency metric
    seq  = [results["frame_t0"], results["frame_025"],
            results["frame_050"], results["frame_075"], results["frame_t1"]]
    tc   = temporal_consistency_score(seq)
    results["temporal_consistency"] = tc
    logger.info("Temporal Consistency Score: %.4f", tc)

    # Save everything
    engine.save_all(results, output_dir=output_dir)

    return results
