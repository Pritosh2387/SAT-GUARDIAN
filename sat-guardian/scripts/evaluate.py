"""
SAT-GUARDIAN Evaluation Script
================================
Evaluates generated frames against pseudo ground truth using
all quality metrics (SSIM, PSNR, MSE, FSIM) and saves a report.

Usage
-----
    cd sat-guardian
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/best_model.pth
    python scripts/evaluate.py --output-dir eval_outputs/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data_loader import load_sample_data, generate_sample_frames
from preprocessing import resize_frame
from inference import SATGuardianInference, load_config
from metrics import (
    evaluate_frame,
    temporal_consistency_score,
    compute_ssim, compute_psnr, compute_mse, compute_fsim,
)
from visualization import plot_metrics_chart, plt_close_all

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground truth generation (linear interpolation baseline)
# ---------------------------------------------------------------------------

def generate_ground_truth_linear(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
) -> dict:
    """
    Generate pseudo ground truth via linear blending for evaluation.
    Note: In a real scenario these would be real observed intermediate frames.
    """
    return {
        "frame_025": 0.75 * frame_t0 + 0.25 * frame_t1,
        "frame_050": 0.50 * frame_t0 + 0.50 * frame_t1,
        "frame_075": 0.25 * frame_t0 + 0.75 * frame_t1,
    }


# ---------------------------------------------------------------------------
# Baseline: linear interpolation
# ---------------------------------------------------------------------------

def baseline_linear(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
) -> dict:
    return {
        "frame_025": 0.75 * frame_t0 + 0.25 * frame_t1,
        "frame_050": 0.50 * frame_t0 + 0.50 * frame_t1,
        "frame_075": 0.25 * frame_t0 + 0.75 * frame_t1,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    cfg = load_config(args.config)
    H, W = cfg["data"]["frame_height"], cfg["data"]["frame_width"]

    # Load sample data
    sample = load_sample_data(cfg["data"]["sample_dir"], H=H, W=W)
    frame_t0 = resize_frame(sample["frame_t0"], (H, W))
    frame_t1 = resize_frame(sample["frame_t1"], (H, W))
    wind_u   = sample["wind_u"]
    wind_v   = sample["wind_v"]

    # Run SAT-GUARDIAN
    engine  = SATGuardianInference(args.config, args.model)
    results = engine.run(frame_t0, frame_t1, wind_u, wind_v)

    # Pseudo ground truth (linear blend)
    gt = generate_ground_truth_linear(frame_t0, frame_t1)

    # Baseline metrics
    bl = baseline_linear(frame_t0, frame_t1)

    frames_pairs = [
        ("T0.25", results["frame_025"], gt["frame_025"], bl["frame_025"]),
        ("T0.50", results["frame_050"], gt["frame_050"], bl["frame_050"]),
        ("T0.75", results["frame_075"], gt["frame_075"], bl["frame_075"]),
    ]

    sat_metrics  = []
    base_metrics = []

    print("\n" + "=" * 70)
    print(f"  {'Frame':<8} {'Metric':<8} {'SAT-GUARDIAN':>14}  {'Baseline (Lin)':>15}")
    print("=" * 70)

    for label, pred, target, bl_pred in frames_pairs:
        sm = evaluate_frame(pred,    target, label=f"SAT {label}")
        bm = evaluate_frame(bl_pred, target, label=f"BL  {label}")
        sat_metrics.append(sm)
        base_metrics.append(bm)

        for metric in ["ssim", "psnr", "mse", "fsim"]:
            print(f"  {label:<8} {metric.upper():<8} {sm[metric]:>14.4f}  {bm[metric]:>15.4f}")
        print()

    # Temporal consistency
    seq_sat = [frame_t0, results["frame_025"], results["frame_050"],
               results["frame_075"], frame_t1]
    seq_bl  = [frame_t0, bl["frame_025"], bl["frame_050"], bl["frame_075"], frame_t1]

    tc_sat = temporal_consistency_score(seq_sat)
    tc_bl  = temporal_consistency_score(seq_bl)

    print(f"  {'Temporal Consistency':<28} {tc_sat:>14.4f}  {tc_bl:>15.4f}")
    print("=" * 70)

    # Cloud motion
    cm = results["cloud_motion"]
    print(f"\n  Cloud Motion Score (SAT-GUARDIAN): {cm['overall_score']:.1f}/100")
    print(f"  {cm['interpretation']}")

    # Save report
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "sat_guardian_metrics": sat_metrics,
        "baseline_metrics":     base_metrics,
        "cloud_motion":         cm,
        "temporal_consistency": {"sat_guardian": tc_sat, "baseline": tc_bl},
    }
    report_path = out / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)

    # Save metrics chart (SAT-GUARDIAN results)
    plot_metrics_chart(sat_metrics, save_path=str(out / "sat_metrics_chart.png"))
    plt_close_all()
    plot_metrics_chart(base_metrics, save_path=str(out / "baseline_metrics_chart.png"))
    plt_close_all()

    print(f"\n[DONE]  Evaluation report saved to: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAT-GUARDIAN Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     default=str(_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--model",      default=None)
    parser.add_argument("--output-dir", default=str(_ROOT / "outputs" / "evaluation"))
    parser.add_argument("--log-level",  default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    evaluate(args)
