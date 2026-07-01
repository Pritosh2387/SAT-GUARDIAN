"""
SAT-GUARDIAN Demo Script
========================
Run the complete SAT-GUARDIAN pipeline on synthetic sample data and
save all outputs (frames, GIF, confidence maps, dashboard, metrics).

Usage
-----
    cd sat-guardian
    python scripts/demo.py
    python scripts/demo.py --output-dir custom_outputs/
    python scripts/demo.py --model models/best_model.pth
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src importable regardless of working directory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from inference import run_on_sample, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAT-GUARDIAN: Physics-Aware Satellite Frame Interpolation Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     default=str(_ROOT / "configs" / "config.yaml"),
                        help="Path to config YAML")
    parser.add_argument("--output-dir", default=str(_ROOT / "outputs"),
                        help="Directory to save all outputs")
    parser.add_argument("--model",      default=None,
                        help="Optional: path to trained model checkpoint (.pth)")
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    logger = logging.getLogger("demo")

    banner = """
+--------------------------------------------------------------+
|         SAT-GUARDIAN -- Satellite Frame Interpolation        |
|     Physics-Aware | AI-Guided | ERA5 Wind-Constrained        |
+--------------------------------------------------------------+
    """
    print(banner)

    t_start = time.time()

    # ----------------------------------------------------------------
    # Run pipeline
    # ----------------------------------------------------------------
    logger.info("Starting demo pipeline ...")
    results = run_on_sample(
        config_path = args.config,
        output_dir  = args.output_dir,
        model_path  = args.model,
    )

    elapsed = time.time() - t_start

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    cm = results["cloud_motion"]
    tc = results.get("temporal_consistency", 0.0)

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print(f"  Output directory   : {args.output_dir}")
    print(f"  Wall time          : {elapsed:.1f}s")
    print(f"  Frames generated   : T0.25, T0.50, T0.75")
    print(f"  Cloud Motion Score : {cm['overall_score']:.1f}/100")
    print(f"  Interpretation     : {cm['interpretation']}")
    print(f"  Temporal Consist.  : {tc:.4f}")
    print("=" * 60)
    print("\nOutput files:")
    out = Path(args.output_dir)
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}")

    print("\n[DONE]  Check the outputs/ directory for all generated artefacts.\n")


if __name__ == "__main__":
    main()
