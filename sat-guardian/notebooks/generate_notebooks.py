"""
SAT-GUARDIAN Demo Notebook Generator
"""
import json
from pathlib import Path

DEMO_NOTEBOOK = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "cells": [
  {
   "cell_type": "markdown",
   "id": "m1",
   "metadata": {},
   "source": [
    "# 🛰️ SAT-GUARDIAN: Interactive Demo\n",
    "**Physics-Aware Adaptive Satellite Frame Interpolation System**\n",
    "\n",
    "This notebook demonstrates the full SAT-GUARDIAN pipeline:\n",
    "1. Load / generate sample satellite frames and ERA5 wind\n",
    "2. Compute physics-constrained optical flow\n",
    "3. Generate T0.25, T0.50, T0.75 via cascaded interpolation\n",
    "4. Compute confidence maps and Cloud Motion Consistency Score\n",
    "5. Evaluate quality metrics\n",
    "6. Create visualisations and GIF animation\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys, os\n",
    "from pathlib import Path\n",
    "\n",
    "# Add src to path\n",
    "ROOT = Path('..').resolve()\n",
    "sys.path.insert(0, str(ROOT / 'src'))\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "matplotlib.rcParams['figure.dpi'] = 120\n",
    "matplotlib.rcParams['figure.facecolor'] = '#0f1117'\n",
    "matplotlib.rcParams['text.color'] = 'white'\n",
    "%matplotlib inline\n",
    "\n",
    "print('✅ Imports successful')\n",
    "print(f'Root: {ROOT}')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "m2",
   "metadata": {},
   "source": ["## Step 1: Load Sample Data"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from data_loader import load_sample_data\n",
    "\n",
    "sample = load_sample_data(sample_dir=str(ROOT / 'data' / 'sample'), H=256, W=256)\n",
    "\n",
    "frame_t0 = sample['frame_t0']\n",
    "frame_t1 = sample['frame_t1']\n",
    "wind_u   = sample['wind_u']\n",
    "wind_v   = sample['wind_v']\n",
    "\n",
    "print(f'T0:  shape={frame_t0.shape}, range=[{frame_t0.min():.3f}, {frame_t0.max():.3f}]')\n",
    "print(f'T1:  shape={frame_t1.shape}, range=[{frame_t1.min():.3f}, {frame_t1.max():.3f}]')\n",
    "print(f'U:   shape={wind_u.shape},  range=[{wind_u.min():.2f}, {wind_u.max():.2f}] m/s')\n",
    "print(f'V:   shape={wind_v.shape},  range=[{wind_v.min():.2f}, {wind_v.max():.2f}] m/s')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\n",
    "fig.patch.set_facecolor('#0f1117')\n",
    "for ax in axes:\n",
    "    ax.set_facecolor('#0f1117')\n",
    "\n",
    "axes[0].imshow(frame_t0, cmap='gray', vmin=0, vmax=1)\n",
    "axes[0].set_title('T₀ (Input)', color='white', fontweight='bold')\n",
    "axes[0].axis('off')\n",
    "\n",
    "axes[1].imshow(frame_t1, cmap='gray', vmin=0, vmax=1)\n",
    "axes[1].set_title('T₁ (Input)', color='white', fontweight='bold')\n",
    "axes[1].axis('off')\n",
    "\n",
    "im2 = axes[2].imshow(wind_u, cmap='RdBu_r')\n",
    "axes[2].set_title('ERA5 Wind U (m/s)', color='white', fontweight='bold')\n",
    "axes[2].axis('off')\n",
    "plt.colorbar(im2, ax=axes[2])\n",
    "\n",
    "im3 = axes[3].imshow(wind_v, cmap='RdBu_r')\n",
    "axes[3].set_title('ERA5 Wind V (m/s)', color='white', fontweight='bold')\n",
    "axes[3].axis('off')\n",
    "plt.colorbar(im3, ax=axes[3])\n",
    "\n",
    "plt.suptitle('Input Data Overview', color='white', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "m3",
   "metadata": {},
   "source": ["## Step 2: Run SAT-GUARDIAN Pipeline"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import logging\n",
    "logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')\n",
    "\n",
    "from inference import SATGuardianInference\n",
    "\n",
    "engine  = SATGuardianInference(str(ROOT / 'configs' / 'config.yaml'))\n",
    "results = engine.run(frame_t0, frame_t1, wind_u, wind_v)\n",
    "\n",
    "print('\\n✅ Generated frames:')\n",
    "for key in ['frame_025', 'frame_050', 'frame_075']:\n",
    "    f = results[key]\n",
    "    print(f'  {key}: shape={f.shape}, range=[{f.min():.3f}, {f.max():.3f}]')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "m4",
   "metadata": {},
   "source": ["## Step 3: Visualise Generated Frames"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from visualization import plot_frame_comparison\n",
    "\n",
    "fig = plot_frame_comparison(\n",
    "    results['frame_t0'], results['frame_025'],\n",
    "    results['frame_050'], results['frame_075'],\n",
    "    results['frame_t1'],\n",
    ")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "m5",
   "metadata": {},
   "source": ["## Step 4: Optical Flow Analysis"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from visualization import plot_flow_overlay\n",
    "\n",
    "fig = plot_flow_overlay(results['frame_t0'], results['flow_fwd'])\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "m6",
   "metadata": {},
   "source": ["## Step 5: Confidence Map"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from confidence_map import confidence_stats\n",
    "from visualization import plot_confidence_heatmap\n",
    "\n",
    "stats = confidence_stats(results['confidence_050'])\n",
    "print('Confidence Stats:')\n",
    "for k, v in stats.items():\n",
    "    print(f'  {k:15s}: {v:.4f}')\n",
    "\n",
    "fig = plot_confidence_heatmap(results['confidence_050'], frame=results['frame_050'])\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "m7",
   "metadata": {},
   "source": ["## Step 6: Cloud Motion Score & Metrics"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from metrics import evaluate_frame, temporal_consistency_score\n",
    "\n",
    "cm = results['cloud_motion']\n",
    "print(f'Cloud Motion Score : {cm[\"overall_score\"]:.1f}/100')\n",
    "print(f'Interpretation     : {cm[\"interpretation\"]}')\n",
    "\n",
    "# Quality metrics vs linear interpolation GT\n",
    "gt_050 = 0.5 * frame_t0 + 0.5 * frame_t1\n",
    "m = evaluate_frame(results['frame_050'], gt_050, 'T0.50 vs Linear GT')\n",
    "print(f\"\\nSSIM: {m['ssim']:.4f} | PSNR: {m['psnr']:.2f} dB | MSE: {m['mse']:.6f} | FSIM: {m['fsim']:.4f}\")\n",
    "\n",
    "seq = [results['frame_t0'], results['frame_025'], results['frame_050'],\n",
    "       results['frame_075'], results['frame_t1']]\n",
    "tc = temporal_consistency_score(seq)\n",
    "print(f'Temporal Consistency: {tc:.4f}')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "m8",
   "metadata": {},
   "source": ["## Step 7: Save All Outputs & Dashboard"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from metrics import evaluate_frame\n",
    "\n",
    "metrics_list = [\n",
    "    evaluate_frame(results['frame_025'], 0.75*frame_t0+0.25*frame_t1, 'T0.25'),\n",
    "    evaluate_frame(results['frame_050'], 0.50*frame_t0+0.50*frame_t1, 'T0.50'),\n",
    "    evaluate_frame(results['frame_075'], 0.25*frame_t0+0.75*frame_t1, 'T0.75'),\n",
    "]\n",
    "\n",
    "engine.save_all(results, output_dir=str(ROOT / 'outputs'), metrics_list=metrics_list)\n",
    "print('\\n✅ All outputs saved to outputs/')\n",
    "print('Check: frame_comparison.png, dashboard.png, animations/interpolation.gif')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "m9",
   "metadata": {},
   "source": [
    "## Step 8: Display Dashboard\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c10",
   "metadata": {},
   "outputs": [],
   "source": [
    "from visualization import plot_dashboard\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "fig = plot_dashboard(\n",
    "    results['frame_t0'], results['frame_025'], results['frame_050'],\n",
    "    results['frame_075'], results['frame_t1'],\n",
    "    results['confidence_050'],\n",
    "    results['flow_fwd'],\n",
    "    results['cloud_motion']['overall_score'],\n",
    "    metrics_list=metrics_list,\n",
    ")\n",
    "plt.show()\n"
   ]
  }
 ]
}

EXPLORATION_NOTEBOOK = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a1",
   "metadata": {},
   "source": [
    "# SAT-GUARDIAN: Data Exploration\n",
    "Explore synthetic satellite data, optical flow, and visualisation utilities.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "ROOT = Path('..').resolve()\n",
    "sys.path.insert(0, str(ROOT / 'src'))\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "matplotlib.rcParams['figure.dpi'] = 120\n",
    "%matplotlib inline\n",
    "print('Imports OK')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2",
   "metadata": {},
   "source": ["## 1. Generate Sample Data"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from data_loader import generate_sample_frames\n",
    "\n",
    "data = generate_sample_frames(H=256, W=256, seed=42)\n",
    "frame_t0, frame_t1 = data['frame_t0'], data['frame_t1']\n",
    "wind_u, wind_v     = data['wind_u'],   data['wind_v']\n",
    "\n",
    "print(f'T0  shape: {frame_t0.shape}, range: [{frame_t0.min():.3f}, {frame_t0.max():.3f}]')\n",
    "print(f'T1  shape: {frame_t1.shape}, range: [{frame_t1.min():.3f}, {frame_t1.max():.3f}]')\n",
    "print(f'Wind u  range: [{wind_u.min():.2f}, {wind_u.max():.2f}] m/s')\n",
    "print(f'Wind v  range: [{wind_v.min():.2f}, {wind_v.max():.2f}] m/s')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\n",
    "axes[0].imshow(frame_t0, cmap='gray'); axes[0].set_title('T₀ (Input)'); axes[0].axis('off')\n",
    "axes[1].imshow(frame_t1, cmap='gray'); axes[1].set_title('T₁ (Input)'); axes[1].axis('off')\n",
    "axes[2].imshow(wind_u, cmap='RdBu_r'); axes[2].set_title('ERA5 Wind U (m/s)'); axes[2].axis('off')\n",
    "axes[3].imshow(wind_v, cmap='RdBu_r'); axes[3].set_title('ERA5 Wind V (m/s)'); axes[3].axis('off')\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3",
   "metadata": {},
   "source": ["## 2. Physics-Constrained Optical Flow"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from optical_flow import compute_optical_flow, flow_to_rgb, flow_to_magnitude_angle\n",
    "\n",
    "flow_fwd, opt_flow, wind_flow = compute_optical_flow(\n",
    "    frame_t0, frame_t1, wind_u, wind_v,\n",
    "    optical_weight=0.70, physics_weight=0.30\n",
    ")\n",
    "\n",
    "mag, ang = flow_to_magnitude_angle(flow_fwd)\n",
    "print(f'Flow magnitude: mean={mag.mean():.3f}, max={mag.max():.3f} pixels')\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "axes[0].imshow(flow_to_rgb(opt_flow));  axes[0].set_title('Raw Optical Flow');       axes[0].axis('off')\n",
    "axes[1].imshow(flow_to_rgb(wind_flow)); axes[1].set_title('ERA5 Wind Flow');          axes[1].axis('off')\n",
    "axes[2].imshow(flow_to_rgb(flow_fwd));  axes[2].set_title('Physics-Constrained Flow');axes[2].axis('off')\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4",
   "metadata": {},
   "source": ["## 3. Frame Generation (Cascade)"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from three_frame_generator import generate_all_frames\n",
    "\n",
    "gen = generate_all_frames(frame_t0, frame_t1, wind_u, wind_v, strategy='flow')\n",
    "\n",
    "fig, axes = plt.subplots(1, 5, figsize=(20, 4))\n",
    "data_to_show = [\n",
    "    (frame_t0,        'T₀ (Input)'),\n",
    "    (gen['frame_025'], 'T₀.₂₅ (Gen)'),\n",
    "    (gen['frame_050'], 'T₀.₅₀ (Gen)'),\n",
    "    (gen['frame_075'], 'T₀.₇₅ (Gen)'),\n",
    "    (frame_t1,         'T₁ (Input)'),\n",
    "]\n",
    "for ax, (frm, lbl) in zip(axes, data_to_show):\n",
    "    ax.imshow(frm, cmap='gray', vmin=0, vmax=1)\n",
    "    ax.set_title(lbl, fontsize=11, fontweight='bold')\n",
    "    ax.axis('off')\n",
    "plt.suptitle('SAT-GUARDIAN: Generated Intermediate Frames', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a5",
   "metadata": {},
   "source": ["## 4. Confidence Map"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from confidence_map import compute_confidence_map, confidence_stats\n",
    "\n",
    "conf  = compute_confidence_map(gen['optical_flow_fwd'], gen['wind_flow_fwd'],\n",
    "                               flow_bwd=gen['flow_bwd'], flow_fwd=gen['flow_fwd'])\n",
    "stats = confidence_stats(conf)\n",
    "print('Confidence stats:', {k: round(v,4) for k,v in stats.items()})\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "axes[0].imshow(gen['frame_050'], cmap='gray'); axes[0].set_title('T₀.₅₀ Frame'); axes[0].axis('off')\n",
    "im = axes[1].imshow(conf, cmap='RdYlGn', vmin=0, vmax=1)\n",
    "plt.colorbar(im, ax=axes[1], label='Confidence')\n",
    "axes[1].set_title('Pixel Confidence Map'); axes[1].axis('off')\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6",
   "metadata": {},
   "source": ["## 5. Cloud Motion Score & Metrics"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from cloud_motion_score import score_report\n",
    "from metrics import evaluate_frame\n",
    "\n",
    "cm_report = score_report(gen['flow_fwd'], gen['wind_flow_fwd'])\n",
    "print('Cloud Motion Score:', cm_report['overall_score'], '/100')\n",
    "print('Interpretation    :', cm_report['interpretation'])\n",
    "\n",
    "gt_050 = 0.5 * frame_t0 + 0.5 * frame_t1\n",
    "m = evaluate_frame(gen['frame_050'], gt_050, 'T0.50')\n",
    "print(f\"\\nMetrics: SSIM={m['ssim']:.4f} | PSNR={m['psnr']:.2f}dB | MSE={m['mse']:.5f} | FSIM={m['fsim']:.4f}\")\n"
   ]
  }
 ]
}


if __name__ == "__main__":
    nb_dir = Path(__file__).parent
    
    with open(nb_dir / "exploration.ipynb", "w") as f:
        json.dump(EXPLORATION_NOTEBOOK, f, indent=1)
    print(f"Written: {nb_dir / 'exploration.ipynb'}")

    with open(nb_dir / "demo.ipynb", "w") as f:
        json.dump(DEMO_NOTEBOOK, f, indent=1)
    print(f"Written: {nb_dir / 'demo.ipynb'}")
