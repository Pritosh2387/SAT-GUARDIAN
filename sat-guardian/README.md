# 🛰️ SAT-GUARDIAN
### Physics-Aware Adaptive Satellite Frame Interpolation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Generate sub-30-minute satellite imagery by combining AI frame interpolation with atmospheric physics**

</div>

---

## 🌍 Problem Statement

India's **INSAT-3DS** geostationary satellite captures atmospheric imagery at **30-minute intervals**. This temporal gap is a critical limitation for:

- ⛈️ **Rapid cyclone intensification** monitoring (can change dramatically in < 30 min)
- 🌪️ **Severe convective storm** tracking
- 🔥 **Forest fire** spread prediction
- 🌊 **Flood** early warning systems

**SAT-GUARDIAN** bridges this gap by synthesising intermediate frames at **T+7.5 min, T+15 min, and T+22.5 min** using a physics-aware AI pipeline — effectively creating a **4× temporal super-resolution** of satellite observations.

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       SAT-GUARDIAN Pipeline                      │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────────────────┐  │
│  │  Frame T0 │   │ ERA5     │   │ Frame T1                    │  │
│  │  (Input)  │   │ Wind u,v │   │ (Input)                     │  │
│  └────┬──────┘   └────┬─────┘   └──────┬──────────────────────┘  │
│       │               │                 │                          │
│       └───────────────┼─────────────────┘                         │
│                       │                                           │
│              ┌─────────▼──────────┐                              │
│              │ Physics-Constrained │                              │
│              │   Optical Flow      │                              │
│              │ 0.7×Farneback +    │                              │
│              │ 0.3×ERA5 Wind       │                              │
│              └─────────┬──────────┘                              │
│                        │ Motion Field                             │
│              ┌─────────▼──────────┐                              │
│              │  LightUNet Model   │                              │
│              │  (Encoder-Decoder) │                              │
│              └─────────┬──────────┘                              │
│                        │                                          │
│           ┌────────────┼────────────┐                            │
│           ▼            ▼            ▼                            │
│        T₀.₂₅        T₀.₅₀        T₀.₇₅                         │
│      (T+7.5min)   (T+15min)   (T+22.5min)                       │
│           │            │            │                            │
│           └────────────┼────────────┘                            │
│                        │                                          │
│          ┌─────────────┼──────────────┐                          │
│          ▼             ▼              ▼                           │
│    Confidence     Cloud Motion    Image Quality                   │
│       Map         Score (CMCS)    (SSIM/PSNR)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description | Algorithm |
|-----------|-------------|-----------|
| **Optical Flow** | Dense motion estimation | Farneback + ERA5 blend (70/30) |
| **Frame Generator** | Cascaded interpolation | Bilateral warping + LightUNet |
| **LightUNet** | Neural interpolator | 4-level U-Net (encoder-decoder) |
| **Confidence Map** | Per-pixel reliability | Flow discrepancy + FB-consistency |
| **CMCS** | Physics alignment score | Cosine similarity (flow vs wind) |

---

## 📁 Project Structure

```
sat-guardian/
│
├── 📄 README.md
├── 📋 requirements.txt
│
├── ⚙️  configs/
│   └── config.yaml              ← Master configuration
│
├── 📂 data/
│   ├── raw/                     ← Real INSAT-3DS .nc files
│   ├── processed/               ← ERA5 wind .nc files
│   └── sample/                  ← Auto-generated synthetic data
│
├── 📓 notebooks/
│   ├── exploration.ipynb        ← Data exploration & EDA
│   └── demo.ipynb               ← Interactive end-to-end demo
│
├── 🧠 src/
│   ├── data_loader.py           ← NetCDF + numpy data loading
│   ├── preprocessing.py         ← Normalisation, resizing, masking
│   ├── optical_flow.py          ← Physics-constrained Farneback flow
│   ├── interpolation_model.py   ← LightUNet encoder-decoder model
│   ├── three_frame_generator.py ← Cascaded T0.25/0.50/0.75 generation
│   ├── confidence_map.py        ← Pixel confidence estimation
│   ├── cloud_motion_score.py    ← CMCS (cosine similarity metric)
│   ├── metrics.py               ← SSIM, PSNR, MSE, FSIM
│   ├── visualization.py         ← Dashboard, GIF, heatmaps, charts
│   └── inference.py             ← End-to-end inference engine
│
├── 🤖 models/                   ← Saved model checkpoints
│
├── 📊 outputs/
│   ├── generated_frames/        ← T0.25, T0.50, T0.75 PNG images
│   ├── confidence_maps/         ← Confidence heatmaps
│   └── animations/              ← Interpolation GIF
│
├── 🚀 scripts/
│   ├── demo.py                  ← Quick end-to-end demo
│   ├── train.py                 ← Model training script
│   └── evaluate.py              ← Evaluation vs baseline
│
└── 🧪 tests/
    └── test_all.py              ← Full unit test suite
```

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/sat-guardian.git
cd sat-guardian

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
# Generate synthetic sample data + run full pipeline
python scripts/demo.py
```

This will:
1. ✅ Generate synthetic INSAT-3DS-like frames (T0, T1)
2. ✅ Generate synthetic ERA5 wind fields (u, v)
3. ✅ Compute physics-constrained optical flow
4. ✅ Generate intermediate frames: **T0.25, T0.50, T0.75**
5. ✅ Compute **confidence maps**
6. ✅ Compute **Cloud Motion Consistency Score** (0–100)
7. ✅ Save PNG images, GIF animation, and dashboard plot

### 3. Evaluate Against Baseline

```bash
python scripts/evaluate.py
```

### 4. Train the Model (Optional)

```bash
# Quick training run on synthetic data
python scripts/train.py --epochs 20 --batch-size 4

# Use trained model in demo
python scripts/demo.py --model models/best_model.pth
```

### 5. Run Unit Tests

```bash
pytest tests/ -v
```

---

## 💻 Usage with Real Data

### INSAT-3DS Frames

```python
from src.data_loader import load_insat_frame

# Load from NetCDF file
frame_t0 = load_insat_frame("data/raw/insat3ds_T0.nc",
                             variable="brightness_temperature")
frame_t1 = load_insat_frame("data/raw/insat3ds_T1.nc",
                             variable="brightness_temperature")
```

### ERA5 Wind Data

```python
from src.data_loader import load_era5_wind

# Load ERA5 u/v wind components
wind_u, wind_v = load_era5_wind("data/processed/era5_wind.nc",
                                 u_variable="u", v_variable="v",
                                 target_shape=(256, 256))
```

### Programmatic Inference

```python
from src.inference import SATGuardianInference

engine  = SATGuardianInference("configs/config.yaml")
results = engine.run(frame_t0, frame_t1, wind_u, wind_v)
engine.save_all(results, output_dir="outputs/")

# Access results
frame_025  = results["frame_025"]    # T+7.5 min
frame_050  = results["frame_050"]    # T+15 min
frame_075  = results["frame_075"]    # T+22.5 min
confidence = results["confidence_050"]
cmcs       = results["cloud_motion"]["overall_score"]
```

---

## 🔬 Technical Deep Dive

### Phase 1 — Physics-Constrained Optical Flow

```
final_flow = 0.70 × Farneback_flow + 0.30 × ERA5_wind_flow_px
```

The Farneback algorithm computes dense optical flow between satellite frames.
ERA5 wind vectors (m/s) are converted to pixel displacement units using:

```
pixel_displacement = wind_speed(m/s) × Δt(s) / pixel_size(m)
```

### Phase 2 — Cascaded Frame Generation

```
Step 1: T₀.₅₀ = bilateral_warp(T₀, T₁, flow, α=0.5)
Step 2: T₀.₂₅ = bilateral_warp(T₀, T₀.₅₀, flow×0.5, α=0.5)
Step 3: T₀.₇₅ = bilateral_warp(T₀.₅₀, T₁, flow×0.5, α=0.5)
```

Bilateral warping blends forward and backward warped frames:
```
result = (1-α) × warp(src, α×flow_fwd) + α × warp(dst, (1-α)×flow_bwd)
```

### Phase 3 — LightUNet Model

A compact U-Net with 4 encoder/decoder levels (~2M parameters):
- **Input**: 3 channels (T0, T1, flow magnitude)
- **Output**: 1 channel (intermediate frame)
- **Loss**: L1 + SSIM combined loss
- **Training**: Cosine annealing LR schedule with gradient clipping

### Phase 4 — Confidence Map

```python
discrepancy = ||optical_flow - ERA5_wind_flow||₂
confidence  = exp(-discrepancy / scale_95th_percentile)
```

Optionally penalised by forward-backward flow inconsistency (occlusion detection).

### Phase 5 — Cloud Motion Consistency Score (CMCS)

```python
CMCS = mean(cosine_similarity(flow_vector, wind_vector)) × 100
     → Range: 0 to 100
```

| Score | Interpretation |
|-------|---------------|
| 80–100 | Excellent – highly physics-consistent |
| 60–80  | Good – moderate consistency |
| 40–60  | Fair – some physics disagreement |
| 0–40   | Poor – inconsistent with ERA5 wind |

---

## 📊 Sample Outputs

```
outputs/
├── frame_comparison.png        ← T0 | T0.25 | T0.50 | T0.75 | T1
├── dashboard.png               ← Full pipeline summary dashboard
├── flow_overlay.png            ← Quiver plot + HSV flow map
├── generated_frames/
│   ├── frame_t0.png
│   ├── frame_t025.png
│   ├── frame_t050.png
│   ├── frame_t075.png
│   └── frame_t1.png
├── confidence_maps/
│   ├── confidence_t050.png
│   └── confidence_t050_colourmap.png
└── animations/
    └── interpolation.gif       ← Animated sequence
```

---

## 🗺️ ERA5 Data Download

```bash
pip install cdsapi

# Create ~/.cdsapirc with your CDS credentials, then:
python - <<EOF
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-pressure-levels', {
    'product_type': 'reanalysis',
    'variable': ['u_component_of_wind', 'v_component_of_wind'],
    'pressure_level': ['500'],
    'year': '2024', 'month': '06', 'day': '15',
    'time': '00:00', 'format': 'netcdf'
}, 'era5_wind.nc')
EOF
```

---

## 🚀 Future Work

| Priority | Enhancement |
|----------|-------------|
| 🔥 High | Train on real INSAT-3DS data paired with NWP model output |
| 🔥 High | Implement video-based flow networks (RAFT, FlowNet2) |
| 🔥 High | Multi-channel support (IR, WV, VIS channels) |
| 📈 Medium | Diffusion model-based frame generation |
| 📈 Medium | Uncertainty quantification with MC-Dropout |
| 📈 Medium | Real-time streaming inference pipeline |
| 🔬 Research | Assimilation of microwave sounder data |
| 🔬 Research | Physics-informed neural network (PINN) loss |

---

## 📚 References

1. Farneback, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA.
2. Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
3. Niklaus, S. et al. (2018). *Video Frame Interpolation via Adaptive Separable Convolution*. ICCV.
4. Hersbach, H. et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society.
5. ISRO. *INSAT-3DS Satellite Documentation*. SAC/ISRO.

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the BAH Hackathon 🏆 | Satellite AI Track**

*Bridging temporal gaps in Earth observation with physics-aware AI*

</div>
