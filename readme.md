<div align="center">

# ☁️ SAT-GUARDIAN

### Physics-Aware Adaptive Satellite Frame Interpolation System

**Transforming INSAT-3DS from a 30-minute observer into a 7.5-minute weather intelligence system using atmospheric physics and AI.**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch)
![INSAT-3DS](https://img.shields.io/badge/INSAT--3DS-ISRO-orange?style=for-the-badge)
![ERA5](https://img.shields.io/badge/ERA5-ECMWF-blue?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-31%2F31%20Passed-brightgreen?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-MVP-success?style=for-the-badge)

**Bharatiya Antariksh Hackathon 2026 – Problem Statement 12**

</div>

---

# The Problem

Satellite observations from INSAT-3DS are available every **30 minutes**.

However, severe weather events such as:

- Cyclones
- Thunderstorms
- Convective cloud growth
- Extreme rainfall events

can evolve significantly within this interval.

This creates a **temporal blind spot** where critical atmospheric changes are not observed.

Traditional video interpolation methods treat satellite images like ordinary videos and ignore atmospheric physics, often generating unrealistic cloud motion.

---

# Our Solution

SAT-GUARDIAN combines:

- Physics-constrained optical flow
- Atmospheric wind information from ERA5
- AI-based frame interpolation
- Confidence estimation
- Adaptive temporal resolution

to generate:

```
T0 → T0.25 → T0.50 → T0.75 → T1
```

effectively transforming:

```
30 minutes → 7.5 minutes
```

of temporal resolution.

---

# Key Features

## Physics-Constrained Optical Flow

```python
final_flow =
0.70 × OpticalFlow +
0.30 × ERA5_Wind
```

Clouds cannot move against atmospheric winds.

Our model cannot either.

---

## Cascaded Three-Frame Generation

Instead of generating only one frame:

```
T0 → T0.50 → T1
```

SAT-GUARDIAN generates:

```
T0 → T0.25 → T0.50 → T0.75 → T1
```

providing:

- 4× higher temporal resolution
- Better storm monitoring
- More observations during severe weather.

---

## Pixel-Wise Confidence Maps

Each generated pixel is assigned a reliability score.

Meteorologists know:

- which regions are highly trustworthy
- which regions require caution.

---

## Cloud Motion Consistency Score (CMCS)

Measures how well generated cloud motion agrees with atmospheric wind fields.

```
CMCS =
mean(
cosine_similarity(
predicted_flow,
ERA5_wind
)
) × 100
```

| Score | Interpretation |
|-------|----------------|
| 80-100 | Excellent |
| 60-80 | Good |
| 40-60 | Fair |
| 0-40 | Poor |

---

## NaN Deep Space Masking

The model refuses to fabricate data outside the Earth disk.

Deep space pixels remain:

```
NaN
```

instead of generating unrealistic temperatures.

---

# Architecture

```text
INSAT T0 + INSAT T1 + ERA5 Winds
                    ↓
      Physics-Constrained Optical Flow
                    ↓
       Cascaded Frame Generation
                    ↓
        T0.25  T0.50  T0.75
                    ↓
        Confidence Maps + CMCS
                    ↓
      Adaptive Temporal Resolution
                    ↓
          SAT-GUARDIAN Dashboard
```

---

# Repository Structure

```text
sat-guardian/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
│   ├── exploration.ipynb
│   └── demo.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── optical_flow.py
│   ├── interpolation_model.py
│   ├── three_frame_generator.py
│   ├── confidence_map.py
│   ├── cloud_motion_score.py
│   ├── metrics.py
│   ├── visualization.py
│   └── inference.py
├── scripts/
│   ├── demo.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
├── tests/
├── requirements.txt
└── README.md
```

---

# Demo Results (Synthetic Validation)

| Metric | Value |
|---------|--------|
| SSIM | 0.99+ |
| PSNR | 45+ dB |
| FSIM | 0.98+ |
| CMCS | 97.8 / 100 |
| Runtime | 5.7 sec |

---

# Sample Outputs

The demo generates:

- Interpolated frames
- Confidence maps
- GIF animations
- Dashboard visualizations
- Evaluation reports

Outputs are stored in:

```text
outputs/
├── generated_frames/
├── confidence_maps/
├── animations/
└── evaluation/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/sat-guardian.git
cd sat-guardian
```

---

## Create Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux:

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Demo

```bash
python -X utf8 scripts/demo.py
```

---

## Evaluation

```bash
python -X utf8 scripts/evaluate.py
```

---

## Unit Tests

```bash
pytest tests/ -v
```

---

## Training

```bash
python -X utf8 scripts/train.py --epochs 10
```

---

# Using Real INSAT-3DS Data

```python
from src.inference import SATGuardianInference

engine = SATGuardianInference()

results = engine.run(
    frame_t0,
    frame_t1,
    wind_u,
    wind_v
)
```

Outputs:

```python
results["frame_025"]
results["frame_050"]
results["frame_075"]
results["confidence_map"]
results["cmcs"]
```

---

# Future Work

- Training on large-scale INSAT-3DS datasets
- RAFT optical flow integration
- Diffusion-based frame synthesis
- Multi-channel satellite imagery support
- Physics-informed neural networks
- Real-time deployment pipeline
- Multi-satellite data fusion

---

# Disclaimer

This repository is an **MVP prototype** developed for the **Bharatiya Antariksh Hackathon 2026**.

Current implementation uses synthetic data generators for demonstration and software validation.

Future work includes:

- Training on real INSAT-3DS data
- Validation on historical weather events
- Operational deployment studies

---

# Team

**Project:** SAT-GUARDIAN  
**Hackathon:** Bharatiya Antariksh Hackathon 2026  
**Problem Statement:** Fill the Frames Seamlessly – Enhancing Temporal Resolution of Satellite Imagery using AI/ML based on Optical Flow.

---

<div align="center">

## SAT-GUARDIAN

### Physics-aware. Uncertainty-honest. India-tuned.

**Bridging India's atmospheric blind spots with physics-constrained AI.**

</div>
