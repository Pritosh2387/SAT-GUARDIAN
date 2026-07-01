"""
SAT-GUARDIAN Unit Tests
========================
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_generate_sample_frames_shape(self):
        from data_loader import generate_sample_frames
        data = generate_sample_frames(H=64, W=64)
        assert data["frame_t0"].shape == (64, 64)
        assert data["frame_t1"].shape == (64, 64)
        assert data["wind_u"].shape   == (64, 64)
        assert data["wind_v"].shape   == (64, 64)

    def test_generate_sample_frames_range(self):
        from data_loader import generate_sample_frames
        data = generate_sample_frames(H=64, W=64)
        assert 0 <= data["frame_t0"].min() and data["frame_t0"].max() <= 1.0
        assert 0 <= data["frame_t1"].min() and data["frame_t1"].max() <= 1.0

    def test_load_insat_frame_numpy(self):
        from data_loader import load_insat_frame
        arr = np.random.rand(64, 64).astype(np.float32) * 300
        frame = load_insat_frame(arr, normalize=True)
        assert frame.shape == (64, 64)
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0

    def test_era5_synthetic(self):
        from data_loader import load_era5_wind
        u, v = load_era5_wind(None, target_shape=(64, 64))
        assert u.shape == (64, 64)
        assert v.shape == (64, 64)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def setup_method(self):
        self.frame = np.random.rand(64, 64).astype(np.float32)

    def test_normalize_minmax(self):
        from preprocessing import normalize_frame
        out = normalize_frame(self.frame, "minmax")
        assert out.min() >= 0.0
        assert abs(out.max() - 1.0) < 1e-5

    def test_normalize_zscore(self):
        from preprocessing import normalize_frame
        out = normalize_frame(self.frame, "zscore")
        assert abs(out.mean()) < 0.1

    def test_resize_frame(self):
        from preprocessing import resize_frame
        out = resize_frame(self.frame, (32, 32))
        assert out.shape == (32, 32)

    def test_cloud_mask(self):
        from preprocessing import compute_cloud_mask
        mask = compute_cloud_mask(self.frame, threshold=0.5)
        assert mask.dtype == bool
        assert mask.shape == self.frame.shape

    def test_wind_to_pixels(self):
        from preprocessing import normalize_wind_to_pixels
        u = np.ones((64, 64), dtype=np.float32) * 10
        v = np.ones((64, 64), dtype=np.float32) * 5
        u_px, v_px = normalize_wind_to_pixels(u, v, 64, 64)
        assert u_px.shape == (64, 64)
        assert v_px.shape == (64, 64)


# ---------------------------------------------------------------------------
# Optical flow tests
# ---------------------------------------------------------------------------

class TestOpticalFlow:
    def setup_method(self):
        from data_loader import generate_sample_frames
        data = generate_sample_frames(H=64, W=64, seed=0)
        self.f0, self.f1 = data["frame_t0"], data["frame_t1"]
        self.u,  self.v  = data["wind_u"],   data["wind_v"]

    def test_farneback_shape(self):
        from optical_flow import compute_farneback_flow
        flow = compute_farneback_flow(self.f0, self.f1)
        assert flow.shape == (64, 64, 2)

    def test_physics_constrained_flow(self):
        from optical_flow import compute_optical_flow
        flow, opt, wind = compute_optical_flow(self.f0, self.f1, self.u, self.v)
        assert flow.shape == (64, 64, 2)
        assert opt.shape  == (64, 64, 2)
        assert wind.shape == (64, 64, 2)

    def test_warp_frame(self):
        from optical_flow import warp_frame, compute_farneback_flow
        flow   = compute_farneback_flow(self.f0, self.f1)
        warped = warp_frame(self.f0, flow)
        assert warped.shape == self.f0.shape
        assert warped.min() >= 0.0
        assert warped.max() <= 1.0

    def test_flow_rgb(self):
        from optical_flow import compute_farneback_flow, flow_to_rgb
        flow = compute_farneback_flow(self.f0, self.f1)
        rgb  = flow_to_rgb(flow)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8


# ---------------------------------------------------------------------------
# Three-frame generator tests
# ---------------------------------------------------------------------------

class TestThreeFrameGenerator:
    def setup_method(self):
        from data_loader import generate_sample_frames
        from optical_flow import compute_optical_flow
        data = generate_sample_frames(H=64, W=64, seed=1)
        self.f0, self.f1 = data["frame_t0"], data["frame_t1"]
        u, v = data["wind_u"], data["wind_v"]
        self.fwd, _, self.wind = compute_optical_flow(self.f0, self.f1, u, v)
        self.bwd, _, _         = compute_optical_flow(self.f1, self.f0, u, v)

    def test_mid_frame(self):
        from three_frame_generator import generate_mid_frame
        mid = generate_mid_frame(self.f0, self.f1, self.fwd, self.bwd)
        assert mid.shape == self.f0.shape
        assert 0 <= mid.min() and mid.max() <= 1.0

    def test_quarter_frame(self):
        from three_frame_generator import generate_mid_frame, generate_quarter_frame
        mid = generate_mid_frame(self.f0, self.f1, self.fwd, self.bwd)
        q   = generate_quarter_frame(self.f0, mid, self.fwd, self.bwd)
        assert q.shape == self.f0.shape

    def test_three_quarter_frame(self):
        from three_frame_generator import generate_mid_frame, generate_three_quarter_frame
        mid  = generate_mid_frame(self.f0, self.f1, self.fwd, self.bwd)
        tq   = generate_three_quarter_frame(mid, self.f1, self.fwd, self.bwd)
        assert tq.shape == self.f0.shape

    def test_generate_all(self):
        from three_frame_generator import generate_all_frames
        from data_loader import generate_sample_frames
        data = generate_sample_frames(H=64, W=64, seed=2)
        results = generate_all_frames(data["frame_t0"], data["frame_t1"],
                                      data["wind_u"], data["wind_v"])
        assert "frame_025" in results
        assert "frame_050" in results
        assert "frame_075" in results
        assert results["frame_050"].shape == (64, 64)


# ---------------------------------------------------------------------------
# Confidence map tests
# ---------------------------------------------------------------------------

class TestConfidenceMap:
    def setup_method(self):
        self.flow_opt  = np.random.randn(64, 64, 2).astype(np.float32)
        self.flow_wind = np.random.randn(64, 64, 2).astype(np.float32) * 0.5

    def test_confidence_range(self):
        from confidence_map import compute_confidence_map
        conf = compute_confidence_map(self.flow_opt, self.flow_wind)
        assert conf.shape == (64, 64)
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0

    def test_confidence_stats(self):
        from confidence_map import compute_confidence_map, confidence_stats
        conf  = compute_confidence_map(self.flow_opt, self.flow_wind)
        stats = confidence_stats(conf)
        assert "mean" in stats
        assert 0 <= stats["mean"] <= 1


# ---------------------------------------------------------------------------
# Cloud motion score tests
# ---------------------------------------------------------------------------

class TestCloudMotionScore:
    def test_score_range(self):
        from cloud_motion_score import compute_cloud_motion_score
        pred_flow = np.random.randn(64, 64, 2).astype(np.float32)
        wind_flow = np.random.randn(64, 64, 2).astype(np.float32)
        score = compute_cloud_motion_score(pred_flow, wind_flow)
        assert 0.0 <= score <= 100.0

    def test_perfect_alignment(self):
        from cloud_motion_score import compute_cloud_motion_score
        flow = np.ones((64, 64, 2), dtype=np.float32)
        score = compute_cloud_motion_score(flow, flow)
        assert score >= 95.0  # should be ~100

    def test_spatial_map_range(self):
        from cloud_motion_score import compute_spatial_consistency_map
        pred = np.random.randn(64, 64, 2).astype(np.float32)
        wind = np.random.randn(64, 64, 2).astype(np.float32)
        smap = compute_spatial_consistency_map(pred, wind)
        assert smap.shape == (64, 64)
        assert smap.min() >= 0.0
        assert smap.max() <= 1.0


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def setup_method(self):
        self.pred   = np.random.rand(64, 64).astype(np.float32)
        self.target = np.random.rand(64, 64).astype(np.float32)

    def test_mse_identical(self):
        from metrics import compute_mse
        assert compute_mse(self.pred, self.pred) == pytest.approx(0.0, abs=1e-8)

    def test_psnr_identical(self):
        from metrics import compute_psnr
        assert compute_psnr(self.pred, self.pred) == float("inf")

    def test_ssim_range(self):
        from metrics import compute_ssim
        s = compute_ssim(self.pred, self.target)
        assert -1.0 <= s <= 1.0

    def test_fsim_range(self):
        from metrics import compute_fsim
        f = compute_fsim(self.pred, self.target)
        assert 0.0 <= f <= 1.0

    def test_evaluate_frame_keys(self):
        from metrics import evaluate_frame
        result = evaluate_frame(self.pred, self.target, "test")
        assert all(k in result for k in ["mse", "psnr", "ssim", "fsim"])

    def test_temporal_consistency(self):
        from metrics import temporal_consistency_score
        frames = [np.random.rand(64, 64).astype(np.float32) for _ in range(5)]
        score  = temporal_consistency_score(frames)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_build_model(self):
        from interpolation_model import build_model
        model = build_model({"base_filters": 8, "depth": 2})
        assert model is not None

    def test_forward_pass(self):
        import torch
        from interpolation_model import build_model
        model = build_model({"base_filters": 8, "depth": 2})
        model.eval()
        x   = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_loss_shape(self):
        import torch
        from interpolation_model import InterpolationLoss
        loss_fn = InterpolationLoss()
        pred   = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)
        loss   = loss_fn(pred, target)
        assert loss.ndim == 0    # scalar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
