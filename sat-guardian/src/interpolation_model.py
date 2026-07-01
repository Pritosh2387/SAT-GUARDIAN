"""
SAT-GUARDIAN: Lightweight U-Net Interpolation Model
=====================================================
A compact encoder-decoder U-Net that takes (T0, T1, flow_magnitude) as
input and predicts an intermediate satellite frame.

Architecture
------------
Input : (B, 3, H, W)  – channel 0: T0, channel 1: T1, channel 2: flow mag
Output: (B, 1, H, W)  – predicted intermediate frame in [0, 1]

Encoder: 4 downsampling blocks (Conv → BN → ReLU → MaxPool)
Bottleneck: 2 × Conv
Decoder: 4 upsampling blocks (ConvTranspose → skip-cat → Conv → BN → ReLU)
Head: 1×1 Conv → Sigmoid
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Two ConvBNReLU + MaxPool. Returns (pooled, skip)."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch,  out_ch, dropout=dropout)
        self.conv2 = ConvBNReLU(out_ch, out_ch, dropout=dropout)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv2(self.conv1(x))
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """ConvTranspose2d up-sample → concat skip → two ConvBNReLU."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch, dropout=dropout)
        self.conv2 = ConvBNReLU(out_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if size mismatch
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


# ---------------------------------------------------------------------------
# LightUNet
# ---------------------------------------------------------------------------

class LightUNet(nn.Module):
    """
    Lightweight U-Net for satellite frame interpolation.

    Parameters
    ----------
    in_channels  : int   number of input channels (default 3)
    out_channels : int   number of output channels (default 1)
    base_filters : int   filters in the first encoder block (doubles each level)
    depth        : int   number of encoder / decoder levels (default 4)
    dropout      : float dropout probability (0 = disabled)
    """

    def __init__(
        self,
        in_channels:  int = 3,
        out_channels: int = 1,
        base_filters: int = 32,
        depth:        int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders: nn.ModuleList = nn.ModuleList()
        in_ch = in_channels
        self.encoder_channels: List[int] = []
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(in_ch, out_ch, dropout=dropout))
            self.encoder_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        bn_ch = base_filters * (2 ** depth)
        self.bottleneck = nn.Sequential(
            ConvBNReLU(in_ch,  bn_ch, dropout=dropout),
            ConvBNReLU(bn_ch,  bn_ch, dropout=dropout),
        )

        # Decoder
        self.decoders: nn.ModuleList = nn.ModuleList()
        dec_in = bn_ch
        for i in reversed(range(depth)):
            skip_ch = self.encoder_channels[i]
            dec_out = base_filters * (2 ** i)
            self.decoders.append(DecoderBlock(dec_in, skip_ch, dec_out, dropout=dropout))
            dec_in = dec_out

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(dec_in, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        logger.info("LightUNet initialised | depth=%d | base=%d | params=%s",
                    depth, base_filters, f"{n_params:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class InterpolationLoss(nn.Module):
    """
    Combined loss for frame interpolation:
        L = λ_l1 * L1 + λ_ssim * (1 - SSIM) + λ_perceptual * L_percep
    """

    def __init__(
        self,
        lambda_l1:      float = 1.0,
        lambda_ssim:    float = 0.5,
        window_size:    int   = 11,
    ):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = F.l1_loss(pred, target)

        ssim_val = self._ssim(pred, target)
        ssim_loss = 1.0 - ssim_val

        total = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss
        return total

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Simplified differentiable SSIM."""
        C1, C2 = 0.01**2, 0.03**2
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.window_size // 2)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.window_size // 2)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2

        sig1 = F.avg_pool2d(img1 * img1, self.window_size, 1, self.window_size // 2) - mu1_sq
        sig2 = F.avg_pool2d(img2 * img2, self.window_size, 1, self.window_size // 2) - mu2_sq
        sig12= F.avg_pool2d(img1 * img2, self.window_size, 1, self.window_size // 2) - mu1_mu2

        numerator   = (2*mu1_mu2 + C1) * (2*sig12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2)
        return (numerator / (denominator + 1e-8)).mean()


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def build_model(config: Optional[Dict[str, Any]] = None) -> LightUNet:
    """Build a LightUNet from a config dict (or defaults)."""
    cfg = config or {}
    return LightUNet(
        in_channels  = cfg.get("input_channels",  3),
        out_channels = cfg.get("output_channels", 1),
        base_filters = cfg.get("base_filters",   32),
        depth        = cfg.get("depth",           4),
        dropout      = cfg.get("dropout",         0.1),
    )


def get_device(preference: str = "auto") -> torch.device:
    """Select compute device: auto, cuda, mps, or cpu."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def frames_to_tensor(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    flow_mag: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Stack T0, T1, and flow magnitude into a (1, 3, H, W) model input tensor.
    """
    inp = np.stack([frame_t0, frame_t1, flow_mag], axis=0)[np.newaxis]  # (1,3,H,W)
    return torch.from_numpy(inp).float().to(device)


def save_checkpoint(model: LightUNet, path: str, epoch: int, loss: float):
    """Save model weights and training metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "loss":  loss,
        "state_dict": model.state_dict(),
        "model_config": {
            "in_channels":  model.encoders[0].conv1.block[0].weight.shape[1],
            "out_channels": model.head[0].weight.shape[0],
            "depth":        model.depth,
        },
    }, path)
    logger.info("Checkpoint saved: %s (epoch=%d, loss=%.4f)", path, epoch, loss)


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> LightUNet:
    """Load a model from a checkpoint file."""
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt.get("model_config", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info("Loaded checkpoint: %s (epoch=%d)", path, ckpt.get("epoch", -1))
    return model
