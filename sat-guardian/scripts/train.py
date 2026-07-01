"""
SAT-GUARDIAN Training Script
==============================
Dummy training pipeline for the LightUNet interpolation model.
Uses synthetic data to demonstrate the training loop.

Usage
-----
    cd sat-guardian
    python scripts/train.py
    python scripts/train.py --epochs 20 --batch-size 4
    python scripts/train.py --data-dir data/processed/
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data_loader import generate_sample_frames
from interpolation_model import LightUNet, InterpolationLoss, build_model, get_device, save_checkpoint
from optical_flow import compute_optical_flow, flow_to_magnitude_angle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SatelliteFrameDataset(Dataset):
    """
    Synthetic dataset of (T0, T1, flow_mag, T_mid) tuples for training.
    In a real scenario, this would read from processed .npy triplets.
    """

    def __init__(self, n_samples: int = 200, H: int = 256, W: int = 256, seed: int = 42):
        self.H, self.W = H, W
        self.samples = []
        rng = np.random.default_rng(seed)
        logger.info("Generating %d synthetic training samples ...", n_samples)

        for i in range(n_samples):
            data = generate_sample_frames(H=H, W=W, seed=int(rng.integers(0, 100000)))
            t0  = data["frame_t0"]
            t1  = data["frame_t1"]
            t05 = 0.5 * t0 + 0.5 * t1  # pseudo ground truth midframe

            flow, _, _ = compute_optical_flow(t0, t1)
            mag, _ = flow_to_magnitude_angle(flow)
            mag_norm = (mag - mag.min()) / (mag.max() + 1e-8)

            self.samples.append({
                "input":  np.stack([t0, t1, mag_norm], axis=0).astype(np.float32),
                "target": t05[np.newaxis].astype(np.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.from_numpy(s["input"]), torch.from_numpy(s["target"])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = get_device(args.device)
    logger.info("Device: %s", device)

    # Dataset + loaders
    dataset = SatelliteFrameDataset(
        n_samples=args.n_samples, H=args.height, W=args.width
    )
    val_size   = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    logger.info("Train=%d | Val=%d", train_size, val_size)

    # Model
    model = build_model({
        "input_channels":  3,
        "output_channels": 1,
        "base_filters":    args.base_filters,
        "depth":           args.depth,
        "dropout":         0.1,
    }).to(device)

    criterion = InterpolationLoss(lambda_l1=1.0, lambda_ssim=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    logger.info("Training for %d epochs ...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                pred = model(inp)
                val_losses.append(criterion(pred, tgt).item())

        scheduler.step()

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        elapsed = time.time() - t0

        logger.info(
            "Epoch [%3d/%d] | train=%.4f | val=%.4f | lr=%.2e | %.1fs",
            epoch, args.epochs, train_loss, val_loss,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, str(ckpt_dir / "best_model.pth"), epoch, val_loss)

        if epoch % args.save_every == 0:
            save_checkpoint(model, str(ckpt_dir / f"epoch_{epoch:03d}.pth"), epoch, val_loss)

    # Save training curves
    _save_training_curves(history, str(ckpt_dir / "training_curves.png"))
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


def _save_training_curves(history: dict, path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.plot(history["train_loss"], label="Train", color="#00d4ff")
    ax.plot(history["val_loss"],   label="Val",   color="#ff6b6b")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss",  color="white")
    ax.set_title("Training Curves – LightUNet", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1d2e", labelcolor="white", edgecolor="#444")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Training curves saved: %s", path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAT-GUARDIAN LightUNet Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--n-samples",     type=int,   default=100,
                        help="Number of synthetic training samples")
    parser.add_argument("--height",        type=int,   default=256)
    parser.add_argument("--width",         type=int,   default=256)
    parser.add_argument("--base-filters",  type=int,   default=32)
    parser.add_argument("--depth",         type=int,   default=4)
    parser.add_argument("--save-every",    type=int,   default=5)
    parser.add_argument("--checkpoint-dir",default=str(_ROOT / "models"))
    parser.add_argument("--device",        default="auto")
    parser.add_argument("--log-level",     default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args)
