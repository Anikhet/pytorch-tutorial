#!/usr/bin/env python3
"""
H100/Colab-optimized training for Sketch-to-Image Diffusion.

Upload this to Google Colab and run:
    !pip install torch torchvision tqdm matplotlib pillow
    !python train_colab.py --epochs 200

With H100, training should take ~10-15 minutes for 200 epochs.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import urllib.request


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.norm2(F.silu(self.conv2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class SketchUNet(nn.Module):
    """U-Net for sketch-to-image diffusion (GPU version with attention)."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 128, time_emb_dim: int = 256):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.down1 = nn.ModuleList([ResidualBlock(channels[0], channels[1], time_emb_dim), ResidualBlock(channels[1], channels[1], time_emb_dim)])
        self.down1_sample = nn.Conv2d(channels[1], channels[1], 3, stride=2, padding=1)

        self.down2 = nn.ModuleList([ResidualBlock(channels[1], channels[2], time_emb_dim), ResidualBlock(channels[2], channels[2], time_emb_dim), AttentionBlock(channels[2])])
        self.down2_sample = nn.Conv2d(channels[2], channels[2], 3, stride=2, padding=1)

        self.down3 = nn.ModuleList([ResidualBlock(channels[2], channels[3], time_emb_dim), ResidualBlock(channels[3], channels[3], time_emb_dim), AttentionBlock(channels[3])])
        self.down3_sample = nn.Conv2d(channels[3], channels[3], 3, stride=2, padding=1)

        # Middle
        self.mid1 = ResidualBlock(channels[3], channels[3], time_emb_dim)
        self.mid_attn = AttentionBlock(channels[3])
        self.mid2 = ResidualBlock(channels[3], channels[3], time_emb_dim)

        # Decoder
        self.up3_sample = nn.ConvTranspose2d(channels[3], channels[3], 4, stride=2, padding=1)
        self.up3 = nn.ModuleList([ResidualBlock(channels[3] + channels[3], channels[2], time_emb_dim), ResidualBlock(channels[2], channels[2], time_emb_dim), AttentionBlock(channels[2])])

        self.up2_sample = nn.ConvTranspose2d(channels[2], channels[2], 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([ResidualBlock(channels[2] + channels[2], channels[1], time_emb_dim), ResidualBlock(channels[1], channels[1], time_emb_dim), AttentionBlock(channels[1])])

        self.up1_sample = nn.ConvTranspose2d(channels[1], channels[1], 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([ResidualBlock(channels[1] + channels[1], channels[0], time_emb_dim), ResidualBlock(channels[0], channels[0], time_emb_dim)])

        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def get_time_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x: torch.Tensor, sketch: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sketch, x], dim=1)
        t_emb = self.time_mlp(self.get_time_embedding(t, self.init_conv.out_channels))

        h = self.init_conv(x)

        # Encoder
        h1 = h
        for block in self.down1:
            h1 = block(h1, t_emb) if isinstance(block, ResidualBlock) else block(h1)
        h1_down = self.down1_sample(h1)

        h2 = h1_down
        for block in self.down2:
            h2 = block(h2, t_emb) if isinstance(block, ResidualBlock) else block(h2)
        h2_down = self.down2_sample(h2)

        h3 = h2_down
        for block in self.down3:
            h3 = block(h3, t_emb) if isinstance(block, ResidualBlock) else block(h3)
        h3_down = self.down3_sample(h3)

        # Middle
        mid = self.mid1(h3_down, t_emb)
        mid = self.mid_attn(mid)
        mid = self.mid2(mid, t_emb)

        # Decoder
        up3 = self.up3_sample(mid)
        up3 = torch.cat([up3, h3], dim=1)
        for block in self.up3:
            up3 = block(up3, t_emb) if isinstance(block, ResidualBlock) else block(up3)

        up2 = self.up2_sample(up3)
        up2 = torch.cat([up2, h2], dim=1)
        for block in self.up2:
            up2 = block(up2, t_emb) if isinstance(block, ResidualBlock) else block(up2)

        up1 = self.up1_sample(up2)
        up1 = torch.cat([up1, h1], dim=1)
        for block in self.up1:
            up1 = block(up1, t_emb) if isinstance(block, ResidualBlock) else block(up1)

        return self.out_conv(self.out_norm(F.silu(up1)))


class NoiseScheduler:
    """DDPM/DDIM noise scheduler."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, device: str = "cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def ddim_sample(self, model, sketch: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        batch_size = sketch.shape[0]
        device = sketch.device
        x = torch.randn(batch_size, 3, sketch.shape[2], sketch.shape[3], device=device)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(x, sketch, t_batch)
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            x0_pred = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

        return x


def download_edges2shoes(data_dir: str = "data"):
    """Download edges2shoes dataset."""
    import tarfile
    url = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz"
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    tar_path = data_path / "edges2shoes.tar.gz"
    extract_path = data_path / "edges2shoes"

    if extract_path.exists():
        print(f"Dataset already exists at {extract_path}")
        return extract_path

    print("Downloading edges2shoes dataset (~300MB)...")
    urllib.request.urlretrieve(url, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_path)
    tar_path.unlink()
    print(f"Dataset ready at {extract_path}")
    return extract_path


class Edges2ShoesDataset(Dataset):
    """Dataset for edges2shoes paired images."""

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 64):
        self.root = Path(root_dir) / split
        self.image_files = sorted(self.root.glob("*.jpg"))
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size * 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        img = self.transform(img)
        w = img.shape[2] // 2
        edge = img[:, :, :w].mean(dim=0, keepdim=True)
        shoe = img[:, :, w:]
        return edge, shoe


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, size: int = 2000, image_size: int = 64):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        edge = torch.zeros(1, self.image_size, self.image_size)
        image = torch.zeros(3, self.image_size, self.image_size)
        cx, cy = np.random.randint(16, self.image_size - 16, 2)
        rx, ry = np.random.randint(8, 20, 2)
        color = torch.rand(3) * 2 - 1
        y, x = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing='ij')
        mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
        edge[0, mask] = 1.0
        for c in range(3):
            image[c][mask] = color[c]
        return edge, image


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    if args.synthetic:
        train_dataset = SyntheticDataset(size=5000, image_size=args.image_size)
        val_dataset = SyntheticDataset(size=500, image_size=args.image_size)
    else:
        data_path = download_edges2shoes()
        train_dataset = Edges2ShoesDataset(data_path, "train", args.image_size)
        val_dataset = Edges2ShoesDataset(data_path, "val", args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = SketchUNet(base_channels=args.base_channels).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    noise_scheduler = NoiseScheduler(num_timesteps=1000, device=device)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    save_dir = Path("pretrained")
    save_dir.mkdir(exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for edges, images in pbar:
            edges, images = edges.to(device), images.to(device)
            t = torch.randint(0, 1000, (edges.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = noise_scheduler.add_noise(images, noise, t)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    pred = model(noisy, edges, t)
                    loss = F.mse_loss(pred, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(noisy, edges, t)
                loss = F.mse_loss(pred, noise)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for edges, images in val_loader:
                edges, images = edges.to(device), images.to(device)
                t = torch.randint(0, 1000, (edges.shape[0],), device=device)
                noise = torch.randn_like(images)
                noisy = noise_scheduler.add_noise(images, noise, t)
                pred = model(noisy, edges, t)
                val_loss += F.mse_loss(pred, noise).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'val_loss': val_loss}, save_dir / "sketch2image_best.pth")

        if (epoch + 1) % 50 == 0:
            generate_samples(model, noise_scheduler, val_loader, device, save_dir / f"samples_epoch{epoch+1}.png")

    torch.save(model.state_dict(), save_dir / "sketch2image_final.pth")
    print(f"Training complete! Model saved to {save_dir}")


def generate_samples(model, scheduler, loader, device, save_path, n=4):
    model.eval()
    edges, real = next(iter(loader))
    edges, real = edges[:n].to(device), real[:n].to(device)

    with torch.no_grad():
        gen = scheduler.ddim_sample(model, edges, num_steps=50)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    for i in range(n):
        axes[i, 0].imshow(edges[i, 0].cpu(), cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(np.clip((gen[i].permute(1,2,0).cpu().numpy()+1)/2, 0, 1))
        axes[i, 1].axis('off')
        axes[i, 2].imshow(np.clip((real[i].permute(1,2,0).cpu().numpy()+1)/2, 0, 1))
        axes[i, 2].axis('off')
    axes[0, 0].set_title("Edge")
    axes[0, 1].set_title("Generated")
    axes[0, 2].set_title("Real")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    train(args)
