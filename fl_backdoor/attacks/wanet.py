"""WaNet attack utilities.

This module mirrors the BadNets API so client/server can switch attacks
without changing the main training flow.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .base import AttackBase, AttackConfig
from .badnets import select_malicious_clients


def _get_image_hw(image: torch.Tensor) -> tuple[int, int]:
    """Return image height and width for [C, H, W] or [H, W]."""
    if image.dim() == 3:
        _, h, w = image.shape
        return int(h), int(w)
    if image.dim() == 2:
        h, w = image.shape
        return int(h), int(w)
    raise ValueError(f"Unsupported image shape for WaNet: {tuple(image.shape)}")


def _make_identity_grid(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build normalized identity grid in [-1, 1]."""
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)  # [1, H, W, 2]


def _make_warp_grid(
    h: int,
    w: int,
    grid_size: int,
    noise_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a smooth spatial warp grid.

    This is a practical WaNet-style approximation:
    - sample low-resolution random displacement
    - upsample to full resolution
    - add to identity grid
    """
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer.")
    if float(noise_scale) <= 0.0:
        raise ValueError("noise_scale must be positive.")

    rng = np.random.default_rng(seed)

    # Low-res displacement: [1, 2, grid_size, grid_size]
    displacement = rng.normal(loc=0.0, scale=1.0, size=(1, 2, grid_size, grid_size))
    displacement = torch.tensor(displacement, device=device, dtype=dtype)

    # Upsample to full resolution: [1, 2, H, W]
    displacement = F.interpolate(
        displacement,
        size=(h, w),
        mode="bicubic",
        align_corners=True,
    )

    # Back to [1, H, W, 2]
    displacement = displacement.permute(0, 2, 3, 1)

    # Normalize and scale
    max_abs = displacement.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    displacement = displacement / max_abs
    displacement = displacement * float(noise_scale)

    identity = _make_identity_grid(h, w, device=device, dtype=dtype)
    grid = identity + displacement
    return grid.clamp(-1.0, 1.0).contiguous()


def _warp_image(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Apply grid_sample-based warp to a single image."""
    original_dtype = image.dtype
    original_dim = image.dim()

    if original_dim == 2:
        image = image.unsqueeze(0)  # [1, H, W]
    elif original_dim != 3:
        raise ValueError(f"Unsupported image shape for WaNet: {tuple(image.shape)}")

    # grid_sample expects [N, C, H, W]
    x = image.unsqueeze(0)
    if not x.is_floating_point():
        x = x.float()

    warped = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )

    warped = warped.squeeze(0)  # [C, H, W]
    if original_dim == 2:
        warped = warped.squeeze(0)  # [H, W]

    if original_dtype.is_floating_point:
        warped = warped.to(original_dtype)

    return warped


class PoisonedWanetDataset(Dataset):
    """Wrap a dataset and apply WaNet to a fixed subset of samples."""

    def __init__(
        self,
        base_dataset,
        poison_rate: float,
        target_label: int,
        grid_size: int,
        noise_scale: float,
        seed: int = 42,
        exclude_target_label: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.poison_rate = float(poison_rate)
        self.target_label = int(target_label)
        self.grid_size = int(grid_size)
        self.noise_scale = float(noise_scale)
        self.seed = int(seed)
        self.exclude_target_label = bool(exclude_target_label)

        rng = np.random.default_rng(self.seed)
        self.poison_flags = rng.random(len(base_dataset)) < self.poison_rate
        self._grid_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _grid_for(self, image: torch.Tensor) -> torch.Tensor:
        h, w = _get_image_hw(image)
        key = (h, w, str(image.device))

        grid = self._grid_cache.get(key)
        if grid is None:
            grid = _make_warp_grid(
                h=h,
                w=w,
                grid_size=self.grid_size,
                noise_scale=self.noise_scale,
                seed=self.seed + 1,
                device=image.device,
                dtype=torch.float32,
            )
            self._grid_cache[key] = grid
        return grid

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]

        image = sample["img"].clone()
        label = torch.as_tensor(sample["label"]).long()

        should_poison = bool(self.poison_flags[idx])
        if self.exclude_target_label and int(label.item()) == self.target_label:
            should_poison = False

        if should_poison:
            image = _warp_image(image, self._grid_for(image))
            label = torch.tensor(self.target_label, dtype=torch.long)

        return {"img": image, "label": label}


class TriggeredWanetDataset(Dataset):
    """Apply WaNet to every sample, used for ASR evaluation."""

    def __init__(
        self,
        base_dataset,
        grid_size: int,
        noise_scale: float,
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.grid_size = int(grid_size)
        self.noise_scale = float(noise_scale)
        self.seed = int(seed)
        self._grid_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _grid_for(self, image: torch.Tensor) -> torch.Tensor:
        h, w = _get_image_hw(image)
        key = (h, w, str(image.device))

        grid = self._grid_cache.get(key)
        if grid is None:
            grid = _make_warp_grid(
                h=h,
                w=w,
                grid_size=self.grid_size,
                noise_scale=self.noise_scale,
                seed=self.seed + 1,
                device=image.device,
                dtype=torch.float32,
            )
            self._grid_cache[key] = grid
        return grid

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        image = sample["img"].clone()
        label = torch.as_tensor(sample["label"]).long()
        image = _warp_image(image, self._grid_for(image))
        return {"img": image, "label": label}


class WanetAttack(AttackBase):
    """WaNet attack implementation following the common AttackBase API."""

    def __init__(self, config: AttackConfig) -> None:
        super().__init__(config)
        self.grid_size = int(config.extra.get("grid_size", config.trigger_size))
        self.noise_scale = float(config.extra.get("noise_scale", 0.05))

        if self.grid_size <= 0:
            raise ValueError("grid_size must be a positive integer.")
        if self.noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive.")

    def select_malicious_clients(self, num_clients: int) -> set[int]:
        return select_malicious_clients(
            num_clients=num_clients,
            malicious_ratio=self.config.malicious_ratio,
            seed=self.config.seed,
        )

    def get_poisoned_loader(self, trainloader: DataLoader) -> DataLoader:
        poisoned_dataset = PoisonedWanetDataset(
            base_dataset=trainloader.dataset,
            poison_rate=self.config.poison_rate,
            target_label=self.config.target_label,
            grid_size=self.grid_size,
            noise_scale=self.noise_scale,
            seed=self.config.seed,
            exclude_target_label=True,
        )

        return DataLoader(
            poisoned_dataset,
            batch_size=trainloader.batch_size or 1,
            shuffle=True,
            num_workers=getattr(trainloader, "num_workers", 0),
            pin_memory=getattr(trainloader, "pin_memory", False),
            drop_last=getattr(trainloader, "drop_last", False),
        )

    def get_triggered_loader(self, testloader: DataLoader) -> DataLoader:
        triggered_dataset = TriggeredWanetDataset(
            base_dataset=testloader.dataset,
            grid_size=self.grid_size,
            noise_scale=self.noise_scale,
            seed=self.config.seed,
        )

        return DataLoader(
            triggered_dataset,
            batch_size=testloader.batch_size or 1,
            shuffle=False,
            num_workers=getattr(testloader, "num_workers", 0),
            pin_memory=getattr(testloader, "pin_memory", False),
            drop_last=getattr(testloader, "drop_last", False),
        )


def build_wanet_attack(
    malicious_ratio: float = 0.2,
    poison_rate: float = 0.05,
    target_label: int = 0,
    trigger_size: int = 4,
    seed: int = 42,
    grid_size: int | None = None,
    noise_scale: float = 0.05,
) -> WanetAttack:
    """Convenience factory for WaNet."""
    resolved_grid_size = int(trigger_size if grid_size is None else grid_size)

    config = AttackConfig(
        attack_type="wanet",
        malicious_ratio=malicious_ratio,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=resolved_grid_size,
        seed=seed,
        extra={
            "grid_size": resolved_grid_size,
            "noise_scale": float(noise_scale),
        },
    )
    return WanetAttack(config)