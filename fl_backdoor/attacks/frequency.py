"""Frequency-domain backdoor attack utilities.

This module provides a practical frequency attack that mirrors
the existing BadNets / WaNet API:

- choose a fixed subset of malicious clients
- poison a fraction of local training samples
- apply a reusable frequency trigger to every test sample for ASR evaluation

Supported frequency modes:
- DCT: modify low/high frequency coefficients via orthonormal 2D DCT
- FFT: modify low/high frequency coefficients via rFFT2 / irFFT2

Compared with a pure random-noise trigger, this version defaults to a
structured spectral trigger that is easier for the model to learn,
which usually makes ASR much more stable.

Important:
- If `img_raw` exists in the dataset sample, this attack operates on the
  raw image domain first, then re-normalizes to model space.
- This keeps the attack visible to the model before normalization, which
  improves backdoor learnability without affecting BadNets / WaNet.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import AttackBase, AttackConfig
from .selection import normalize_fixed_malicious_clients, select_malicious_clients

FrequencyMode = Literal["dct", "fft"]
FrequencyBand = Literal["low", "high"]
TriggerStyle = Literal["structured", "random"]


def _normalize_name(value: str) -> str:
    return str(value).lower().strip().replace("-", "_")


def _get_image_hw(image: torch.Tensor) -> tuple[int, int]:
    if image.dim() == 3:
        _, h, w = image.shape
        return int(h), int(w)
    if image.dim() == 2:
        h, w = image.shape
        return int(h), int(w)
    raise ValueError(
        f"Unsupported image shape for frequency attack: {tuple(image.shape)}"
    )


def _ensure_channel_first(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return a [C, H, W] tensor and whether the input was originally 2D."""
    if image.dim() == 3:
        return image, False
    if image.dim() == 2:
        return image.unsqueeze(0), True
    raise ValueError(
        f"Unsupported image shape for frequency attack: {tuple(image.shape)}"
    )


def _restore_shape(image: torch.Tensor, was_2d: bool) -> torch.Tensor:
    if was_2d:
        return image.squeeze(0)
    return image


def _normalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] image tensor to CIFAR-10 model space [-1, 1]."""
    return torch.clamp((x - 0.5) / 0.5, -1.0, 1.0)


@lru_cache(maxsize=None)
def _dct_matrices(h: int, w: int, device_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return orthonormal DCT-II matrices for height and width."""
    device = torch.device(device_str)

    def make_matrix(n: int) -> torch.Tensor:
        mat = torch.empty((n, n), dtype=torch.float32, device=device)
        scale0 = np.sqrt(1.0 / n)
        scale = np.sqrt(2.0 / n)
        for k in range(n):
            alpha = scale0 if k == 0 else scale
            for i in range(n):
                mat[k, i] = float(alpha * np.cos(np.pi * (i + 0.5) * k / n))
        return mat

    return make_matrix(h), make_matrix(w)


def _dct2(x: torch.Tensor) -> torch.Tensor:
    """Apply orthonormal 2D DCT to x with shape [C, H, W] or [H, W]."""
    x3d, was_2d = _ensure_channel_first(x)
    if not x3d.is_floating_point():
        x3d = x3d.float()

    h, w = x3d.shape[-2], x3d.shape[-1]
    dh, dw = _dct_matrices(int(h), int(w), str(x3d.device))

    coeffs = torch.einsum("uh,chw->cuw", dh, x3d)
    coeffs = torch.einsum("cuw,vw->cuv", coeffs, dw.t())
    return _restore_shape(coeffs, was_2d)


def _idct2(x: torch.Tensor) -> torch.Tensor:
    """Inverse of _dct2."""
    x3d, was_2d = _ensure_channel_first(x)
    if not x3d.is_floating_point():
        x3d = x3d.float()

    h, w = x3d.shape[-2], x3d.shape[-1]
    dh, dw = _dct_matrices(int(h), int(w), str(x3d.device))

    image = torch.einsum("cuw,vw->cuv", x3d, dw)
    image = torch.einsum("uh,cuv->chv", dh.t(), image)
    return _restore_shape(image, was_2d)


def _resolve_window_size(window_size: int | None, trigger_size: int) -> int:
    resolved = int(trigger_size if window_size is None else window_size)
    if resolved <= 0:
        raise ValueError("frequency window size must be a positive integer.")
    return resolved


def _resolve_mode(mode: str) -> str:
    mode = _normalize_name(mode)
    if mode in {"dct", "fft"}:
        return mode
    raise ValueError("frequency_mode must be one of: 'dct', 'fft'.")


def _resolve_band(band: str) -> str:
    band = _normalize_name(band)
    if band in {"low", "high"}:
        return band
    raise ValueError("frequency_band must be one of: 'low', 'high'.")


def _resolve_style(style: str) -> str:
    style = _normalize_name(style)
    if style in {"structured", "random"}:
        return style
    raise ValueError("trigger_style must be one of: 'structured', 'random'.")


def _band_slices(h: int, w: int, window: int, band: str, *, fft: bool) -> tuple[slice, slice]:
    window = max(1, min(int(window), int(h), int(w)))
    if fft:
        # rFFT width is reduced to W//2 + 1
        w = w // 2 + 1
        window = max(1, min(window, int(h), int(w)))

    if band == "low":
        return slice(0, window), slice(0, window)

    # High-frequency corner band.
    return slice(h - window, h), slice(w - window, w)


def _make_structured_pattern(
    *,
    mode: str,
    h: int,
    w: int,
    window: int,
    band: str,
    intensity: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a deterministic smooth trigger in the selected spectral band."""
    rs, cs = _band_slices(h, w, window, band, fft=(mode == "fft"))
    ph = max(1, rs.stop - rs.start)
    pw = max(1, cs.stop - cs.start)

    ys = torch.linspace(0.0, 1.0, ph, device=device, dtype=torch.float32)
    xs = torch.linspace(0.0, 1.0, pw, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    if band == "low":
        dist = yy**2 + xx**2
    else:
        dist = (1.0 - yy) ** 2 + (1.0 - xx) ** 2

    sigma = 0.35
    pattern = torch.exp(-dist / (2.0 * sigma**2))
    pattern = pattern / pattern.max().clamp_min(1e-6)
    pattern = pattern * float(intensity)
    return pattern


def _make_random_pattern(
    *,
    mode: str,
    h: int,
    w: int,
    window: int,
    band: str,
    intensity: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a random but deterministic trigger pattern.

    This version is still fixed across the run (seeded), but unlike the
    earlier zero-mean Gaussian version it uses nonnegative amplitudes,
    which is easier for a classifier to learn.
    """
    rng = np.random.default_rng(seed)
    rs, cs = _band_slices(h, w, window, band, fft=(mode == "fft"))
    ph = max(1, rs.stop - rs.start)
    pw = max(1, cs.stop - cs.start)

    noise = rng.normal(loc=0.0, scale=1.0, size=(ph, pw))
    noise = np.abs(noise)
    noise = noise / (np.max(noise) + 1e-12)
    noise = torch.tensor(noise, dtype=torch.float32, device=device)
    noise = noise * float(intensity)
    return noise


def _make_trigger_pattern(
    *,
    mode: str,
    h: int,
    w: int,
    window: int,
    band: str,
    intensity: float,
    seed: int,
    device: torch.device,
    trigger_style: str,
) -> torch.Tensor:
    """Create a reusable trigger in the selected spectral band."""
    style = _resolve_style(trigger_style)

    if style == "structured":
        return _make_structured_pattern(
            mode=mode,
            h=h,
            w=w,
            window=window,
            band=band,
            intensity=intensity,
            device=device,
        )

    return _make_random_pattern(
        mode=mode,
        h=h,
        w=w,
        window=window,
        band=band,
        intensity=intensity,
        seed=seed,
        device=device,
    )


def _apply_frequency_trigger(
    image: torch.Tensor,
    *,
    mode: str,
    band: str,
    window: int,
    intensity: float,
    seed: int,
    mix_alpha: float,
    trigger_style: str,
    input_domain: str = "normalized",
) -> torch.Tensor:
    """Apply a fixed spectral trigger to one image.

    input_domain:
        - "raw": image is in [0, 1], output is clamped to [0, 1]
        - "normalized": image is in model space, output is clamped to [-1, 1]
    """
    image3d, was_2d = _ensure_channel_first(image)
    original_dtype = image3d.dtype
    if not image3d.is_floating_point():
        image3d = image3d.float()

    h, w = _get_image_hw(image3d)
    mode = _resolve_mode(mode)
    band = _resolve_band(band)
    window = int(window)
    intensity = float(intensity)
    mix_alpha = float(mix_alpha)
    input_domain = _normalize_name(input_domain)

    if mode == "dct":
        coeffs = _dct2(image3d)
        rs, cs = _band_slices(h, w, window, band, fft=False)
        trigger = _make_trigger_pattern(
            mode=mode,
            h=h,
            w=w,
            window=window,
            band=band,
            intensity=intensity,
            seed=seed,
            device=image3d.device,
            trigger_style=trigger_style,
        )
        coeffs = coeffs.clone()
        coeffs[..., rs, cs] = coeffs[..., rs, cs] + trigger
        out = _idct2(coeffs)
    else:
        freq = torch.fft.rfft2(image3d, norm="ortho")
        rs, cs = _band_slices(h, w, window, band, fft=True)
        trigger = _make_trigger_pattern(
            mode=mode,
            h=h,
            w=w,
            window=window,
            band=band,
            intensity=intensity,
            seed=seed,
            device=image3d.device,
            trigger_style=trigger_style,
        )
        freq = freq.clone()
        freq[..., rs, cs] = freq[..., rs, cs] + trigger.to(freq.real.dtype).to(freq.dtype)
        out = torch.fft.irfft2(freq, s=(h, w), norm="ortho")

    if mix_alpha < 1.0:
        out = (1.0 - mix_alpha) * image3d + mix_alpha * out

    if input_domain == "raw":
        out = torch.clamp(out, 0.0, 1.0)
    else:
        out = torch.clamp(out, -1.0, 1.0)

    if original_dtype.is_floating_point:
        out = out.to(original_dtype)

    return _restore_shape(out, was_2d)


class PoisonedFrequencyDataset(Dataset):
    """Apply a spectral trigger to a fixed subset of local samples."""

    def __init__(
        self,
        base_dataset,
        poison_rate: float,
        target_label: int,
        frequency_mode: str,
        frequency_band: str,
        frequency_window_size: int,
        frequency_intensity: float,
        mix_alpha: float,
        trigger_style: str,
        seed: int = 42,
        exclude_target_label: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.poison_rate = float(poison_rate)
        self.target_label = int(target_label)
        self.frequency_mode = _resolve_mode(frequency_mode)
        self.frequency_band = _resolve_band(frequency_band)
        self.frequency_window_size = int(frequency_window_size)
        self.frequency_intensity = float(frequency_intensity)
        self.mix_alpha = float(mix_alpha)
        self.trigger_style = _resolve_style(trigger_style)
        self.seed = int(seed)
        self.exclude_target_label = bool(exclude_target_label)

        rng = np.random.default_rng(self.seed)
        self.poison_flags = rng.random(len(base_dataset)) < self.poison_rate

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]

        raw_image = sample.get("img_raw", None)
        use_raw_domain = raw_image is not None

        # Prefer raw image if available so the trigger is injected before
        # normalization. This is important for learnability.
        if use_raw_domain:
            image = raw_image.clone()
        else:
            image = sample["img"].clone()

        label = torch.as_tensor(sample["label"]).long()

        should_poison = bool(self.poison_flags[idx])
        if self.exclude_target_label and int(label.item()) == self.target_label:
            should_poison = False

        if should_poison:
            image = _apply_frequency_trigger(
                image,
                mode=self.frequency_mode,
                band=self.frequency_band,
                window=self.frequency_window_size,
                intensity=self.frequency_intensity,
                seed=self.seed,
                mix_alpha=self.mix_alpha,
                trigger_style=self.trigger_style,
                input_domain="raw" if use_raw_domain else "normalized",
            )
            label = torch.tensor(self.target_label, dtype=torch.long)

        # If we used raw input, return normalized model-space tensor.
        if use_raw_domain:
            image = _normalize_cifar10(image)

        return {"img": image, "label": label}


class TriggeredFrequencyDataset(Dataset):
    """Apply a spectral trigger to every sample for ASR evaluation."""

    def __init__(
        self,
        base_dataset,
        frequency_mode: str,
        frequency_band: str,
        frequency_window_size: int,
        frequency_intensity: float,
        mix_alpha: float,
        trigger_style: str,
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.frequency_mode = _resolve_mode(frequency_mode)
        self.frequency_band = _resolve_band(frequency_band)
        self.frequency_window_size = int(frequency_window_size)
        self.frequency_intensity = float(frequency_intensity)
        self.mix_alpha = float(mix_alpha)
        self.trigger_style = _resolve_style(trigger_style)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]

        raw_image = sample.get("img_raw", None)
        use_raw_domain = raw_image is not None

        if use_raw_domain:
            image = raw_image.clone()
        else:
            image = sample["img"].clone()

        label = torch.as_tensor(sample["label"]).long()

        image = _apply_frequency_trigger(
            image,
            mode=self.frequency_mode,
            band=self.frequency_band,
            window=self.frequency_window_size,
            intensity=self.frequency_intensity,
            seed=self.seed,
            mix_alpha=self.mix_alpha,
            trigger_style=self.trigger_style,
            input_domain="raw" if use_raw_domain else "normalized",
        )

        if use_raw_domain:
            image = _normalize_cifar10(image)

        return {"img": image, "label": label}


def get_poisoned_loader(
    trainloader: DataLoader,
    *,
    poison_rate: float,
    target_label: int,
    frequency_mode: str,
    frequency_band: str,
    frequency_window_size: int,
    frequency_intensity: float,
    mix_alpha: float = 1.0,
    trigger_style: str = "structured",
    seed: int = 42,
    exclude_target_label: bool = True,
) -> DataLoader:
    """Build a poisoned training loader without changing batch shape."""
    poisoned_dataset = PoisonedFrequencyDataset(
        base_dataset=trainloader.dataset,
        poison_rate=poison_rate,
        target_label=target_label,
        frequency_mode=frequency_mode,
        frequency_band=frequency_band,
        frequency_window_size=frequency_window_size,
        frequency_intensity=frequency_intensity,
        mix_alpha=mix_alpha,
        trigger_style=trigger_style,
        seed=seed,
        exclude_target_label=exclude_target_label,
    )

    return DataLoader(
        poisoned_dataset,
        batch_size=trainloader.batch_size or 1,
        shuffle=True,
        num_workers=getattr(trainloader, "num_workers", 0),
        pin_memory=getattr(trainloader, "pin_memory", False),
        drop_last=getattr(trainloader, "drop_last", False),
    )


def get_triggered_loader(
    testloader: DataLoader,
    *,
    frequency_mode: str,
    frequency_band: str,
    frequency_window_size: int,
    frequency_intensity: float,
    mix_alpha: float = 1.0,
    trigger_style: str = "structured",
    seed: int = 42,
) -> DataLoader:
    """Build a triggered test loader for ASR evaluation."""
    triggered_dataset = TriggeredFrequencyDataset(
        base_dataset=testloader.dataset,
        frequency_mode=frequency_mode,
        frequency_band=frequency_band,
        frequency_window_size=frequency_window_size,
        frequency_intensity=frequency_intensity,
        mix_alpha=mix_alpha,
        trigger_style=trigger_style,
        seed=seed,
    )

    return DataLoader(
        triggered_dataset,
        batch_size=testloader.batch_size or 1,
        shuffle=False,
        num_workers=getattr(testloader, "num_workers", 0),
        pin_memory=getattr(testloader, "pin_memory", False),
        drop_last=getattr(testloader, "drop_last", False),
    )


class FrequencyAttack(AttackBase):
    """Frequency-domain backdoor attack implementation."""

    def __init__(self, config: AttackConfig) -> None:
        super().__init__(config)

        extra = config.extra or {}
        self.frequency_mode = _resolve_mode(
            extra.get("frequency_mode", extra.get("mode", "dct"))
        )
        self.frequency_band = _resolve_band(
            extra.get("frequency_band", extra.get("band", "low"))
        )
        self.frequency_window_size = _resolve_window_size(
            extra.get("frequency_window_size", extra.get("window_size", None)),
            config.trigger_size,
        )
        self.frequency_intensity = float(
            extra.get("frequency_intensity", extra.get("intensity", 0.35))
        )
        self.mix_alpha = float(extra.get("mix_alpha", extra.get("alpha", 1.0)))
        self.trigger_style = _resolve_style(
            extra.get("trigger_style", extra.get("style", "structured"))
        )

        print(f">>> [DEBUG] frequency_mode = {self.frequency_mode}")
        print(f">>> [DEBUG] frequency_band = {self.frequency_band}")
        print(f">>> [DEBUG] frequency_window_size = {self.frequency_window_size}")
        print(f">>> [DEBUG] frequency_intensity = {self.frequency_intensity}")
        print(f">>> [DEBUG] mix_alpha = {self.mix_alpha}")
        print(f">>> [DEBUG] trigger_style = {self.trigger_style}")

        if self.frequency_intensity < 0.0:
            raise ValueError("frequency_intensity must be non-negative.")
        if not (0.0 < self.mix_alpha <= 1.0):
            raise ValueError("mix_alpha must be in (0.0, 1.0].")

    def get_malicious_clients(self, total_clients: int, server_round: int = 0) -> set[int]:
        return select_malicious_clients(
            num_clients=total_clients,
            malicious_ratio=self.config.malicious_ratio,
            seed=self.config.seed,
            malicious_mode=self.config.extra.get("malicious_mode", "random"),
            fixed_malicious_clients=self.config.extra.get("fixed_malicious_clients", None),
            server_round=server_round,
        )


    def get_poisoned_loader(self, trainloader: DataLoader) -> DataLoader:
        return get_poisoned_loader(
            trainloader=trainloader,
            poison_rate=self.config.poison_rate,
            target_label=self.config.target_label,
            frequency_mode=self.frequency_mode,
            frequency_band=self.frequency_band,
            frequency_window_size=self.frequency_window_size,
            frequency_intensity=self.frequency_intensity,
            mix_alpha=self.mix_alpha,
            trigger_style=self.trigger_style,
            seed=self.config.seed,
            exclude_target_label=True,
        )

    def get_triggered_loader(self, testloader: DataLoader) -> DataLoader:
        return get_triggered_loader(
            testloader=testloader,
            frequency_mode=self.frequency_mode,
            frequency_band=self.frequency_band,
            frequency_window_size=self.frequency_window_size,
            frequency_intensity=self.frequency_intensity,
            mix_alpha=self.mix_alpha,
            trigger_style=self.trigger_style,
            seed=self.config.seed,
        )


def build_frequency_attack(
    malicious_ratio: float = 0.2,
    poison_rate: float = 0.05,
    target_label: int = 0,
    trigger_size: int = 4,
    seed: int = 42,
    frequency_mode: str = "dct",
    frequency_band: str = "low",
    frequency_window_size: int | None = None,
    frequency_intensity: float = 0.35,
    mix_alpha: float = 1.0,
    trigger_style: str = "structured",
    malicious_mode: str = "random",
    fixed_malicious_clients: list[int] | tuple[int, ...] | None = None,
    dataset_meta=None,
) -> FrequencyAttack:
    """Convenience factory for the frequency attack."""
    resolved_window_size = _resolve_window_size(frequency_window_size, trigger_size)

    config = AttackConfig(
        attack_type="frequency",
        malicious_ratio=malicious_ratio,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=resolved_window_size,
        seed=seed,
        extra={
            "frequency_mode": _resolve_mode(frequency_mode),
            "frequency_band": _resolve_band(frequency_band),
            "frequency_window_size": resolved_window_size,
            "frequency_intensity": float(frequency_intensity),
            "mix_alpha": float(mix_alpha),
            "trigger_style": _resolve_style(trigger_style),
            "malicious_mode": malicious_mode,
            "fixed_malicious_clients": normalize_fixed_malicious_clients(fixed_malicious_clients),
        },
        dataset_meta=dataset_meta,
    )
    return FrequencyAttack(config)