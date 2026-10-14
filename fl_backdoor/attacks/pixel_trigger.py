"""Dataset-aware pixel triggers shared by DBA and FCBA.

Trigger values are expressed in raw [0, 1] pixel space. Dataset samples contain
normalized model inputs, so only masked pixels are converted between spaces.
Without dataset metadata, legacy callers are assumed to supply raw pixels.
"""

from math import isfinite
from numbers import Integral

import torch


def get_image_shape(dataset_meta=None, image_size: int = 32) -> tuple[int, int, int]:
    """Use dataset metadata, with a legacy RGB square-image fallback."""
    shape = dataset_meta.input_shape if dataset_meta is not None else (3, image_size, image_size)
    if (not isinstance(shape, (tuple, list)) or len(shape) != 3
            or any(isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
                   for size in shape)):
        raise ValueError("input_shape must contain positive integer channels, height and width.")
    shape = tuple(int(size) for size in shape)
    normalization_stats(dataset_meta, shape[0])
    return shape


def normalization_stats(dataset_meta, channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate per-channel normalization; no metadata means raw input."""
    if dataset_meta is None:
        return (0.0,) * channels, (1.0,) * channels
    mean = tuple(float(value) for value in dataset_meta.mean)
    std = tuple(float(value) for value in dataset_meta.std)
    if len(mean) != channels or len(std) != channels:
        raise ValueError("normalization mean/std must match the input channels.")
    if not all(isfinite(value) for value in mean):
        raise ValueError("normalization mean must be finite.")
    if not all(isfinite(value) and value > 0 for value in std):
        raise ValueError("normalization std must be finite and positive.")
    return mean, std


def validate_trigger_value(value: float) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("global_trigger_value must be finite and in raw pixel range [0, 1].")
    return value


def resolve_trigger_location(
    input_shape: tuple[int, int, int],
    trigger_shape: tuple[int, int],
    location=None,
) -> tuple[int, int]:
    """Place automatic triggers at bottom-right; reject manual clipping."""
    _, height, width = input_shape
    trigger_height, trigger_width = trigger_shape
    if trigger_height <= 0 or trigger_width <= 0 or trigger_height > height or trigger_width > width:
        raise ValueError(f"Trigger size {trigger_shape} does not fit image shape {input_shape}.")
    if location is None:
        return height - trigger_height, width - trigger_width
    if (not isinstance(location, (tuple, list)) or len(location) != 2
            or any(isinstance(value, bool) or not isinstance(value, Integral) for value in location)):
        raise ValueError("global_trigger_location must contain two integer coordinates (row, col).")
    row, col = (int(value) for value in location)
    if row < 0 or col < 0 or row + trigger_height > height or col + trigger_width > width:
        raise ValueError(
            f"Trigger location {(row, col)} with size {trigger_shape} is out of bounds "
            f"for image shape {input_shape}; leave location empty for automatic placement."
        )
    return row, col


def apply_pixel_trigger(
    image: torch.Tensor,
    mask: torch.Tensor,
    value: float,
    location: tuple[int, int],
    *,
    dataset_meta=None,
    additive: bool = False,
) -> torch.Tensor:
    """Fill (DBA) or add (FCBA) a trigger without touching unmasked pixels."""
    if image.ndim != 3 or not image.is_floating_point():
        raise ValueError("Pixel triggers require a floating-point image with shape (C, H, W).")
    channels, _, _ = image.shape
    if dataset_meta is not None and tuple(image.shape) != get_image_shape(dataset_meta):
        raise ValueError(
            f"Sample shape {tuple(image.shape)} does not match dataset shape {dataset_meta.input_shape}."
        )
    if mask.ndim != 3 or mask.shape[0] != channels:
        raise ValueError("Trigger mask shape must match the image channels.")
    trigger_height, trigger_width = mask.shape[-2:]
    row, col = resolve_trigger_location(tuple(image.shape), (trigger_height, trigger_width), location)
    value = validate_trigger_value(value)
    mean_values, std_values = normalization_stats(dataset_meta, channels)
    mean = image.new_tensor(mean_values).view(channels, 1, 1)
    std = image.new_tensor(std_values).view(channels, 1, 1)
    output = image.clone()
    region = output[:, row:row + trigger_height, col:col + trigger_width]
    if additive:
        # Preserve FCBA's additive semantics, but clamp in RAW pixel space.
        changed = ((region * std + mean + value).clamp(0.0, 1.0) - mean) / std
    else:
        changed = (value - mean) / std
    active = mask.to(device=image.device, dtype=torch.bool)
    output[:, row:row + trigger_height, col:col + trigger_width] = torch.where(active, changed, region)
    return output


def apply_square_trigger(
    image: torch.Tensor,
    size: int,
    value: float,
    location: tuple[int, int],
    *,
    dataset_meta=None,
) -> torch.Tensor:
    mask = torch.ones((image.shape[0], size, size), dtype=torch.bool, device=image.device)
    return apply_pixel_trigger(image, mask, value, location, dataset_meta=dataset_meta)