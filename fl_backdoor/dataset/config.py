"""Dataset metadata configuration."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetMeta:
    """Metadata for a dataset."""
    name: str
    num_classes: int
    input_shape: Tuple[int, int, int]  # (C, H, W)
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    hf_dataset_path: str                # HuggingFace dataset identifier
    hf_subset: str | None = None        # optional subset (e.g., "train")