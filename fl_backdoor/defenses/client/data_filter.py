"""Client-side sample filtering based on feature distribution analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .base import ClientDefenseBase, ClientDefenseConfig


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    """Convert raw scores to robust z-scores."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad + 1e-12
    return (values - median) / scale


def _extract_features(model: Any, images: torch.Tensor) -> torch.Tensor:
    """Use the model's feature extractor when available."""
    if hasattr(model, "features"):
        feats = model.features(images)
        return torch.flatten(feats, start_dim=1)

    # Fallback: use logits as features
    logits = model(images)
    return torch.flatten(logits, start_dim=1)


class FeatureDistributionFilterDefense(ClientDefenseBase):
    """Filter suspicious local samples using model feature distances."""

    def __init__(self, config: ClientDefenseConfig) -> None:
        super().__init__(config)

        extra = self.config.extra
        self.filter_ratio = float(extra.get("filter_ratio", 0.1))
        self.min_keep = int(extra.get("min_keep", 16))
        self.scoring_batch_size = int(extra.get("scoring_batch_size", 128))
        self.use_label_centroids = bool(extra.get("use_label_centroids", True))
        self.label_blend_alpha = float(extra.get("label_blend_alpha", 0.5))
        self.min_class_samples = int(extra.get("min_class_samples", 8))

        if not (0.0 <= self.filter_ratio < 1.0):
            raise ValueError("filter_ratio must be in [0.0, 1.0).")
        if self.min_keep < 1:
            raise ValueError("min_keep must be >= 1.")
        if self.scoring_batch_size < 1:
            raise ValueError("scoring_batch_size must be >= 1.")
        if not (0.0 <= self.label_blend_alpha <= 1.0):
            raise ValueError("label_blend_alpha must be in [0.0, 1.0].")
        if self.min_class_samples < 1:
            raise ValueError("min_class_samples must be >= 1.")

    def apply(
        self,
        model: Any,
        trainloader: DataLoader,
        device: Any,
    ) -> tuple[DataLoader, dict[str, Any]]:
        dataset = trainloader.dataset
        total_samples = len(dataset)

        if total_samples == 0 or self.filter_ratio <= 0.0:
            return trainloader, {
                "client-defense-applied": 0,
                "client-defense-filtered-samples": 0,
                "client-defense-kept-samples": total_samples,
                "client-defense-filter-ratio": 0.0,
                "client-defense-mean-score": 0.0,
                "client-defense-max-score": 0.0,
                "client-defense-label-aware": int(self.use_label_centroids),
            }

        scan_batch_size = min(self.scoring_batch_size, total_samples)

        scan_loader = DataLoader(
            dataset,
            batch_size=scan_batch_size,
            shuffle=False,
            num_workers=getattr(trainloader, "num_workers", 0),
            pin_memory=getattr(trainloader, "pin_memory", False),
            drop_last=False,
        )

        model.eval()
        feature_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in scan_loader:
                images = batch["img"].to(device)
                labels = torch.as_tensor(batch["label"]).long().cpu()

                feats = _extract_features(model, images).detach().cpu()
                feature_chunks.append(feats)
                label_chunks.append(labels)

        if not feature_chunks:
            return trainloader, {
                "client-defense-applied": 0,
                "client-defense-filtered-samples": 0,
                "client-defense-kept-samples": total_samples,
                "client-defense-filter-ratio": 0.0,
                "client-defense-mean-score": 0.0,
                "client-defense-max-score": 0.0,
                "client-defense-label-aware": int(self.use_label_centroids),
            }

        features = torch.cat(feature_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)

        # ------------------------
        # Global feature distance
        # ------------------------
        global_center = features.mean(dim=0, keepdim=True)
        global_dist = torch.linalg.vector_norm(features - global_center, dim=1).numpy()

        combined_dist = global_dist.copy()

        # ------------------------
        # Optional label-aware refinement
        # ------------------------
        if self.use_label_centroids:
            label_dist = global_dist.copy()
            unique_labels = torch.unique(labels)

            for lbl in unique_labels:
                mask = labels == lbl
                count = int(mask.sum().item())

                if count < self.min_class_samples:
                    continue

                class_features = features[mask]
                class_center = class_features.mean(dim=0, keepdim=True)
                class_dist = torch.linalg.vector_norm(
                    class_features - class_center,
                    dim=1,
                ).numpy()
                label_dist[mask.numpy()] = class_dist

            combined_dist = (
                self.label_blend_alpha * global_dist
                + (1.0 - self.label_blend_alpha) * label_dist
            )

        scores = _robust_standardize(combined_dist)

        keep_count = max(self.min_keep, int(round(total_samples * (1.0 - self.filter_ratio))))
        keep_count = min(keep_count, total_samples)

        keep_indices = np.argsort(scores)[:keep_count]
        keep_indices = sorted(int(i) for i in keep_indices)

        filtered_dataset = Subset(dataset, keep_indices)

        filtered_loader = DataLoader(
            filtered_dataset,
            batch_size=trainloader.batch_size or 1,
            shuffle=True,
            num_workers=getattr(trainloader, "num_workers", 0),
            pin_memory=getattr(trainloader, "pin_memory", False),
            drop_last=getattr(trainloader, "drop_last", False),
        )

        stats = {
            "client-defense-applied": 1,
            "client-defense-filtered-samples": int(total_samples - keep_count),
            "client-defense-kept-samples": int(keep_count),
            "client-defense-filter-ratio": float((total_samples - keep_count) / total_samples),
            "client-defense-mean-score": float(np.mean(scores)) if len(scores) else 0.0,
            "client-defense-max-score": float(np.max(scores)) if len(scores) else 0.0,
            "client-defense-label-aware": int(self.use_label_centroids),
            "client-defense-label-blend-alpha": float(self.label_blend_alpha),
        }

        return filtered_loader, stats