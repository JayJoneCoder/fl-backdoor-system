"""Full Combination Backdoor Attack (FCBA) implementation.

FCBA divides a global trigger into m sub-blocks and generates all non-empty
proper subsets (2^m - 2) as local triggers. Each malicious client receives
a unique local trigger (combination of sub-blocks). The global trigger is
used for ASR evaluation.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import AttackBase, AttackConfig
from .selection import normalize_fixed_malicious_clients, select_malicious_clients


class FCBAPoisonedDataset(Dataset):
    """Dataset that poisons each sample with the client-specific local trigger."""

    def __init__(
        self,
        base_dataset,
        poison_rate: float,
        target_label: int,
        trigger_config: dict,   # contains 'mask', 'value', 'location'
        seed: int = 42,
        exclude_target_label: bool = True,
        image_size: int = 32,
    ) -> None:
        self.base_dataset = base_dataset
        self.poison_rate = float(poison_rate)
        self.target_label = int(target_label)
        self.trigger_mask_small = trigger_config["mask"]   # shape (C, H_small, W_small)
        self.trigger_value = trigger_config["value"]
        self.trigger_location = trigger_config["location"]  # (row, col) top-left corner
        self.exclude_target_label = bool(exclude_target_label)
        self.image_size = image_size

        rng = np.random.default_rng(seed)
        self.poison_flags = rng.random(len(base_dataset)) < self.poison_rate

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        image = sample["img"].clone()
        label = torch.as_tensor(sample["label"]).long()

        should_poison = bool(self.poison_flags[idx])
        if self.exclude_target_label and int(label.item()) == self.target_label:
            should_poison = False

        if should_poison:
            # Create full-size mask
            full_mask = torch.zeros(3, self.image_size, self.image_size)
            r, c = self.trigger_location
            h, w = self.trigger_mask_small.shape[1], self.trigger_mask_small.shape[2]
            # Ensure the mask fits within image boundaries
            r_end = min(r + h, self.image_size)
            c_end = min(c + w, self.image_size)
            full_mask[:, r:r_end, c:c_end] = self.trigger_mask_small[:, :r_end-r, :c_end-c]
            image = image + self.trigger_value * full_mask.to(image.device)
            image = torch.clamp(image, 0.0, 1.0)
            label = torch.tensor(self.target_label, dtype=torch.long)

        return {"img": image, "label": label}


class FCBATriggeredDataset(Dataset):
    """Apply the FULL global trigger for ASR evaluation."""

    def __init__(self, base_dataset, global_mask_small: torch.Tensor, value: float,
                 location: tuple[int, int], image_size: int = 32):
        self.base_dataset = base_dataset
        self.value = value
        self.location = location
        self.image_size = image_size
        # Create full-size mask once
        full_mask = torch.zeros(3, image_size, image_size)
        r, c = location
        h, w = global_mask_small.shape[1], global_mask_small.shape[2]
        r_end = min(r + h, image_size)
        c_end = min(c + w, image_size)
        full_mask[:, r:r_end, c:c_end] = global_mask_small[:, :r_end-r, :c_end-c]
        self.full_mask = full_mask

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        image = sample["img"].clone()
        label = torch.as_tensor(sample["label"]).long()
        image = image + self.value * self.full_mask.to(image.device)
        image = torch.clamp(image, 0.0, 1.0)
        return {"img": image, "label": label}


class FCBAAttack(AttackBase):
    """Full Combination Backdoor Attack.

    Parameters in extra:
        num_sub_blocks (int): number of sub-blocks to split the trigger (m >= 2).
        sub_block_size (int): size of each sub-block (square). If not provided,
            automatically computed from trigger_size and num_sub_blocks.
        global_trigger_value (float): pixel value for the trigger (default 1.0).
        split_strategy (str): 'grid' or 'random' for sub-block positions.
        global_trigger_location (tuple): (row, col) of top-left corner of global trigger.
            Default is bottom-right corner (image_size - trigger_size).
        image_size (int): size of input images (default 32 for CIFAR-10).
    """

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.trigger_size = config.trigger_size
        self.num_sub_blocks = config.extra.get("num_sub_blocks", 4)
        self.sub_block_size = config.extra.get("sub_block_size", None)
        self.global_trigger_value = config.extra.get("global_trigger_value", 1.0)
        self.split_strategy = config.extra.get("split_strategy", "grid")
        self.image_size = config.extra.get("image_size", 32)  # assume CIFAR-10 size
        # Default location: bottom-right corner
        default_loc = (self.image_size - self.trigger_size, self.image_size - self.trigger_size)
        self.global_trigger_location = config.extra.get("global_trigger_location", default_loc)

        if self.num_sub_blocks < 2:
            raise ValueError("num_sub_blocks must be >= 2 for FCBA.")

        # Compute sub-block size if not given
        if self.sub_block_size is None:
            n = int(np.sqrt(self.num_sub_blocks))
            if n * n != self.num_sub_blocks:
                raise ValueError(
                    f"num_sub_blocks={self.num_sub_blocks} is not a perfect square. "
                    "Please specify sub_block_size manually."
                )
            self.sub_block_size = self.trigger_size // n
            if self.sub_block_size <= 0:
                raise ValueError("Trigger too small to split into given number of sub-blocks.")

        # Generate the global trigger mask (small)
        self.global_mask_small = self._generate_global_mask()

        # Generate list of sub-block masks (each is a binary tensor of shape (C, H, W))
        self.sub_block_masks = self._generate_sub_block_masks()

        # Generate all non-empty proper subsets of sub-blocks
        self.local_trigger_masks = self._generate_local_trigger_masks()

        # Number of local triggers (malicious clients needed)
        self.num_local_triggers = len(self.local_trigger_masks)

        print(f"[FCBA] num_sub_blocks={self.num_sub_blocks}, "
              f"sub_block_size={self.sub_block_size}, "
              f"num_local_triggers={self.num_local_triggers}")

    def _generate_global_mask(self) -> torch.Tensor:
        """Create the full global trigger mask (binary, small)."""
        mask = torch.zeros(3, self.trigger_size, self.trigger_size)
        mask[:, :, :] = 1.0
        return mask

    def _generate_sub_block_masks(self) -> list[torch.Tensor]:
        """Generate masks for each sub-block (binary, same size as global trigger area)."""
        masks = []
        if self.split_strategy == "grid":
            n = self.trigger_size // self.sub_block_size
            for i in range(n):
                for j in range(n):
                    mask = torch.zeros(3, self.trigger_size, self.trigger_size)
                    row_start = i * self.sub_block_size
                    row_end = row_start + self.sub_block_size
                    col_start = j * self.sub_block_size
                    col_end = col_start + self.sub_block_size
                    mask[:, row_start:row_end, col_start:col_end] = 1.0
                    masks.append(mask)
        elif self.split_strategy == "random":
            rng = self.rng()
            n = self.trigger_size // self.sub_block_size
            positions = [(i, j) for i in range(n) for j in range(n)]
            rng.shuffle(positions)
            for (i, j) in positions[:self.num_sub_blocks]:
                mask = torch.zeros(3, self.trigger_size, self.trigger_size)
                row_start = i * self.sub_block_size
                row_end = row_start + self.sub_block_size
                col_start = j * self.sub_block_size
                col_end = col_start + self.sub_block_size
                mask[:, row_start:row_end, col_start:col_end] = 1.0
                masks.append(mask)
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy}")
        return masks

    def _generate_local_trigger_masks(self) -> list[torch.Tensor]:
        """Generate all non-empty proper subsets of sub-block masks."""
        indices = list(range(len(self.sub_block_masks)))
        local_masks = []
        for r in range(1, len(indices)):
            for combo in itertools.combinations(indices, r):
                combined = torch.zeros_like(self.sub_block_masks[0])
                for idx in combo:
                    combined = torch.max(combined, self.sub_block_masks[idx])
                local_masks.append(combined)
        return local_masks

    def get_local_trigger_config(self, client_id: int) -> dict:
        """Return trigger config (mask, value, location) for the given client."""
        idx = client_id % self.num_local_triggers
        mask = self.local_trigger_masks[idx]
        return {
            "mask": mask,
            "value": self.global_trigger_value,
            "location": self.global_trigger_location,
        }

    def get_global_trigger_config(self) -> dict:
        return {
            "mask": self.global_mask_small,
            "value": self.global_trigger_value,
            "location": self.global_trigger_location,
        }

    def get_malicious_clients(self, total_clients: int, server_round: int = 0) -> set[int]:
        mode = self.config.extra.get("malicious_mode", "random")
        fixed_clients = self.config.extra.get("fixed_malicious_clients", None)

        malicious = select_malicious_clients(
            num_clients=total_clients,
            malicious_ratio=self.config.malicious_ratio,
            seed=self.config.seed,
            malicious_mode=mode,
            fixed_malicious_clients=fixed_clients,
            server_round=server_round,
        )
        print(f"[Attack][FCBA] round={server_round}, mode={mode}, selected={sorted(malicious)}")
        return malicious

    def get_poisoned_loader(self, trainloader: DataLoader) -> DataLoader:
        if not hasattr(self, "_current_client_id"):
            raise RuntimeError(
                "FCBA requires _current_client_id to be set before calling get_poisoned_loader."
            )
        client_id = self._current_client_id
        local_config = self.get_local_trigger_config(client_id)

        poisoned_dataset = FCBAPoisonedDataset(
            base_dataset=trainloader.dataset,
            poison_rate=self.config.poison_rate,
            target_label=self.config.target_label,
            trigger_config=local_config,
            seed=self.config.seed,
            exclude_target_label=True,
            image_size=self.image_size,
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
        global_config = self.get_global_trigger_config()
        return DataLoader(
            FCBATriggeredDataset(
                testloader.dataset,
                global_config["mask"],
                global_config["value"],
                global_config["location"],
                image_size=self.image_size,
            ),
            batch_size=testloader.batch_size or 1,
            shuffle=False,
            num_workers=getattr(testloader, "num_workers", 0),
            pin_memory=getattr(testloader, "pin_memory", False),
            drop_last=getattr(testloader, "drop_last", False),
        )


def build_fcba_attack(
    malicious_ratio: float = 0.2,
    poison_rate: float = 0.05,
    target_label: int = 0,
    trigger_size: int = 4,
    seed: int = 42,
    malicious_mode: str = "random",
    fixed_malicious_clients: list[int] | tuple[int, ...] | None = None,
    num_sub_blocks: int = 4,
    sub_block_size: int | None = None,
    global_trigger_value: float = 1.0,
    split_strategy: str = "grid",
    global_trigger_location: tuple[int, int] | None = None,
    image_size: int = 32,
    dataset_meta=None,
) -> FCBAAttack:
    """Factory for FCBA attack."""
    extra = {
        "malicious_mode": malicious_mode,
        "fixed_malicious_clients": normalize_fixed_malicious_clients(fixed_malicious_clients),
        "num_sub_blocks": num_sub_blocks,
        "sub_block_size": sub_block_size,
        "global_trigger_value": global_trigger_value,
        "split_strategy": split_strategy,
        "image_size": image_size,
    }
    if global_trigger_location is not None:
        extra["global_trigger_location"] = global_trigger_location
    config = AttackConfig(
        attack_type="fcba",
        malicious_ratio=malicious_ratio,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=trigger_size,
        seed=seed,
        extra=extra,
        dataset_meta=dataset_meta,
    )
    return FCBAAttack(config)