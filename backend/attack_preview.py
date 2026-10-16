"""Read-only trigger previews using the same image transforms as experiments."""

import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from threading import RLock

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from backend.config_manager import CONFIG_SCHEMA
from backend.config_validation import normalize_values, parse_integer
from fl_backdoor.attacks import build_attack
from fl_backdoor.attacks.dba import DBAPoisonedDataset
from fl_backdoor.attacks.fcba import FCBAPoisonedDataset
from fl_backdoor.config import ExperimentConfig
from fl_backdoor.dataset import get_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_LOCK = RLock()
_CIFAR_LABELS = ("飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车")


@lru_cache(maxsize=2)
def _load_training_dataset(name: str, root: str):
    # Never download a dataset or start training as a side effect of opening UI.
    dataset_type = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[name]
    try:
        return dataset_type(root=root, train=True, download=False, transform=None)
    except (RuntimeError, FileNotFoundError) as exc:
        raise FileNotFoundError("本地未找到完整数据集，请先准备或下载所选数据集，再刷新预览。") from exc


def _image_url(model_image: torch.Tensor, meta) -> str:
    mean = model_image.new_tensor(meta.mean).view(-1, 1, 1)
    std = model_image.new_tensor(meta.std).view(-1, 1, 1)
    pixels = ((model_image.detach().cpu() * std + mean).clamp(0, 1) * 255).round().byte()
    array = pixels[0].numpy() if pixels.shape[0] == 1 else pixels.permute(1, 2, 0).numpy()
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def preview_attack(request: dict) -> dict:
    """Force a trigger onto one selected training image, without poison sampling.

    Distributed attacks return both the client's training pattern and the full
    ASR pattern. In particular, rate=0 and target-class images are still visualized.
    This is deliberately NOT a prediction of training-time poison selection.
    """
    values = request.get("config", {})
    if not isinstance(values, dict):
        raise ValueError("预览配置必须是对象。")
    keys = {key for key, field in CONFIG_SCHEMA.items() if field["group"] == "attack"}
    keys.update({"attack", "dataset", "seed", "target-label", "trigger-size", "num-clients"})
    selected = {key: value for key, value in values.items() if key in keys}
    try:
        normalized = normalize_values(selected, CONFIG_SCHEMA)
        cfg = ExperimentConfig.from_run_config(normalized)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("预览参数不合法，请检查数据集、种子及攻击参数的类型和范围。") from exc

    if cfg.dataset not in {"mnist", "cifar10"}:
        raise ValueError("预览仅支持 MNIST 和 CIFAR-10 数据集。")
    dataset = get_dataset(cfg.dataset)
    if not 0 <= cfg.target_label < dataset.meta.num_classes:
        raise ValueError("目标标签必须在 0 到 9 之间。")
    try:
        index = parse_integer(request.get("index", 0), "图片编号")
        client_id = parse_integer(request.get("client_id", 0), "客户端 ID")
    except ValueError as exc:
        raise ValueError("图片编号和客户端 ID 必须是非负整数。") from exc
    scope = request.get("scope", "global")
    if scope not in {"global", "local"}:
        raise ValueError("触发器展示方式只能是完整触发器或客户端局部触发器。")
    if cfg.num_clients < 1 or client_id >= cfg.num_clients:
        raise ValueError("客户端 ID 必须小于客户端总数，且不能为负数。")
    if cfg.trigger_size > min(dataset.meta.input_shape[1:]):
        raise ValueError("触发器尺寸不能超过所选数据集的图像尺寸。")
    # FCBA materializes 2**m-2 masks. Bound preview work for malformed drafts.
    if cfg.attack == "fcba" and cfg.fcba_num_sub_blocks > 10:
        raise ValueError("预览最多支持 10 个 FCBA 子块，请减小子块数量后重试。")

    root = Path(os.environ.get("FL_BACKDOOR_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
    with _DATA_LOCK:
        training = _load_training_dataset(cfg.dataset, str(root))
    if index >= len(training):
        raise ValueError(f"图片编号超出范围，请选择第 1 到 {len(training)} 张图片。")
    image, label = training[index]
    sample = dataset._apply_transforms((image, label))
    try:
        attack = build_attack(dataset_meta=dataset.meta, **cfg.to_attack_kwargs())
        loader = DataLoader([sample], batch_size=1, shuffle=False, num_workers=0)
        local_dataset = None
        local_details = {}
        # Use the actual training wrappers, overriding ONLY poison selection.
        # The client ID still selects the same pattern/combination as training.
        if cfg.attack == "dba":
            local_dataset = DBAPoisonedDataset(
                loader.dataset, poison_rate=1.0, target_label=cfg.target_label,
                trigger_config=attack.get_sub_trigger_config(client_id), seed=cfg.seed,
                exclude_target_label=False, dataset_meta=dataset.meta,
            )
            local_details = {
                "local_trigger_index": client_id % attack.num_sub_patterns,
                "local_trigger_count": attack.num_sub_patterns,
                "local_sub_blocks": 1,
            }
        elif cfg.attack == "fcba":
            trigger = attack.get_local_trigger_config(client_id)
            local_dataset = FCBAPoisonedDataset(
                loader.dataset, poison_rate=1.0, target_label=cfg.target_label,
                trigger_config=trigger, seed=cfg.seed, exclude_target_label=False,
                image_size=attack.image_size, dataset_meta=dataset.meta,
            )
            local_details = {
                "local_trigger_index": client_id % attack.num_local_triggers,
                "local_trigger_count": attack.num_local_triggers,
                "local_sub_blocks": int(trigger["mask"][0].count_nonzero()) // attack.sub_block_size ** 2,
            }
        # Access the dataset directly: no shuffled iteration and no global RNG use.
        with torch.no_grad():
            global_image = attack.get_triggered_loader(loader).dataset[0]["img"]
            local_image = local_dataset[0]["img"] if local_dataset is not None else None
            poisoned = local_image if scope == "local" and local_image is not None else global_image
    except (ValueError, TypeError, RuntimeError, IndexError) as exc:
        raise ValueError("无法生成后门预览，请检查触发器尺寸、位置、子块划分及攻击参数。") from exc

    def label_text(value):
        return str(value) if cfg.dataset == "mnist" else f"{value}（{_CIFAR_LABELS[value]}）"

    def changed_pixels(image):
        return int((image - sample["img"]).abs().gt(1e-6).any(dim=0).sum())

    result = {
        "dataset": cfg.dataset,
        "attack": cfg.attack,
        "index": index,
        "total": len(training),
        "label": int(label),
        "label_text": label_text(int(label)),
        "target_label": cfg.target_label,
        "target_label_text": label_text(cfg.target_label),
        "scope": scope,
        "client_id": client_id,
        "clean_image": _image_url(sample["img"], dataset.meta),
        "poisoned_image": _image_url(poisoned, dataset.meta),
        "changed_pixels": changed_pixels(poisoned),
        "notice": "效果示意：强制植入触发器，不代表该图片在实际训练中被选中投毒；不保存配置、不修改数据集。",
    }
    if local_image is not None:
        result.update({
            "local_image": _image_url(local_image, dataset.meta),
            "global_image": _image_url(global_image, dataset.meta),
            "local_changed_pixels": changed_pixels(local_image),
            "global_changed_pixels": changed_pixels(global_image),
            **local_details,
        })
    return result