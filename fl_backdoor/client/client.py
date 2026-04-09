"""fl_backdoor: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_backdoor.task import Net, load_data
from fl_backdoor.task import test as test_fn
from fl_backdoor.task import train as train_fn
from fl_backdoor.attacks import build_attack
from fl_backdoor.defenses import build_defense_pipeline_from_run_config
from fl_backdoor.attacks.selection import normalize_fixed_malicious_clients
from fl_backdoor.dataset import get_dataset

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load model
    # Get dataset meta for model instantiation
    dataset_name = context.run_config.get("dataset", "cifar10")
    dataset_meta = get_dataset(dataset_name).meta

    model = Net(
        input_shape=dataset_meta.input_shape,
        num_classes=dataset_meta.num_classes
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    trainloader, _ = load_data(partition_id, num_partitions, batch_size, dataset_name)
    # 检查第一个 batch 的形状
    sample_batch = next(iter(trainloader))
    print(f"[Client {partition_id}] dataset={dataset_name}, image shape={sample_batch['img'].shape}")

    server_round = int(msg.content["config"].get("server-round", 0))

    # Attack config
    attack_type = str(
        context.run_config.get("attack-type", context.run_config.get("attack", "badnets"))
    ).lower().strip()

    malicious_ratio = float(context.run_config.get("malicious-ratio", 0.2))
    poison_rate = float(context.run_config.get("poison-rate", 0.05))
    target_label = int(context.run_config.get("target-label", 0))
    trigger_size = int(context.run_config.get("trigger-size", 4))
    seed = int(context.run_config.get("seed", 42))

    grid_size = context.run_config.get("wanet-grid-size", None)
    noise_scale = float(context.run_config.get("wanet-noise", 0.05))

    frequency_mode = str(context.run_config.get("frequency-mode", "dct"))
    frequency_band = str(context.run_config.get("frequency-band", "low"))
    frequency_window_size = context.run_config.get("frequency-window-size", None)
    frequency_intensity = float(context.run_config.get("frequency-intensity", 0.10))
    frequency_mix_alpha = float(context.run_config.get("frequency-mix-alpha", 1.0))

    malicious_mode = str(context.run_config.get("attack-malicious-mode", "random")).lower().strip()

    fixed_malicious_clients_raw = context.run_config.get("attack-fixed-clients", None)
    fixed_malicious_clients = (
        list(normalize_fixed_malicious_clients(fixed_malicious_clients_raw))
        if fixed_malicious_clients_raw is not None
        else None
    )

    # DBA specific parameters
    dba_num_sub_patterns = int(context.run_config.get("dba-num-sub-patterns", 4))
    dba_sub_pattern_size = context.run_config.get("dba-sub-pattern-size", None)
    if dba_sub_pattern_size is not None:
        dba_sub_pattern_size = int(dba_sub_pattern_size)
    dba_global_trigger_value = float(context.run_config.get("dba-global-trigger-value", 1.0))
    dba_split_strategy = str(context.run_config.get("dba-split-strategy", "grid"))

    # FCBA specific parameters
    fcba_num_sub_blocks = int(context.run_config.get("fcba-num-sub-blocks", 4))
    fcba_sub_block_size = context.run_config.get("fcba-sub-block-size", None)
    if fcba_sub_block_size is not None:
        fcba_sub_block_size = int(fcba_sub_block_size)
    fcba_global_trigger_value = float(context.run_config.get("fcba-global-trigger-value", 1.0))
    fcba_split_strategy = str(context.run_config.get("fcba-split-strategy", "grid"))
    fcba_global_trigger_location = context.run_config.get("fcba-global-trigger-location", None)
    # If needed, parse tuple from string like "[28,28]"

    print(f">>> [DEBUG] attack_type = {attack_type}")

    # Get dataset metadata for attack construction
    dataset_name = context.run_config.get("dataset", "cifar10")
    dataset_meta = get_dataset(dataset_name).meta

    attack = build_attack(
        attack_type=attack_type,
        dataset_meta=dataset_meta,  # <--- 新增参数
        malicious_ratio=malicious_ratio,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=trigger_size,
        seed=seed,
        malicious_mode=malicious_mode,
        fixed_malicious_clients=fixed_malicious_clients,
        grid_size=None if grid_size is None else int(grid_size),
        noise_scale=noise_scale,
        frequency_mode=frequency_mode,
        frequency_band=frequency_band,
        frequency_window_size=None if frequency_window_size is None else int(frequency_window_size),
        frequency_intensity=frequency_intensity,
        frequency_mix_alpha=frequency_mix_alpha,
        # DBA parameters
        dba_num_sub_patterns=dba_num_sub_patterns,
        dba_sub_pattern_size=dba_sub_pattern_size,
        dba_global_trigger_value=dba_global_trigger_value,
        dba_split_strategy=dba_split_strategy,
        # FCBA parameters
        fcba_num_sub_blocks=fcba_num_sub_blocks,
        fcba_sub_block_size=fcba_sub_block_size,
        fcba_global_trigger_value=fcba_global_trigger_value,
        fcba_split_strategy=fcba_split_strategy,
        fcba_global_trigger_location=fcba_global_trigger_location,
    )

    # For attacks that need client ID (DBA, FCBA)
    if attack.name in ("dba", "fcba"):
        attack._current_client_id = partition_id

    is_malicious = attack.is_malicious_client(
        partition_id,
        num_partitions,
        server_round=server_round,
    )

    if is_malicious:
        print(f"[Client {partition_id}] MALICIOUS CLIENT")
        trainloader = attack.get_poisoned_loader(trainloader)
    else:
        print(f"[Client {partition_id}] benign client")

    # ------------------------
    # Build Pipeline
    # ------------------------
    pipeline = build_defense_pipeline_from_run_config(
        context.run_config,
        seed=seed,
    )

    client_defense = pipeline.build_client_defense()

    print(f">>> [DEBUG] client_defense = {client_defense}")

    trainloader, defense_stats = client_defense.apply(
        model=model,
        trainloader=trainloader,
        device=device,
    )

    print(f">>> [DEBUG] client defense applied: {defense_stats}")

    # Train
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Return
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "client_id": int(partition_id),
        "server_round": int(server_round),
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "is_malicious": int(is_malicious),
        **defense_stats,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    dataset_name = context.run_config.get("dataset", "cifar10")
    dataset_meta = get_dataset(dataset_name).meta

    model = Net(
        input_shape=dataset_meta.input_shape,
        num_classes=dataset_meta.num_classes
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, dataset_name)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
