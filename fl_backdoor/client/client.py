"""fl_backdoor: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_backdoor.task import Net, load_data
from fl_backdoor.task import test as test_fn
from fl_backdoor.task import train as train_fn
from fl_backdoor.attacks import build_attack
from fl_backdoor.defenses import build_defense_pipeline_from_run_config

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load model
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Attack config
    attack_type = str(
        context.run_config.get("attack-type", context.run_config.get("attack", "badnets"))
    ).lower()
    malicious_ratio = float(context.run_config.get("malicious-ratio", 0.2))
    poison_rate = float(context.run_config.get("poison-rate", 0.05))
    target_label = int(context.run_config.get("target-label", 0))
    trigger_size = int(context.run_config.get("trigger-size", 4))
    seed = int(context.run_config.get("seed", 42))
    grid_size = context.run_config.get("wanet-grid-size", None)
    noise_scale = float(context.run_config.get("wanet-noise", 0.05))
    print(f">>> [DEBUG] attack_type = {attack_type}")
    
    attack = build_attack(
        attack_type=attack_type,
        malicious_ratio=malicious_ratio,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=trigger_size,
        seed=seed,
        grid_size=None if grid_size is None else int(grid_size),
        noise_scale=noise_scale,
    )

    is_malicious = attack.is_malicious_client(
        partition_id,
        num_partitions,
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
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

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
