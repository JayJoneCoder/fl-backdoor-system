"""fl_backdoor: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_backdoor.task import Net, load_data
from fl_backdoor.task import test as test_fn
from fl_backdoor.task import train as train_fn
from fl_backdoor.attacks import build_attack
from fl_backdoor.defenses.pipeline import build_defense_pipeline
from fl_backdoor.attacks.selection import normalize_fixed_malicious_clients
from fl_backdoor.dataset import get_dataset
from fl_backdoor.config import ExperimentConfig

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    cfg = ExperimentConfig.from_run_config(context.run_config)
    dataset_meta = get_dataset(cfg.dataset).meta

    model = Net(
        input_shape=dataset_meta.input_shape,
        num_classes=dataset_meta.num_classes
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions, cfg.batch_size, cfg.dataset)

    server_round = int(msg.content["config"].get("server-round", 0))

    # 构建攻击
    attack = build_attack(
        dataset_meta=dataset_meta,
        **cfg.to_attack_kwargs()
    )

    if attack.name in ("dba", "fcba"):
        attack._current_client_id = partition_id

    is_malicious = attack.is_malicious_client(
        partition_id, num_partitions, server_round=server_round
    )

    if is_malicious:
        print(f"[Client {partition_id}] MALICIOUS CLIENT")
        trainloader = attack.get_poisoned_loader(trainloader)
    else:
        print(f"[Client {partition_id}] benign client")

    # 构建防御流水线（客户端部分）
    pipeline_kwargs = cfg.get_defense_pipeline_kwargs()
    pipeline = build_defense_pipeline(**pipeline_kwargs)
    client_defense = pipeline.build_client_defense()
    print(f">>> [DEBUG] client_defense = {client_defense}")

    trainloader, defense_stats = client_defense.apply(model, trainloader, device)

    train_loss = train_fn(
        model, trainloader, cfg.local_epochs,
        msg.content["config"]["lr"], device
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
