"""fl_backdoor: A Flower / PyTorch app."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_backdoor.attacks import build_attack
from fl_backdoor.defenses.pipeline import build_defense_pipeline
from fl_backdoor.task import Net, evaluation_mode, load_centralized_dataset, test
from fl_backdoor.utils.logger import CSVLogger
from fl_backdoor.dataset import get_dataset
from fl_backdoor.config import ExperimentConfig
from fl_backdoor.utils.randomness import seed_everything

from pathlib import Path
import os
from fl_backdoor.utils.run_artifacts import ensure_experiment, save_final_model, write_run_metadata

# Create ServerApp
app = ServerApp()


def build_strategy(cfg: ExperimentConfig) -> FedAvg:
    """Do not inherit Flower's minimum-two-nodes default for a one-node run."""
    return FedAvg(fraction_train=1.0, fraction_evaluate=cfg.fraction_evaluate,
                  min_train_nodes=cfg.num_clients, min_evaluate_nodes=1,
                  min_available_nodes=cfg.num_clients)


# =========================================================
# Evaluation (Server-side)
# =========================================================
def get_global_evaluate_fn(
    attack,
    *,
    logger: CSVLogger,
    dataset_name: str,
    dataset_meta,
):
    print(">>> [DEBUG] Enter get_global_evaluate_fn")

    try:
        clean_testloader = load_centralized_dataset(dataset_name)
        print(">>> [DEBUG] Loaded clean_testloader")
    except Exception as e:
        print("!!! ERROR while loading clean_testloader:", e)
        import traceback
        traceback.print_exc()
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_label = getattr(attack.config, "target_label", 0)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        print(f">>> [DEBUG] global_evaluate called (round={server_round})")

        try:
            model = Net(
                input_shape=dataset_meta.input_shape,
                num_classes=dataset_meta.num_classes,
            )
            model.load_state_dict(arrays.to_torch_state_dict())
            model.to(device)

            # ------------------------
            # Clean ACC
            # ------------------------
            print(">>> [DEBUG] Running clean test")
            test_loss, test_acc = test(model, clean_testloader, device)

            # ------------------------
            # ASR
            # ------------------------
            if attack.name == "none":
                asr = 0.0
            else:
                triggered_testloader = attack.get_triggered_loader(clean_testloader)
                asr = evaluate_asr(model, triggered_testloader, device, target_label)

            print(
                f"[Round {server_round:02d}] "
                f"ACC={test_acc * 100:.2f}% | "
                f"ASR={asr * 100:.2f}% | "
                f"LOSS={test_loss:.4f}"
            )

            logger.log(
                round=server_round,
                accuracy=float(test_acc),
                asr=float(asr),
                loss=float(test_loss),
            )

            return MetricRecord(
                {
                    "accuracy": float(test_acc),
                    "loss": float(test_loss),
                    "asr": float(asr),
                }
            )

        except Exception as e:
            print("!!! ERROR in global_evaluate:", e)
            import traceback
            traceback.print_exc()
            raise

    return global_evaluate


def evaluate_asr(model, triggered_testloader, device, target_label: int) -> float:
    """Evaluate ASR without updating model state or changing original layer modes."""
    print(">>> [DEBUG] Enter evaluate_asr")

    success = 0
    total = 0

    try:
        with evaluation_mode(model):
            for batch in triggered_testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                mask = labels != target_label
                if mask.sum().item() == 0:
                    continue

                images = images[mask]
                labels = labels[mask]

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                success += (preds == target_label).sum().item()
                total += labels.size(0)

    except Exception as e:
        print("!!! ERROR in evaluate_asr:", e)
        import traceback
        traceback.print_exc()
        raise

    return success / total if total > 0 else 0.0


# =========================================================
# Main
# =========================================================
@app.main()
def main(grid: Grid, context: Context) -> None:
    experiment_dir = None
    metadata = None
    managed_run_id = os.environ.get("FL_BACKDOOR_RUN_ID")
    try:
        print(">>> [DEBUG] Server main() START")
        print(">>> run_config =", dict(context.run_config))

 # 统一配置解析
        cfg = ExperimentConfig.from_run_config(context.run_config)
        seed_everything(cfg.seed)
        print(f"[Config] 实际仿真客户端数={cfg.num_clients}，随机种子={cfg.seed}")

        # 获取数据集元信息
        dataset_meta = get_dataset(cfg.dataset).meta
        print(f">>> [DEBUG] Dataset: {cfg.dataset}, input_shape={dataset_meta.input_shape}, num_classes={dataset_meta.num_classes}")

        # 构建攻击
        attack = build_attack(
            dataset_meta=dataset_meta,
            **cfg.to_attack_kwargs()
        )
        print(">>> [DEBUG] Attack built:", attack)

        # 创建日志器（在构建 pipeline 之前）
        experiment_dir, metadata = ensure_experiment(
            Path(cfg.results_dir), cfg.run_name, run_id=managed_run_id)
        if not managed_run_id:
            metadata.update(status="running")
            write_run_metadata(experiment_dir, metadata)
        experiment_logger = CSVLogger(
            save_dir=str(experiment_dir),
            filename=f"{cfg.run_name}.csv"
        )
        print(f">>> [DEBUG] Logger initialized: {experiment_dir}/{cfg.run_name}.csv")

        # 构建防御流水线（使用 cfg 提供的方法）
        pipeline_kwargs = cfg.get_defense_pipeline_kwargs()
        pipeline = build_defense_pipeline(
            **pipeline_kwargs,
            diagnostics_logger=experiment_logger,
        )
        print(">>> [DEBUG] Pipeline ready:", pipeline)

        # 初始化全局模型
        global_model = Net(
            input_shape=dataset_meta.input_shape,
            num_classes=dataset_meta.num_classes,
        )
        arrays = ArrayRecord(global_model.state_dict())

        # 策略设置
        base_strategy = build_strategy(cfg)
        strategy = pipeline.apply(base_strategy)

        # 启动训练
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": cfg.learning_rate}),
            num_rounds=cfg.num_server_rounds,
            evaluate_fn=get_global_evaluate_fn(
                attack,
                logger=experiment_logger,
                dataset_name=cfg.dataset,
                dataset_meta=dataset_meta,
            ),
        )

        print(">>> [DEBUG] Strategy finished")

        # ========================
        # Save model
        # ========================
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        model_path = save_final_model(state_dict, experiment_dir)
        print(f"Final model saved to {model_path}")
        if not managed_run_id:
            metadata.update(status="finished", exit_code=0, error=None)
            write_run_metadata(experiment_dir, metadata)

        print(">>> [DEBUG] Done.")

    except Exception as e:
        if not managed_run_id and experiment_dir is not None and metadata is not None:
            metadata.update(status="failed", exit_code=1, error=str(e))
            try:
                write_run_metadata(experiment_dir, metadata)
            except OSError:
                pass  # Preserve the original training/model-save exception.
        print("!!! FATAL ERROR in server main:", e)
        import traceback
        traceback.print_exc()
        raise