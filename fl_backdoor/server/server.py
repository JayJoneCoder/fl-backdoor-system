"""fl_backdoor: A Flower / PyTorch app."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_backdoor.attacks import build_attack
from fl_backdoor.defenses.pipeline import build_defense_pipeline
from fl_backdoor.task import Net, load_centralized_dataset, test
from fl_backdoor.utils.logger import CSVLogger
from pathlib import Path

# Create ServerApp
app = ServerApp()


# =========================================================
# Evaluation (Server-side)
# =========================================================
def get_global_evaluate_fn(
    attack,
    *,
    logger: CSVLogger,
):
    print(">>> [DEBUG] Enter get_global_evaluate_fn")

    try:
        clean_testloader = load_centralized_dataset()
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
            model = Net()
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
    print(">>> [DEBUG] Enter evaluate_asr")

    model.eval()
    success = 0
    total = 0

    try:
        with torch.no_grad():
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
    try:
        print(">>> [DEBUG] Server main() START")
        print(">>> run_config =", dict(context.run_config))

        # ========================
        # Basic config
        # ========================
        fraction_evaluate: float = context.run_config["fraction-evaluate"]
        num_rounds: int = context.run_config["num-server-rounds"]
        lr: float = context.run_config["learning-rate"]
        seed = int(context.run_config.get("seed", 42))

                # ========================
        # Attack config
        # ========================
        attack_type = str(
            context.run_config.get("attack-type", context.run_config.get("attack", "badnets"))
        ).lower()

        malicious_ratio = float(context.run_config.get("malicious-ratio", 0.2))
        poison_rate = float(context.run_config.get("poison-rate", 0.05))
        target_label = int(context.run_config.get("target-label", 0))
        trigger_size = int(context.run_config.get("trigger-size", 4))

        grid_size = context.run_config.get("wanet-grid-size", None)
        noise_scale = float(context.run_config.get("wanet-noise", 0.05))

        frequency_mode = str(context.run_config.get("frequency-mode", "dct"))
        frequency_band = str(context.run_config.get("frequency-band", "low"))
        frequency_window_size = context.run_config.get("frequency-window-size", None)
        frequency_intensity = float(context.run_config.get("frequency-intensity", 0.10))
        frequency_mix_alpha = float(context.run_config.get("frequency-mix-alpha", 1.0))

        malicious_mode = str(context.run_config.get("attack-malicious-mode", "random"))
        fixed_malicious_clients = context.run_config.get("attack-fixed-clients", None)

        attack = build_attack(
            attack_type=attack_type,
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
        )

        print(">>> [DEBUG] Attack built:", attack)

        # ========================
        # Defense configs
        # ========================
        client_defense_type = str(context.run_config.get("client-defense", "none")).lower()
        detection_type = str(context.run_config.get("detection", "none")).lower()
        aggregation_type = str(context.run_config.get("defense", "none")).lower()

        def extract_kwargs(prefix: str):
            out = {}
            for k, v in dict(context.run_config).items():
                if k.startswith(prefix):
                    key = k.removeprefix(prefix).replace("-", "_")
                    out[key] = v
            return out

        client_defense_kwargs = extract_kwargs("client-defense-")
        detection_kwargs = extract_kwargs("detection-")
        aggregation_kwargs = extract_kwargs("defense-")

        # ========================
        # Naming
        # ========================
        results_dir = str(context.run_config.get("results-dir", "results"))
        run_name = str(context.run_config.get("run-name", "experiment"))

        # 创建子目录
        experiment_dir = Path(results_dir) / run_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        print(f">>> [DEBUG] run_name = {run_name}")

        # ========================
        # Logger
        # ========================
        experiment_logger = CSVLogger(
            save_dir=str(experiment_dir),
            filename=f"{run_name}.csv"
        )
        print(f">>> [DEBUG] Logger initialized: {results_dir}/{run_name}.csv")

        # ========================
        # Model init
        # ========================
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

        # ========================
        # Strategy + Pipeline
        # ========================
        base_strategy = FedAvg(fraction_evaluate=fraction_evaluate)

        pipeline = build_defense_pipeline(
            client_defense_type=client_defense_type,
            detection_type=detection_type,
            aggregation_type=aggregation_type,
            seed=seed,
            client_defense_kwargs=client_defense_kwargs,
            detection_kwargs=detection_kwargs,
            aggregation_kwargs=aggregation_kwargs,
            diagnostics_logger=experiment_logger,
        )

        
        print(">>> [DEBUG] Pipeline ready:", pipeline)

        strategy = pipeline.apply(base_strategy)

        print(">>> [DEBUG] Strategy ready")

        # ========================
        # Start FL
        # ========================
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=get_global_evaluate_fn(
                attack,
                logger=experiment_logger,
            ),
        )

        print(">>> [DEBUG] Strategy finished")

        # ========================
        # Save model
        # ========================
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

        print(">>> [DEBUG] Done.")

    except Exception as e:
        print("!!! FATAL ERROR in server main:", e)
        import traceback
        traceback.print_exc()
        raise