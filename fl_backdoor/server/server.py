"""fl_backdoor: A Flower / PyTorch app."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_backdoor.attacks import build_attack
from fl_backdoor.defenses import build_defended_strategy
from fl_backdoor.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


def get_global_evaluate_fn(attack):
    """Build server-side evaluation function with debug."""

    print(">>> [DEBUG] Enter get_global_evaluate_fn")

    try:
        clean_testloader = load_centralized_dataset()
        print(">>> [DEBUG] Loaded clean_testloader")

        triggered_testloader = attack.get_triggered_loader(clean_testloader)
        print(">>> [DEBUG] Built triggered_testloader")

    except Exception as e:
        print("!!! ERROR in get_global_evaluate_fn:", e)
        import traceback
        traceback.print_exc()
        raise

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_label = getattr(attack.config, "target_label", 0)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        print(f">>> [DEBUG] global_evaluate called (round={server_round})")

        try:
            model = Net()
            model.load_state_dict(arrays.to_torch_state_dict())
            model.to(device)

            # Clean evaluation
            print(">>> [DEBUG] Running clean test")
            test_loss, test_acc = test(model, clean_testloader, device)
            print(">>> [DEBUG] Clean test done")
            
            # Triggered evaluation for ASR
            if attack.name == "none":
                asr = 0.0
            else:
                asr = evaluate_asr(model, triggered_testloader, device, target_label)

            print(
                f"[Round {server_round:02d}] "
                f"ACC={test_acc*100:.2f}% | "
                f"ASR={asr*100:.2f}% | "
                f"LOSS={test_loss:.4f}"
            )

            return MetricRecord(
                {
                    "accuracy": round(float(test_acc), 4),
                    "loss": round(float(test_loss), 4),
                    "asr": round(float(asr), 4),
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
            for i, batch in enumerate(triggered_testloader):

                images = batch["img"]
                labels = batch["label"]

                images = images.to(device)
                labels = labels.to(device)

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


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    try:
        print(">>> [DEBUG] Server main() START")
        print(">>> run_config =", dict(context.run_config))

        # ========================
        # Read run config
        # ========================
        fraction_evaluate: float = context.run_config["fraction-evaluate"]
        num_rounds: int = context.run_config["num-server-rounds"]
        lr: float = context.run_config["learning-rate"]

        # ========================
        # Attack-related config
        # ========================
        attack_type = str(
            context.run_config.get(
                "attack-type",
                context.run_config.get("attack", "badnets"),
            )
        ).lower()

        malicious_ratio = float(context.run_config.get("malicious-ratio", 0.2))
        poison_rate = float(context.run_config.get("poison-rate", 0.05))
        target_label = int(context.run_config.get("target-label", 0))
        trigger_size = int(context.run_config.get("trigger-size", 4))
        seed = int(context.run_config.get("seed", 42))
        grid_size = context.run_config.get("wanet-grid-size", None)
        noise_scale = float(context.run_config.get("wanet-noise", 0.05))

        # ========================
        # Defense-related config
        # ========================
        defense_type = str(context.run_config.get("defense", "none")).lower()
        defense_kwargs = {}

        # Recommended new style: defense-clip-norm, defense-anything-else
        for key, value in dict(context.run_config).items():
            if key.startswith("defense-") and key != "defense":
                defense_key = key.removeprefix("defense-").replace("-", "_")
                defense_kwargs[defense_key] = value

        # Backward compatibility with the earlier key name
        if "clip-norm" in context.run_config and "clip_norm" not in defense_kwargs:
            defense_kwargs["clip_norm"] = context.run_config["clip-norm"]
        
        # ========================
        # Build attack
        # ========================


        print(">>> [DEBUG] Building attack...")


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


        print(">>> [DEBUG] Attack built:", attack)

        # ========================
        # Load global model
        # ========================
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

        # ========================
        # Strategy
        # ========================
        base_strategy = FedAvg(fraction_evaluate=fraction_evaluate)
        strategy = build_defended_strategy(
            base_strategy,
            defense_type=defense_type,
            seed=seed,
            **defense_kwargs,
        )

        print(">>> [DEBUG] Starting strategy...")

        # ========================
        # Start FL
        # ========================
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=get_global_evaluate_fn(attack),
        )

        print(">>> [DEBUG] Strategy finished")

        # ========================
        # Save model
        # ========================
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

    except Exception as e:
        print("!!! FATAL ERROR in server main:", e)
        import traceback
        traceback.print_exc()
        raise