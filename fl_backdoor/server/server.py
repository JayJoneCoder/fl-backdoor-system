"""fl_backdoor: A Flower / PyTorch app."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_backdoor.attacks import build_attack
from fl_backdoor.defenses import build_defended_strategy
from fl_backdoor.task import Net, load_centralized_dataset, test
from fl_backdoor.utils.logger import CSVLogger

# Create ServerApp
app = ServerApp()


# =========================================================
# Evaluation (Server-side)
# =========================================================
def get_global_evaluate_fn(
    attack,
    *,
    results_dir: str = "results",
    run_name: str = "experiment",
):
    """Build server-side evaluation function with debug and CSV logging."""

    print(">>> [DEBUG] Enter get_global_evaluate_fn")

    # Load dataset ONCE (important)
    try:
        clean_testloader = load_centralized_dataset()
        print(">>> [DEBUG] Loaded clean_testloader")
    except Exception as e:
        print("!!! ERROR while loading clean_testloader:", e)
        import traceback
        traceback.print_exc()
        raise

    # Logger
    logger = CSVLogger(
        save_dir=results_dir,
        filename=f"{run_name}.csv",
    )
    print(f">>> [DEBUG] Logger initialized: {results_dir}/{run_name}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_label = getattr(attack.config, "target_label", 0)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate global model each round."""

        print(f">>> [DEBUG] global_evaluate called (round={server_round})")

        try:
            model = Net()
            model.load_state_dict(arrays.to_torch_state_dict())
            model.to(device)

            # ------------------------
            # Clean Accuracy
            # ------------------------
            print(">>> [DEBUG] Running clean test")
            test_loss, test_acc = test(model, clean_testloader, device)
            print(">>> [DEBUG] Clean test done")

            # ------------------------
            # ASR (Attack Success Rate)
            # ------------------------
            if attack.name == "none":
                asr = 0.0
                print(">>> [DEBUG] attack.name == none → skip ASR")
            else:
                print(">>> [DEBUG] Building triggered_testloader")
                triggered_testloader = attack.get_triggered_loader(clean_testloader)
                print(">>> [DEBUG] Built triggered_testloader")

                print(">>> [DEBUG] Running ASR evaluation")
                asr = evaluate_asr(model, triggered_testloader, device, target_label)
                print(">>> [DEBUG] ASR evaluation done")

            # ------------------------
            # Print
            # ------------------------
            print(
                f"[Round {server_round:02d}] "
                f"ACC={test_acc * 100:.2f}% | "
                f"ASR={asr * 100:.2f}% | "
                f"LOSS={test_loss:.4f}"
            )

            # ------------------------
            # Log
            # ------------------------
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
    """Compute Attack Success Rate (ASR)."""

    print(">>> [DEBUG] Enter evaluate_asr")

    model.eval()
    success = 0
    total = 0

    try:
        with torch.no_grad():
            for batch in triggered_testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                # ignore already target-label samples
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
    """Main entry point for the ServerApp."""

    try:
        print(">>> [DEBUG] Server main() START")
        print(">>> run_config =", dict(context.run_config))

        # ========================
        # Basic config
        # ========================
        fraction_evaluate: float = context.run_config["fraction-evaluate"]
        num_rounds: int = context.run_config["num-server-rounds"]
        lr: float = context.run_config["learning-rate"]

        # ========================
        # Attack config
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
        # Defense config
        # ========================
        defense_type = str(context.run_config.get("defense", "none")).lower()
        defense_kwargs = {}

        for key, value in dict(context.run_config).items():
            if key.startswith("defense-") and key != "defense":
                defense_key = key.removeprefix("defense-").replace("-", "_")
                defense_kwargs[defense_key] = value

        if "clip-norm" in context.run_config and "clip_norm" not in defense_kwargs:
            defense_kwargs["clip_norm"] = context.run_config["clip-norm"]

        # ========================
        # Experiment naming (VERY IMPORTANT)
        # ========================
        results_dir = str(context.run_config.get("results-dir", "results"))

        run_name = str(
            context.run_config.get(
                "run-name",
                f"{attack_type}_{defense_type}_seed{seed}",
            )
        )

        print(f">>> [DEBUG] results_dir = {results_dir}")
        print(f">>> [DEBUG] run_name = {run_name}")

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
        # Model init
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
                results_dir=results_dir,
                run_name=run_name,
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