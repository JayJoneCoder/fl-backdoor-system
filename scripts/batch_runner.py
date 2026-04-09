#!/usr/bin/env python3
"""
Batch runner for FL backdoor experiments.
Usage: python scripts/batch_runner.py [--config experiments.json]
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import tomllib
import tomli_w
import os

PROJECT_ROOT = Path(__file__).parent.parent
TOML_PATH = PROJECT_ROOT / "pyproject.toml"
BACKUP_PATH = PROJECT_ROOT / "pyproject.toml.bak"

# 默认实验配置（可以根据需要修改）
DEFAULT_EXPERIMENTS = [

    {
        "name": "fcba_clustering_none",
        "attack": "fcba",
        "detection": "clustering_detection",
        "defense": "none",
        "client-defense": "none",
        "num-server-rounds": 10,
        "poison-rate": 0.3,
    },
    {
        "name": "fcba_none_none",
        "attack": "fcba",
        "detection": "none",
        "defense": "none",
        "client-defense": "none",
        "num-server-rounds": 10,
        "poison-rate": 0.3,
    },
]

def load_config(config_path: Path = None):
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "experiments" in data:
            common = data.get("common", {})
            experiments = data["experiments"]
            merged = []
            for exp in experiments:
                merged_exp = common.copy()
                merged_exp.update(exp)
                merged.append(merged_exp)
            return merged
        else:
            raise ValueError("Invalid JSON format")
    return DEFAULT_EXPERIMENTS

def modify_toml(original_path: Path, backup_path: Path, updates: dict):
    """Modify pyproject.toml in place with given updates under [tool.flwr.app.config]."""
    # Backup original
    shutil.copy2(original_path, backup_path)
    
    # Read original TOML
    with open(original_path, "rb") as f:
        data = tomllib.load(f)
    
    # Ensure the config section exists
    if "tool" not in data:
        data["tool"] = {}
    if "flwr" not in data["tool"]:
        data["tool"]["flwr"] = {}
    if "app" not in data["tool"]["flwr"]:
        data["tool"]["flwr"]["app"] = {}
    if "config" not in data["tool"]["flwr"]["app"]:
        data["tool"]["flwr"]["app"]["config"] = {}
    
    config_section = data["tool"]["flwr"]["app"]["config"]
    
    # Apply updates
    for key, value in updates.items():
        config_section[key] = value
    
    # Write back
    with open(original_path, "wb") as f:
        tomli_w.dump(data, f)

def restore_toml(backup_path: Path, original_path: Path):
    """Restore original pyproject.toml from backup."""
    shutil.copy2(backup_path, original_path)
    backup_path.unlink()

def run_experiment(exp_name: str, updates: dict):
    """Run a single experiment with given toml updates."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Config: {updates}")
    print(f"{'='*60}\n")
    
    updates["run-name"] = exp_name

    # Modify toml
    modify_toml(TOML_PATH, BACKUP_PATH, updates)
    
    # Run Flower simulation
    log_file = PROJECT_ROOT / "results" / exp_name / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ["flwr", "run", ".", "--stream"]
    
    # 设置环境变量，强制 UTF-8
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                env=env,  # 传递环境变量
                check=True
            )
        print(f"Experiment {exp_name} finished successfully. Log saved to {log_file}")
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp_name} failed with exit code {e.returncode}. See log: {log_file}")
    finally:
        # Restore original toml
        restore_toml(BACKUP_PATH, TOML_PATH)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="JSON file with experiment list")
    args = parser.parse_args()
    
    experiments = load_config(args.config)
    for exp in experiments:
        exp_name = exp.pop("name")  # remove name from updates
        # Convert any non-string values to appropriate types (already)
        run_experiment(exp_name, exp)

if __name__ == "__main__":
    main()