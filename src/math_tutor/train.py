"""Command builders for official MiniMind training scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs/math_tutor.yaml.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def local_python(config: dict[str, Any]) -> str:
    configured = config.get("environment", {}).get("local_python")
    return configured or sys.executable


def sft_command(config: dict[str, Any], stage: str = "stage1_official_sft_mini") -> list[str]:
    train_cfg = config["training"][stage]
    return [
        local_python(config),
        "train_full_sft.py",
        "--data_path",
        f"../{train_cfg['data_path']}",
        "--save_dir",
        f"../{train_cfg['save_dir']}",
        "--save_weight",
        train_cfg["save_weight"],
        "--from_weight",
        train_cfg["from_weight"],
        "--from_resume",
        "1" if train_cfg.get("resume") else "0",
        "--epochs",
        str(train_cfg["epochs"]),
        "--batch_size",
        str(train_cfg["batch_size"]),
        "--accumulation_steps",
        str(train_cfg["gradient_accumulation_steps"]),
        "--learning_rate",
        str(train_cfg["learning_rate"]),
        "--max_seq_len",
        str(train_cfg["max_seq_len"]),
        "--save_interval",
        str(train_cfg["save_steps"]),
        "--num_workers",
        str(train_cfg["num_workers"]),
        "--dtype",
        str(train_cfg.get("dtype", "float16")),
        "--device",
        config.get("environment", {}).get("default_device", "cuda:0"),
    ]


def lora_command(config: dict[str, Any]) -> list[str]:
    train_cfg = config["training"]["math_lora"]
    return [
        local_python(config),
        "train_lora.py",
        "--data_path",
        f"../{train_cfg['data_path']}",
        "--save_dir",
        f"../{train_cfg['save_dir']}",
        "--lora_name",
        train_cfg["lora_name"],
        "--from_weight",
        train_cfg["from_weight"],
        "--from_resume",
        "1" if train_cfg.get("resume") else "0",
        "--epochs",
        str(train_cfg["epochs"]),
        "--batch_size",
        str(train_cfg["batch_size"]),
        "--accumulation_steps",
        str(train_cfg["gradient_accumulation_steps"]),
        "--learning_rate",
        str(train_cfg["learning_rate"]),
        "--max_seq_len",
        str(train_cfg["max_seq_len"]),
        "--save_interval",
        str(train_cfg["save_steps"]),
        "--num_workers",
        str(train_cfg["num_workers"]),
        "--dtype",
        str(train_cfg.get("dtype", "float16")),
        "--device",
        config.get("environment", {}).get("default_device", "cuda:0"),
    ]


def run_or_print(command: list[str], run: bool, cuda_visible_devices: str = "0") -> None:
    pretty = " ".join(command)
    print(pretty)
    if not run:
        print("Dry mode: command printed only. Add --run to execute.")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(command, cwd=REPO_ROOT / "trainer", env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or run MiniMind-MathTutor official training commands.")
    parser.add_argument("--config", default="configs/math_tutor.yaml")
    parser.add_argument("--task", choices=["stage1_sft", "math_sft", "math_lora"], default="stage1_sft")
    parser.add_argument("--run", action="store_true", help="Actually execute the official MiniMind trainer.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.task == "stage1_sft":
        command = sft_command(config, "stage1_official_sft_mini")
    elif args.task == "math_sft":
        command = sft_command(config, "math_sft")
    else:
        command = lora_command(config)
    run_or_print(command, args.run, config.get("environment", {}).get("local_cuda_visible_devices", "0"))


if __name__ == "__main__":
    main()
