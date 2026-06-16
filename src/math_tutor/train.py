"""Command builders for official MiniMind training scripts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OVERFIT_DEBUG_DATA = "outputs/overfit_debug_math_sft_100.jsonl"
OVERFIT_DEBUG_SAVE_WEIGHT = "overfit_debug_math_sft"


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs/math_tutor.yaml.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def local_python(config: dict[str, Any]) -> str:
    configured = config.get("environment", {}).get("local_python")
    if configured and Path(configured).exists():
        return configured
    return sys.executable


def _repo_relative_for_trainer(path: str | Path) -> str:
    value = Path(path).as_posix()
    if value.startswith("../") or Path(value).is_absolute():
        return value
    return f"../{value}"


def _sft_train_file(train_cfg: dict[str, Any]) -> str:
    return train_cfg.get("train_file") or train_cfg.get("data_path")


def _sft_output_dir(train_cfg: dict[str, Any]) -> str:
    return train_cfg.get("output_dir") or train_cfg.get("save_dir", "out")


def _checkpoint_to_weight_name(checkpoint: str | Path, hidden_size: int, use_moe: int | bool = 0) -> str:
    name = Path(checkpoint).name
    suffix = f"_{hidden_size}{'_moe' if use_moe else ''}.pth"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return Path(name).stem


def _sft_from_weight(config: dict[str, Any], train_cfg: dict[str, Any]) -> str:
    if train_cfg.get("from_weight"):
        return str(train_cfg["from_weight"])
    checkpoint = train_cfg.get("base_checkpoint")
    if checkpoint:
        model_cfg = config.get("project", {}).get("model_size", {})
        return _checkpoint_to_weight_name(
            checkpoint,
            int(model_cfg.get("hidden_size", 768)),
            model_cfg.get("use_moe", 0),
        )
    return "full_sft"


def _resolve_repo_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _print_dry_run_notes(config: dict[str, Any], mode: str) -> None:
    train_cfg = config["training"][mode]
    if mode == "math_sft":
        base_checkpoint = train_cfg.get("base_checkpoint")
        train_file = _sft_train_file(train_cfg)
        for label, path in (("base_checkpoint", base_checkpoint), ("train_file", train_file)):
            if path and not (REPO_ROOT / path).exists():
                print(f"Dry run note: configured {label} does not exist yet: {path}")
        unused = [key for key in ("valid_file", "warmup_ratio", "eval_steps") if key in train_cfg]
        if unused:
            print(
                "Dry run note: MiniMind trainer/train_full_sft.py does not consume "
                + ", ".join(unused)
                + "; they are kept in config for experiment tracking."
            )


def inspect_sft_setup(config: dict[str, Any], mode: str = "math_sft") -> dict[str, Any]:
    """Inspect checkpoint, data, tokenizer, and label-mask assumptions."""

    train_cfg = config["training"][mode]
    model_cfg = config.get("project", {}).get("model_size", {})
    train_file = _sft_train_file(train_cfg)
    valid_file = train_cfg.get("valid_file")
    base_checkpoint = train_cfg.get("base_checkpoint")
    output_checkpoint = train_cfg.get("output_checkpoint")
    max_seq_len = int(train_cfg.get("max_seq_len", 768))

    report: dict[str, Any] = {
        "mode": mode,
        "base_checkpoint": base_checkpoint,
        "base_checkpoint_exists": bool((path := _resolve_repo_path(base_checkpoint)) and path.exists()),
        "from_weight": _sft_from_weight(config, train_cfg),
        "train_file": train_file,
        "train_file_exists": bool((path := _resolve_repo_path(train_file)) and path.exists()),
        "valid_file": valid_file,
        "valid_file_exists": bool((path := _resolve_repo_path(valid_file)) and path.exists()),
        "output_checkpoint": output_checkpoint,
        "output_checkpoint_exists": bool((path := _resolve_repo_path(output_checkpoint)) and path.exists()),
        "resume": bool(train_cfg.get("resume")),
        "epochs": train_cfg.get("epochs"),
        "learning_rate": train_cfg.get("learning_rate"),
        "batch_size": train_cfg.get("batch_size"),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps"),
        "max_seq_len": max_seq_len,
        "valid_loss_recorded": False,
        "valid_loss_note": "MiniMind trainer/train_full_sft.py does not run validation loss; valid_file is config-only.",
        "train_loss_diagnostics": _loss_trend_diagnostics(_resolve_repo_path(train_cfg.get("train_log") or train_cfg.get("log_file"))),
        "train_loss_note": "Loss trend is printed by the official trainer during training; configure training.math_sft.train_log to parse it.",
        "loss_mask_expected": "MiniMind SFTDataset labels only assistant spans and sets other tokens to -100.",
    }

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(REPO_ROOT / "model")
        report["tokenizer_loaded"] = True
        report["tokenizer_probe"] = _tokenizer_probe(tokenizer)
    except Exception as exc:  # pragma: no cover - optional dependency path
        report["tokenizer_loaded"] = False
        report["tokenizer_error"] = str(exc)

    train_path = _resolve_repo_path(train_file)
    if train_path and train_path.exists():
        try:
            from .data import diagnose_sft_file

            report["data_diagnostics"] = diagnose_sft_file(train_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
        except Exception as exc:
            report["data_diagnostics_error"] = str(exc)
        try:
            report["loss_mask_diagnostics"] = _loss_mask_diagnostics(train_path, tokenizer, max_seq_len)
        except Exception as exc:
            report["loss_mask_diagnostics_error"] = str(exc)

    return report


def _tokenizer_probe(tokenizer: Any) -> dict[str, int]:
    samples = {
        "digits": "1234567890",
        "math_symbols": "1/2 + 3.5 = -4%",
        "english_problem": "Janet has 16 eggs and sells each for $2.",
        "chinese_solution": "答案是：18",
        "boxed": r"\boxed{42}",
    }
    return {name: len(tokenizer(text, add_special_tokens=False).input_ids) for name, text in samples.items()}


def _loss_mask_diagnostics(train_path: Path, tokenizer: Any, max_seq_len: int, samples: int = 3) -> dict[str, Any]:
    if tokenizer is None:
        return {"checked": False, "reason": "tokenizer unavailable"}
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from dataset.lm_dataset import SFTDataset

    dataset = SFTDataset(str(train_path), tokenizer, max_length=max_seq_len)
    checked = min(samples, len(dataset))
    rows: list[dict[str, Any]] = []
    for index in range(checked):
        input_ids, labels = dataset[index]
        label_count = int((labels != -100).sum().item())
        non_pad_count = int((input_ids != tokenizer.pad_token_id).sum().item())
        rows.append(
            {
                "index": index,
                "non_pad_tokens": non_pad_count,
                "label_tokens": label_count,
                "label_ratio": label_count / max(non_pad_count, 1),
                "has_supervised_tokens": label_count > 0,
            }
        )
    return {
        "checked": True,
        "samples": checked,
        "assistant_only_mask": True,
        "rows": rows,
    }


def _loss_trend_diagnostics(log_path: Path | None) -> dict[str, Any]:
    if log_path is None:
        return {"checked": False, "reason": "no train_log configured"}
    if not log_path.exists():
        return {"checked": False, "reason": f"train_log not found: {log_path}"}
    values: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"\bloss:\s*([0-9]+(?:\.[0-9]+)?)", line)
        if match:
            values.append(float(match.group(1)))
    if not values:
        return {"checked": False, "reason": "no loss values found in train_log", "path": str(log_path)}
    return {
        "checked": True,
        "path": str(log_path),
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "minimum": min(values),
        "decreased": values[-1] < values[0],
    }


def official_sft_command(config: dict[str, Any], mode: str = "official_sft") -> list[str]:
    """Build a command that delegates to MiniMind's official train_full_sft.py."""

    train_cfg = config["training"][mode]
    return [
        local_python(config),
        "train_full_sft.py",
        "--data_path",
        _repo_relative_for_trainer(_sft_train_file(train_cfg)),
        "--save_dir",
        _repo_relative_for_trainer(_sft_output_dir(train_cfg)),
        "--save_weight",
        train_cfg["save_weight"],
        "--from_weight",
        _sft_from_weight(config, train_cfg),
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


def math_sft_command(config: dict[str, Any]) -> list[str]:
    """Build the math SFT command using MiniMind's original SFT trainer."""

    return official_sft_command(config, "math_sft")


def overfit_debug_command(config: dict[str, Any]) -> list[str]:
    debug_cfg = config.get("debug", {})
    hidden_size = int(config.get("project", {}).get("model_size", {}).get("hidden_size", 768))
    overfit_config = _overfit_debug_config(config)
    command = official_sft_command(overfit_config, "math_sft")
    expected_checkpoint = REPO_ROOT / "out" / f"{OVERFIT_DEBUG_SAVE_WEIGHT}_{hidden_size}.pth"
    print(f"Overfit debug checkpoint: {expected_checkpoint}")
    print(f"Overfit samples: {debug_cfg.get('overfit_samples', 100)}")
    return command


def sft_command(config: dict[str, Any], stage: str = "official_sft") -> list[str]:
    return official_sft_command(config, stage)


def torchrun_official_sft_command(config: dict[str, Any], nproc_per_node: int = 2) -> list[str]:
    command = official_sft_command(config, "official_sft")
    return ["torchrun", "--nproc_per_node", str(nproc_per_node)] + command[1:]


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
    pretty = " ".join(shlex.quote(part) for part in command)
    print(pretty)
    if not run:
        print("Dry run: command printed only. Add --run to execute.")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(command, cwd=REPO_ROOT / "trainer", env=env, check=True)


def run_official_sft(config: dict[str, Any], mode: str = "official_sft", run: bool = False) -> None:
    command = official_sft_command(config, mode)
    if not run:
        _print_dry_run_notes(config, mode)
    run_or_print(command, run, config.get("environment", {}).get("local_cuda_visible_devices", "0"))


def run_math_sft(config: dict[str, Any], run: bool = False) -> None:
    command = math_sft_command(config)
    if not run:
        _print_dry_run_notes(config, "math_sft")
    run_or_print(command, run, config.get("environment", {}).get("local_cuda_visible_devices", "0"))


def run_diagnostics(config: dict[str, Any], mode: str = "math_sft") -> None:
    report = inspect_sft_setup(config, mode=mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_math_sft_overfit_debug(config: dict[str, Any], run: bool = True) -> None:
    overfit_config = _overfit_debug_config(config)
    if run:
        subset_path = _prepare_overfit_subset(config)
    else:
        subset_path = REPO_ROOT / OVERFIT_DEBUG_DATA
        print("Dry run: overfit subset will be prepared only when training is executed.")
    print(f"Overfit debug data: {subset_path}")
    print(json.dumps(inspect_sft_setup(overfit_config, mode="math_sft"), ensure_ascii=False, indent=2))
    command = official_sft_command(overfit_config, "math_sft")
    run_or_print(command, run, config.get("environment", {}).get("local_cuda_visible_devices", "0"))
    if run:
        from .eval import evaluate_math

        debug_cfg = config.get("debug", {})
        report = evaluate_math(
            subset_path,
            config=overfit_config,
            mode="math_sft",
            sample=False,
            output_path=None,
            debug=True,
            debug_samples=int(debug_cfg.get("debug_eval_samples", 20)),
            debug_output_path=debug_cfg.get("save_debug_predictions", "outputs/debug_predictions.jsonl"),
        )
        print(json.dumps({"overfit_eval": report}, ensure_ascii=False, indent=2))


def _overfit_debug_config(config: dict[str, Any]) -> dict[str, Any]:
    import copy

    debug_cfg = config.get("debug", {})
    overfit_config = copy.deepcopy(config)
    train_cfg = overfit_config["training"]["math_sft"]
    train_cfg["train_file"] = OVERFIT_DEBUG_DATA
    train_cfg["valid_file"] = OVERFIT_DEBUG_DATA
    train_cfg["save_weight"] = OVERFIT_DEBUG_SAVE_WEIGHT
    train_cfg["output_checkpoint"] = f"out/{OVERFIT_DEBUG_SAVE_WEIGHT}_{overfit_config.get('project', {}).get('model_size', {}).get('hidden_size', 768)}.pth"
    train_cfg["epochs"] = int(debug_cfg.get("overfit_epochs", 3))
    train_cfg["save_steps"] = 1000
    train_cfg["resume"] = False
    return overfit_config


def _prepare_overfit_subset(config: dict[str, Any]) -> Path:
    from .data import format_math_sft_records, read_jsonl

    train_cfg = config["training"]["math_sft"]
    debug_cfg = config.get("debug", {})
    source_path = _resolve_repo_path(_sft_train_file(train_cfg))
    if not source_path or not source_path.exists():
        raise FileNotFoundError(f"Math SFT train_file not found: {_sft_train_file(train_cfg)}")
    output_path = REPO_ROOT / OVERFIT_DEBUG_DATA
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    rows = read_jsonl(source_path)[: int(debug_cfg.get("overfit_samples", 100))]
    if debug_cfg.get("format_overfit_data", True):
        rows, stats = format_math_sft_records(rows, final_answer_prefix=debug_cfg.get("final_answer_prefix", "答案是："))
        print(
            "Formatted overfit data: "
            f"read={stats['read']}, formatted={stats['formatted']}, skipped={stats['skipped']}"
        )
        if not rows:
            raise ValueError("No overfit rows remained after final-answer formatting.")
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or run MiniMind-MathTutor official training commands.")
    parser.add_argument("--config", default="configs/math_tutor.yaml")
    parser.add_argument("--task", choices=["official_sft", "math_sft", "math_lora"], default="official_sft")
    parser.add_argument("--run", action="store_true", help="Actually execute the official MiniMind trainer.")
    parser.add_argument("--diagnose", action="store_true", help="Print SFT setup diagnostics and do not train.")
    parser.add_argument("--overfit_debug", action="store_true", help="Run the configured 100-sample overfit debug workflow.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.diagnose:
        run_diagnostics(config, mode=args.task if args.task != "math_lora" else "math_sft")
        return
    if args.overfit_debug:
        run_math_sft_overfit_debug(config, run=args.run)
        return
    if args.task == "official_sft":
        command = official_sft_command(config, "official_sft")
    elif args.task == "math_sft":
        command = math_sft_command(config)
    else:
        command = lora_command(config)
    run_or_print(command, args.run, config.get("environment", {}).get("local_cuda_visible_devices", "0"))


if __name__ == "__main__":
    main()
