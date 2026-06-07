"""Minimal math evaluation for MiniMind-MathTutor."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .train import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def extract_final_answer(text: str, answer_prefix: str = "答案是") -> str:
    """Return the text after the last Chinese final-answer marker."""

    matches = list(re.finditer(rf"{re.escape(answer_prefix)}\s*[:：]?\s*(.+)", text, re.S))
    if not matches:
        return text.strip()
    answer = matches[-1].group(1).strip()
    return answer.splitlines()[0].strip()


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", "", text)
    return text.strip("。.;；，, ")


def answer_contains(prediction: str, answer: str, answer_prefix: str = "答案是") -> bool:
    expected = normalize_answer(answer)
    if not expected:
        return False
    final_answer = normalize_answer(extract_final_answer(prediction, answer_prefix))
    full_prediction = normalize_answer(prediction)
    return expected in final_answer or expected in full_prediction


def _question_from_row(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    for message in conversations:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return str(row.get("question") or row.get("problem") or row.get("input") or row.get("query") or "").strip()


def _assistant_from_row(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    for message in conversations:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return str(row.get("answer") or row.get("solution") or row.get("output") or row.get("response") or "").strip()


def _answer_from_row(row: dict[str, Any], answer_prefix: str = "答案是") -> str:
    final_answer = str(row.get("final_answer") or "").strip()
    if final_answer:
        return final_answer
    return extract_final_answer(_assistant_from_row(row), answer_prefix)


def _checkpoint_path(config: dict[str, Any], mode: str) -> Path:
    train_cfg = config.get("training", {}).get(mode, {})
    checkpoint = train_cfg.get("output_checkpoint")
    if not checkpoint:
        model_cfg = config.get("project", {}).get("model_size", {})
        hidden_size = int(model_cfg.get("hidden_size", 768))
        use_moe = bool(model_cfg.get("use_moe", 0))
        suffix = "_moe" if use_moe else ""
        save_weight = train_cfg.get("save_weight", mode)
        output_dir = train_cfg.get("output_dir") or train_cfg.get("save_dir", "out")
        checkpoint = f"{output_dir}/{save_weight}_{hidden_size}{suffix}.pth"
    path = Path(checkpoint)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_model(config: dict[str, Any], mode: str, device: str):
    import torch
    from transformers import AutoTokenizer

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    model_cfg = config.get("project", {}).get("model_size", {})
    lm_config = MiniMindConfig(
        hidden_size=int(model_cfg.get("hidden_size", 768)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 8)),
        use_moe=bool(model_cfg.get("use_moe", 0)),
    )
    tokenizer = AutoTokenizer.from_pretrained(REPO_ROOT / "model")
    model = MiniMindForCausalLM(lm_config)
    checkpoint = _checkpoint_path(config, mode)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model.half().eval().to(device), tokenizer


def _generate_one(
    model: Any,
    tokenizer: Any,
    question: str,
    *,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    conversation = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            top_p=top_p,
            temperature=max(temperature, 1e-5),
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated_ids[0][len(inputs["input_ids"][0]) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_math(
    input_path: str | Path,
    *,
    config: dict[str, Any],
    mode: str,
    sample: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    answer_prefix = str(eval_cfg.get("answer_prefix", "答案是"))
    rows = read_jsonl(input_path)

    checkpoint = _checkpoint_path(config, mode)
    use_sample_answers = sample and not checkpoint.exists()
    model = tokenizer = None
    if use_sample_answers:
        print(f"Sample mode: checkpoint not found, evaluating stored assistant answers: {checkpoint}")
    else:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {mode}: {checkpoint}")
        device = config.get("environment", {}).get("default_device", "cuda:0")
        model, tokenizer = _load_model(config, mode, device)

    max_new_tokens = int(eval_cfg.get("max_new_tokens", 256))
    temperature = float(eval_cfg.get("temperature", 0.1))
    top_p = float(eval_cfg.get("top_p", 0.95))
    device = config.get("environment", {}).get("default_device", "cuda:0")

    details: list[dict[str, Any]] = []
    for row in rows:
        question = _question_from_row(row)
        expected = _answer_from_row(row, answer_prefix)
        start = time.perf_counter()
        if use_sample_answers:
            prediction = _assistant_from_row(row)
        else:
            prediction = _generate_one(
                model,
                tokenizer,
                question,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        latency = time.perf_counter() - start
        details.append(
            {
                "question": question,
                "expected": expected,
                "prediction": prediction,
                "predicted_final_answer": extract_final_answer(prediction, answer_prefix),
                "answer_contains": answer_contains(prediction, expected, answer_prefix),
                "output_length": len(prediction),
                "latency_seconds": latency,
            }
        )

    total = len(details)
    report = {
        "mode": mode,
        "input": str(input_path),
        "checkpoint": str(checkpoint),
        "sample_fallback": use_sample_answers,
        "total": total,
        "answer_contains": sum(item["answer_contains"] for item in details) / total if total else 0.0,
        "avg_output_length": sum(item["output_length"] for item in details) / total if total else 0.0,
        "avg_latency_seconds": sum(item["latency_seconds"] for item in details) / total if total else 0.0,
    }

    if output_path:
        output = Path(output_path)
        if output.parent.exists():
            output.write_text(json.dumps({"report": report, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(f"Report path parent does not exist, skipped writing: {output.parent}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal MiniMind math evaluation.")
    parser.add_argument("--config", default="configs/math_tutor.yaml")
    parser.add_argument("--mode", choices=["official_sft", "math_sft"], default="math_sft")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample", action="store_true", help="Use configured sample data; falls back to stored answers if the checkpoint is absent.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    eval_cfg = config.get("evaluation", {})
    input_path = args.input or (eval_cfg.get("sample_questions") if args.sample else eval_cfg.get("test_file"))
    if not input_path:
        raise ValueError("No evaluation input configured. Pass --input or set evaluation.test_file.")
    output_path = args.output if args.output is not None else eval_cfg.get("report_path")
    report = evaluate_math(input_path, config=config, mode=args.mode, sample=args.sample, output_path=output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
