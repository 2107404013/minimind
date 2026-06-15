"""Minimal math evaluation and diagnostics for MiniMind-MathTutor."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .train import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def _strip_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", str(text))
    return matches[-1].strip() if matches else str(text)


def _clean_answer_fragment(text: str) -> str:
    value = _strip_boxed(str(text)).strip()
    value = re.split(r"[\n\r]", value, maxsplit=1)[0]
    value = re.split(r"[。；;]", value, maxsplit=1)[0]
    return value.strip(" \t。.,，;；:：")


def extract_final_answer(text: str, answer_prefix: str = "答案是") -> str:
    """Extract a compact final answer from common math-answer formats."""

    raw = str(text or "").strip()
    if not raw:
        return ""

    boxed = _strip_boxed(raw)
    if boxed != raw:
        return _clean_answer_fragment(boxed)

    patterns = [
        r"####\s*([^\n\r]+)",
        rf"{re.escape(answer_prefix)}\s*[:：]?\s*([^\n\r]+)",
        r"(?:therefore,\s*)?the\s+answer\s+is\s*[:：]?\s*([^\n\r]+)",
        r"final\s+answer\s*[:：]?\s*([^\n\r]+)",
        r"answer\s*[:：]\s*([^\n\r]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, raw, re.I))
        if matches:
            return _clean_answer_fragment(matches[-1].group(1))

    return _clean_answer_fragment(raw)


def normalize_answer(answer: str) -> str:
    """Normalize text answers by removing formatting, spaces, commas, and units."""

    text = extract_final_answer(str(answer))
    text = text.lower()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\s*\{(-?\d+(?:\.\d+)?)\}\s*\{(-?\d+(?:\.\d+)?)\}", r"\1/\2", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(dollars?|eggs?|bolts?|cups?|meters?|miles?|minutes?|hours?|days?|percent|percentage)$", "", text)
    text = re.sub(r"[\u4e00-\u9fff]+$", "", text)
    return text.strip("。.,，;；:：")


def _number_tokens(text: str) -> list[str]:
    text = str(text).replace(",", "")
    frac_pattern = r"-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?%?"
    num_pattern = r"-?\d+(?:\.\d+)?%?"
    return re.findall(frac_pattern, text) or re.findall(num_pattern, text)


def _to_fraction(token: str) -> Fraction | None:
    value = token.strip().replace(",", "")
    if not value:
        return None
    is_percent = value.endswith("%")
    value = value[:-1] if is_percent else value
    try:
        if "/" in value:
            left, right = value.split("/", 1)
            parsed = Fraction(left.strip()) / Fraction(right.strip())
        else:
            parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed / 100 if is_percent else parsed


def _canonical_numeric(answer: str) -> Fraction | None:
    normalized = normalize_answer(answer)
    tokens = _number_tokens(normalized)
    if not tokens:
        tokens = _number_tokens(answer)
    return _to_fraction(tokens[-1]) if tokens else None


def exact_match(pred: str, gold: str) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return bool(gold_norm) and pred_norm == gold_norm


def relaxed_match(pred: str, gold: str) -> bool:
    if exact_match(pred, gold):
        return True
    pred_num = _canonical_numeric(pred)
    gold_num = _canonical_numeric(gold)
    return pred_num is not None and gold_num is not None and pred_num == gold_num


def answer_contains(pred_text: str, gold: str, answer_prefix: str = "答案是") -> bool:
    gold_final = extract_final_answer(gold, answer_prefix)
    pred_final = extract_final_answer(pred_text, answer_prefix)
    if relaxed_match(pred_final, gold_final):
        return True
    gold_norm = normalize_answer(gold_final)
    pred_norm = normalize_answer(pred_text)
    return bool(gold_norm) and gold_norm in pred_norm


def invalid_output(pred_text: str, answer_prefix: str = "答案是") -> bool:
    final = extract_final_answer(pred_text, answer_prefix)
    return not final or (_canonical_numeric(final) is None and normalize_answer(final) == normalize_answer(pred_text))


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

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU for evaluation.")
        device = "cpu"

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
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)
    if device.startswith("cuda"):
        model = model.half()
    return model.eval().to(device), tokenizer, device


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
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated_ids[0][len(inputs["input_ids"][0]) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _write_debug_predictions(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    if not output.parent.exists():
        print(f"Debug output parent does not exist, skipped writing: {output.parent}")
        return
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_math(
    input_path: str | Path,
    *,
    config: dict[str, Any],
    mode: str,
    sample: bool = False,
    output_path: str | Path | None = None,
    debug: bool = False,
    debug_samples: int | None = None,
    debug_output_path: str | Path | None = None,
) -> dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    debug_cfg = config.get("debug", {})
    answer_prefix = str(eval_cfg.get("answer_prefix", "答案是"))
    rows = read_jsonl(input_path)

    checkpoint = _checkpoint_path(config, mode)
    use_sample_answers = sample and not checkpoint.exists()
    model = tokenizer = None
    device = config.get("environment", {}).get("default_device", "cuda:0")
    if use_sample_answers:
        print(f"Sample mode: checkpoint not found, evaluating stored assistant answers: {checkpoint}")
    else:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {mode}: {checkpoint}")
        model, tokenizer, device = _load_model(config, mode, device)

    max_new_tokens = int(eval_cfg.get("max_new_tokens", 256))
    temperature = float(eval_cfg.get("temperature", 0.1))
    top_p = float(eval_cfg.get("top_p", 0.95))

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
        extracted = extract_final_answer(prediction, answer_prefix)
        is_exact = exact_match(extracted, expected)
        is_relaxed = relaxed_match(extracted, expected)
        is_correct = answer_contains(prediction, expected, answer_prefix)
        details.append(
            {
                "question": question,
                "gold_answer": extract_final_answer(expected, answer_prefix),
                "expected": expected,
                "model_output": prediction,
                "prediction": prediction,
                "extracted_answer": extracted,
                "predicted_final_answer": extracted,
                "exact_match": is_exact,
                "relaxed_match": is_relaxed,
                "answer_contains": is_correct,
                "is_correct": is_correct,
                "invalid_output": invalid_output(prediction, answer_prefix),
                "checkpoint_path": str(checkpoint),
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
        "exact_match": sum(item["exact_match"] for item in details) / total if total else 0.0,
        "relaxed_match": sum(item["relaxed_match"] for item in details) / total if total else 0.0,
        "answer_contains": sum(item["answer_contains"] for item in details) / total if total else 0.0,
        "invalid_output_rate": sum(item["invalid_output"] for item in details) / total if total else 0.0,
        "avg_output_length": sum(item["output_length"] for item in details) / total if total else 0.0,
        "avg_latency_seconds": sum(item["latency_seconds"] for item in details) / total if total else 0.0,
    }

    if output_path:
        output = Path(output_path)
        if not output.is_absolute():
            output = REPO_ROOT / output
        if output.parent.exists():
            output.write_text(json.dumps({"report": report, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(f"Report path parent does not exist, skipped writing: {output.parent}")

    if debug:
        count = int(debug_samples or debug_cfg.get("debug_eval_samples", 20))
        debug_path = debug_output_path or debug_cfg.get("save_debug_predictions", "outputs/debug_predictions.jsonl")
        _write_debug_predictions(debug_path, details[:count])
        report["debug_predictions"] = str(debug_path)
        report["debug_samples"] = min(count, total)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal MiniMind math evaluation.")
    parser.add_argument("--config", default="configs/math_tutor.yaml")
    parser.add_argument("--mode", choices=["official_sft", "math_sft"], default="math_sft")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample", action="store_true", help="Use configured sample data; falls back to stored answers if the checkpoint is absent.")
    parser.add_argument("--debug", action="store_true", help="Write the first configured debug predictions to outputs/debug_predictions.jsonl.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    eval_cfg = config.get("evaluation", {})
    input_path = args.input or (eval_cfg.get("sample_questions") if args.sample else eval_cfg.get("test_file"))
    if not input_path:
        raise ValueError("No evaluation input configured. Pass --input or set evaluation.test_file.")
    output_path = args.output if args.output is not None else eval_cfg.get("report_path")
    report = evaluate_math(
        input_path,
        config=config,
        mode=args.mode,
        sample=args.sample,
        output_path=output_path,
        debug=args.debug,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
