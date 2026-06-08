"""Qwen teacher generation utilities for MiniMind-MathTutor."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .data import DEFAULT_USER_TEMPLATE, load_yaml, read_json_records, read_jsonl, to_official_sft_record, write_jsonl


TEACHER_INSTRUCTION = "你是一名严谨的数学老师。请一步一步推理，最后用‘答案是：...’给出最终答案。"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DEFAULT_FAILED_PATH = "outputs/failed_teacher.jsonl"


def generate_teacher_solutions(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    load_in_4bit: bool = False,
    limit: int | None = None,
    resume: bool = True,
    failed_path: str | Path = DEFAULT_FAILED_PATH,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.95,
    local_files_only: bool = True,
    official_sft_compatible: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
) -> int:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")

    rows = read_json_records(input_path)
    existing_keys = _existing_keys(output_path) if resume else set()
    mode = "a" if resume and Path(output_path).exists() else "w"
    model, tokenizer = load_qwen_teacher(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
    )

    generated = 0
    with Path(output_path).open(mode, encoding="utf-8", newline="\n") as output_handle:
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            if index % num_shards != shard_index:
                continue

            question = extract_question(row)
            if not question:
                _append_failed(failed_path, row, "missing question", index)
                continue

            key = _record_key(row, question)
            if key in existing_keys:
                continue

            try:
                answer = generate_one(
                    model,
                    tokenizer,
                    question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                record = build_teacher_record(row, question, answer, model_name_or_path)
                if official_sft_compatible:
                    record = to_official_sft_record(record)
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                existing_keys.add(key)
                generated += 1
            except Exception as exc:  # noqa: BLE001 - each failed source row must be recorded.
                _append_failed(failed_path, row, repr(exc), index)

    return generated


def load_qwen_teacher(
    model_name_or_path: str,
    *,
    load_in_4bit: bool = False,
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and torch are required for Qwen teacher generation.") from exc

    quantization_config = None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4bit loading requires transformers BitsAndBytesConfig and bitsandbytes.") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
        quantization_config=quantization_config,
    )
    model.eval()
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    question: str,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    messages = [
        {"role": "system", "content": TEACHER_INSTRUCTION},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = temperature > 0
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    outputs = model.generate(
        **generation_kwargs,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_teacher_record(source_row: dict[str, Any], question: str, answer: str, model_name: str) -> dict[str, Any]:
    final_answer = extract_final_answer(answer) or _optional_text(source_row.get("final_answer"))
    record: dict[str, Any] = {
        "conversations": [
            {
                "role": "user",
                "content": DEFAULT_USER_TEMPLATE.format(question=question),
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "source": _optional_text(source_row.get("source")) or "qwen_teacher",
        "level": _optional_text(source_row.get("level")),
        "type": _optional_text(source_row.get("type")),
        "final_answer": final_answer,
        "teacher_model": model_name,
    }
    return {key: value for key, value in record.items() if value not in (None, "")}


def extract_question(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if isinstance(message, dict) and message.get("role") == "user":
                return _strip_question_prefix(_optional_text(message.get("content")))
    for key in ("question", "problem", "input", "query"):
        value = _optional_text(row.get(key))
        if value:
            return value
    return ""


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"答案是[:：]\s*(.+)", text)
    if not matches:
        return ""
    return matches[-1].strip()


def _existing_keys(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    keys: set[str] = set()
    for row in read_jsonl(output):
        question = extract_question(row)
        if question:
            keys.add(_record_key(row, question))
    return keys


def _append_failed(path: str | Path, row: dict[str, Any], error: str, index: int) -> None:
    failure = {
        "index": index,
        "error": error,
        "row": row,
    }
    failed = Path(path)
    failed.parent.mkdir(parents=True, exist_ok=True)
    with failed.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(failure, ensure_ascii=False) + "\n")


def _record_key(row: dict[str, Any], question: str) -> str:
    raw_id = _optional_text(row.get("id") or row.get("uid"))
    return raw_id or re.sub(r"\s+", " ", question).strip().lower()


def _strip_question_prefix(text: str) -> str:
    marker = "请解答下面的数学题，并给出清晰的解题步骤："
    if text.startswith(marker):
        return text[len(marker) :].strip()
    return text.strip()


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def build_teacher_requests(input_path: str | Path, output_path: str | Path) -> int:
    """Compatibility helper: write prompts without loading a model."""

    rows = read_json_records(input_path)
    requests: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = extract_question(row)
        if not question:
            raise ValueError(f"Row {index} has no question/problem/input/query field.")
        requests.append(
            {
                "id": row.get("id", f"sample-{index}"),
                "question": question,
                "prompt": f"{TEACHER_INSTRUCTION}\n\n题目：\n{question}",
                "teacher_model": DEFAULT_MODEL_NAME,
                "status": "pending",
            }
        )
    write_jsonl(output_path, requests)
    return len(requests)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen teacher answers in MiniMind conversations format.")
    parser.add_argument("--config", default="configs/math_tutor.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow transformers to download/cache model files.")
    parser.add_argument("--prepare-only", action="store_true", help="Write teacher prompt request records without loading Qwen.")
    parser.add_argument(
        "--official-sft-compatible",
        action="store_true",
        help="Write output directly consumable by MiniMind trainer/train_full_sft.py.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Split input rows across this many teacher workers.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current teacher worker index in [0, num_shards).")
    args = parser.parse_args()

    config = load_yaml(args.config)
    teacher_cfg = config.get("teacher", {})
    input_path = args.input or teacher_cfg.get("input_path") or "sample_data/math_sft_sample.jsonl"
    output_path = args.output or teacher_cfg.get("output_path") or "outputs/sample_teacher.jsonl"

    if args.prepare_only:
        count = build_teacher_requests(input_path, output_path)
        print(f"Wrote {count} teacher request records to {output_path}")
        return

    try:
        generated = generate_teacher_solutions(
            input_path,
            output_path,
            model_name_or_path=args.model or teacher_cfg.get("model_name_or_path", DEFAULT_MODEL_NAME),
            load_in_4bit=args.load_in_4bit or bool(teacher_cfg.get("load_in_4bit", False)),
            limit=args.limit if args.limit is not None else teacher_cfg.get("limit"),
            resume=not args.no_resume and bool(teacher_cfg.get("resume", True)),
            failed_path=teacher_cfg.get("failed_path", DEFAULT_FAILED_PATH),
            max_new_tokens=int(teacher_cfg.get("max_new_tokens", 1024)),
            temperature=float(teacher_cfg.get("temperature", 0.2)),
            top_p=float(teacher_cfg.get("top_p", 0.95)),
            local_files_only=not args.allow_download and bool(teacher_cfg.get("local_files_only", True)),
            official_sft_compatible=args.official_sft_compatible or bool(teacher_cfg.get("official_sft_compatible", False)),
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Generated {generated} teacher records to {output_path}")


if __name__ == "__main__":
    main()
