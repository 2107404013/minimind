"""Math data conversion helpers for MiniMind-MathTutor."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


QUESTION_FIELDS = ("question", "problem", "input", "query")
ANSWER_FIELDS = ("answer", "solution", "output", "response")
FINAL_ANSWER_FIELDS = ("final_answer", "answer")
DEFAULT_USER_TEMPLATE = "请解答下面的数学题，并给出清晰的解题步骤：\n{question}"
DEFAULT_FINAL_ANSWER_PREFIX = "答案是："


@dataclass(frozen=True)
class DataBuildStats:
    read: int = 0
    converted: int = 0
    skipped_empty: int = 0
    skipped_duplicate: int = 0
    skipped_too_short: int = 0
    skipped_too_long: int = 0
    train: int = 0
    valid: int = 0
    test: int = 0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_no} is not an object.")
            rows.append(row)
    return rows


def read_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = _records_from_json_object(payload)
    else:
        raise ValueError(f"JSON file {path} must contain an object or a list.")

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSON file {path} contains non-object records.")
    return list(rows)


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(input_path)
    if suffix == ".json":
        return read_json(input_path)
    raise ValueError(f"Unsupported data file extension: {input_path.suffix}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_official_sft_record(row: dict[str, Any]) -> dict[str, Any]:
    """Return the narrow schema expected by MiniMind's official SFTDataset."""

    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Official SFT record requires a conversations list.")

    official_messages: list[dict[str, str]] = []
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("Each conversation message must be an object.")
        official_messages.append(
            {
                "role": _optional_text(message.get("role")),
                "content": _optional_text(message.get("content")),
                "reasoning_content": _optional_text(message.get("reasoning_content")),
                "tools": _optional_text(message.get("tools")),
                "tool_calls": _optional_text(message.get("tool_calls")),
            }
        )
    return {"conversations": official_messages}


def to_official_sft_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_official_sft_record(row) for row in rows]


def format_math_sft_record(
    row: dict[str, Any],
    *,
    final_answer_prefix: str = DEFAULT_FINAL_ANSWER_PREFIX,
) -> dict[str, Any] | None:
    """Normalize an existing SFT row to end with a stable final-answer marker."""

    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None

    user_text = ""
    assistant_text = ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = _optional_text(message.get("content"))
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant" and not assistant_text:
            assistant_text = content

    if not user_text or not assistant_text:
        return None

    final_answer = _optional_text(row.get("final_answer")) or _extract_final_answer_for_format(assistant_text)
    if not final_answer:
        return None

    assistant_text = _append_final_answer_marker(assistant_text, final_answer, final_answer_prefix)
    formatted = {
        "conversations": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "final_answer": final_answer,
    }
    for key in ("source", "level", "type"):
        value = row.get(key)
        if value not in (None, ""):
            formatted[key] = value
    return formatted


def format_math_sft_records(
    rows: Iterable[dict[str, Any]],
    *,
    final_answer_prefix: str = DEFAULT_FINAL_ANSWER_PREFIX,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    row_list = list(rows)
    formatted: list[dict[str, Any]] = []
    skipped = 0
    for row in row_list:
        item = format_math_sft_record(row, final_answer_prefix=final_answer_prefix)
        if item is None:
            skipped += 1
            continue
        formatted.append(item)
    return formatted, {"read": len(row_list), "formatted": len(formatted), "skipped": skipped}


def format_math_sft_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    final_answer_prefix: str = DEFAULT_FINAL_ANSWER_PREFIX,
) -> dict[str, int | str]:
    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]
    formatted, stats = format_math_sft_records(rows, final_answer_prefix=final_answer_prefix)
    write_jsonl(output_path, formatted)
    return {**stats, "input": str(input_path), "output": str(output_path)}


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs/math_tutor.yaml.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def raw_math_to_sft(
    sample: dict[str, Any],
    user_template: str = DEFAULT_USER_TEMPLATE,
) -> dict[str, Any] | None:
    """Convert one raw math record to MiniMind conversations format."""

    question = _first_text(sample, QUESTION_FIELDS)
    assistant = _assistant_text(sample)
    if not question or not assistant:
        return None

    final_answer = _first_text(sample, FINAL_ANSWER_FIELDS)
    source = _optional_text(sample.get("source")) or "unknown"
    level = _optional_text(sample.get("level"))
    problem_type = _optional_text(sample.get("type"))

    row: dict[str, Any] = {
        "conversations": [
            {
                "role": "user",
                "content": user_template.format(question=question),
            },
            {
                "role": "assistant",
                "content": assistant,
            },
        ],
        "source": source,
        "level": level,
        "type": problem_type,
        "final_answer": final_answer,
    }
    return {key: value for key, value in row.items() if value not in (None, "")}


def convert_records(
    rows: Sequence[dict[str, Any]],
    *,
    user_template: str = DEFAULT_USER_TEMPLATE,
    min_question_chars: int = 1,
    min_answer_chars: int = 1,
    max_question_chars: int = 2048,
    max_answer_chars: int = 4096,
    max_total_chars: int = 6144,
) -> tuple[list[dict[str, Any]], DataBuildStats]:
    converted: list[dict[str, Any]] = []
    seen: set[str] = set()
    counters = {
        "skipped_empty": 0,
        "skipped_duplicate": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
    }

    for row in rows:
        item = raw_math_to_sft(row, user_template=user_template)
        if item is None:
            counters["skipped_empty"] += 1
            continue

        question = item["conversations"][0]["content"]
        answer = item["conversations"][1]["content"]
        if len(question) < min_question_chars or len(answer) < min_answer_chars:
            counters["skipped_too_short"] += 1
            continue
        if (
            len(question) > max_question_chars
            or len(answer) > max_answer_chars
            or len(question) + len(answer) > max_total_chars
        ):
            counters["skipped_too_long"] += 1
            continue

        fingerprint = _fingerprint(question, answer)
        if fingerprint in seen:
            counters["skipped_duplicate"] += 1
            continue
        seen.add(fingerprint)
        converted.append(item)

    stats = DataBuildStats(
        read=len(rows),
        converted=len(converted),
        skipped_empty=counters["skipped_empty"],
        skipped_duplicate=counters["skipped_duplicate"],
        skipped_too_short=counters["skipped_too_short"],
        skipped_too_long=counters["skipped_too_long"],
    )
    return converted, stats


def split_records(
    rows: Sequence[dict[str, Any]],
    *,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], [], []
    total = train_ratio + valid_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train_cut = int(len(shuffled) * train_ratio / total)
    valid_cut = train_cut + int(len(shuffled) * valid_ratio / total)
    return shuffled[:train_cut], shuffled[train_cut:valid_cut], shuffled[valid_cut:]


def build_math_data(
    input_path: str | Path,
    output_path: str | Path,
    *,
    user_template: str = DEFAULT_USER_TEMPLATE,
    filters: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    split_paths: dict[str, str | None] | None = None,
    official_sft_compatible: bool = False,
) -> DataBuildStats:
    rows = read_json_records(input_path)
    filters = filters or {}
    converted, stats = convert_records(rows, user_template=user_template, **filters)
    output_rows = to_official_sft_records(converted) if official_sft_compatible else converted
    write_jsonl(output_path, output_rows)

    if split_paths:
        split = split or {}
        train, valid, test = split_records(
            converted,
            train_ratio=float(split.get("train_ratio", 0.9)),
            valid_ratio=float(split.get("valid_ratio", 0.05)),
            test_ratio=float(split.get("test_ratio", 0.05)),
            seed=int(split.get("seed", 42)),
        )
        if split_paths.get("train"):
            write_jsonl(split_paths["train"], to_official_sft_records(train) if official_sft_compatible else train)
        if split_paths.get("valid"):
            write_jsonl(split_paths["valid"], to_official_sft_records(valid) if official_sft_compatible else valid)
        if split_paths.get("test"):
            write_jsonl(split_paths["test"], to_official_sft_records(test) if official_sft_compatible else test)
        stats = DataBuildStats(**{**stats.__dict__, "train": len(train), "valid": len(valid), "test": len(test)})

    return stats


def diagnose_sft_file(
    input_path: str | Path,
    *,
    tokenizer: Any | None = None,
    max_seq_len: int = 768,
    limit: int | None = None,
) -> dict[str, Any]:
    """Inspect MiniMind conversation data without changing the training file."""

    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]

    total_tokens = 0
    max_tokens = 0
    truncated = 0
    question_tokens: list[int] = []
    assistant_tokens: list[int] = []
    final_answer_present = 0
    conversations_valid = 0
    assistant_messages = 0
    answer_marker_count = 0
    examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        conversations = row.get("conversations")
        valid_conversation = isinstance(conversations, list) and len(conversations) >= 2
        if valid_conversation:
            conversations_valid += 1
        user_text = ""
        assistant_text = ""
        if isinstance(conversations, list):
            for message in conversations:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "user" and not user_text:
                    user_text = _optional_text(message.get("content"))
                if message.get("role") == "assistant" and not assistant_text:
                    assistant_text = _optional_text(message.get("content"))
        if assistant_text:
            assistant_messages += 1
        if row.get("final_answer"):
            final_answer_present += 1
        if "答案是" in assistant_text or "####" in assistant_text or "answer is" in assistant_text.lower():
            answer_marker_count += 1

        prompt_text = _conversation_text(conversations) if isinstance(conversations, list) else json.dumps(row, ensure_ascii=False)
        token_count = _token_count(prompt_text, tokenizer)
        q_tokens = _token_count(user_text, tokenizer)
        a_tokens = _token_count(assistant_text, tokenizer)
        total_tokens += token_count
        max_tokens = max(max_tokens, token_count)
        question_tokens.append(q_tokens)
        assistant_tokens.append(a_tokens)
        if token_count > max_seq_len:
            truncated += 1
        if len(examples) < 3:
            examples.append(
                {
                    "index": index,
                    "token_length": token_count,
                    "question_tokens": q_tokens,
                    "assistant_tokens": a_tokens,
                    "has_final_answer": bool(row.get("final_answer")),
                    "question_preview": user_text[:160],
                    "assistant_preview": assistant_text[:200],
                }
            )

    count = len(rows)
    return {
        "path": str(input_path),
        "records": count,
        "conversations_valid": conversations_valid,
        "assistant_messages": assistant_messages,
        "final_answer_present": final_answer_present,
        "missing_final_answer": count - final_answer_present,
        "answer_marker_count": answer_marker_count,
        "max_seq_len": max_seq_len,
        "avg_total_tokens": total_tokens / count if count else 0.0,
        "max_total_tokens": max_tokens,
        "truncated_count": truncated,
        "truncated_rate": truncated / count if count else 0.0,
        "avg_question_tokens": sum(question_tokens) / count if count else 0.0,
        "avg_assistant_tokens": sum(assistant_tokens) / count if count else 0.0,
        "examples": examples,
    }


def _records_from_json_object(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("data", "records", "examples", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return [payload]


def _conversation_text(conversations: list[Any]) -> str:
    pieces: list[str] = []
    for message in conversations:
        if isinstance(message, dict):
            pieces.append(f"{message.get('role', '')}\n{message.get('content', '')}")
    return "\n".join(pieces)


def _token_count(text: str, tokenizer: Any | None) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return len(str(text))
    try:
        return len(tokenizer(str(text), add_special_tokens=False).input_ids)
    except Exception:
        return len(str(text))


def _first_text(sample: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _optional_text(sample.get(key))
        if value:
            return value
    return ""


def _assistant_text(sample: dict[str, Any]) -> str:
    solution = _first_text(sample, ("solution", "response", "output"))
    answer = _first_text(sample, ("answer",))
    if solution and answer and "答案是" not in solution and "Final answer" not in solution:
        return f"{solution}\n\n答案是：{answer}"
    if solution:
        return solution
    if answer:
        return f"答案是：{answer}"
    return ""


def _extract_final_answer_for_format(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    boxed_matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", raw)
    if boxed_matches:
        return _clean_final_answer_fragment(boxed_matches[-1])

    patterns = [
        r"####\s*([^\n\r]+)",
        r"答案是\s*[:：]?\s*([^\n\r]+)",
        r"(?:therefore,\s*)?the\s+answer\s+is\s*[:：]?\s*([^\n\r]+)",
        r"final\s+answer\s*[:：]?\s*([^\n\r]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, raw, re.I))
        if matches:
            return _clean_final_answer_fragment(matches[-1].group(1))

    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?%?", raw.replace(",", ""))
    return numbers[-1].strip() if numbers else ""


def _clean_final_answer_fragment(text: str) -> str:
    value = str(text or "").strip()
    value = re.split(r"[\n\r]", value, maxsplit=1)[0]
    value = re.split(r"[。；;]", value, maxsplit=1)[0]
    value = value.strip(" \t。.,，：:")
    return value


def _append_final_answer_marker(text: str, final_answer: str, final_answer_prefix: str) -> str:
    body = str(text or "").strip()
    marker = f"{final_answer_prefix}{final_answer}"
    if marker in body[-120:]:
        return body
    return f"{body}\n\n{marker}"


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _fingerprint(question: str, answer: str) -> str:
    return f"{_normalize(question)}\n{_normalize(answer)}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MiniMind math SFT data from raw JSON/JSONL.")
    parser.add_argument("--config", default="configs/math_tutor.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample", action="store_true", help="Use sample_data paths and do not write train/valid/test files.")
    parser.add_argument("--write-splits", action="store_true", help="Write configured train/valid/test split files.")
    parser.add_argument("--format-sft-final", action="store_true", help="Normalize an existing SFT JSONL file with final_answer and 答案是： markers.")
    parser.add_argument("--limit", type=int, default=None, help="Limit records for formatting/debug data preparation.")
    parser.add_argument(
        "--official-sft-compatible",
        action="store_true",
        help="Write the narrow conversations schema expected by MiniMind trainer/train_full_sft.py.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_cfg = config.get("data", {})
    paths_cfg = config.get("paths", {})

    input_path = args.input or (
        data_cfg.get("sample_raw_path") if args.sample else data_cfg.get("raw_path")
    ) or paths_cfg.get("math_raw_sample", "sample_data/math_raw_sample.jsonl")
    output_path = args.output or (
        data_cfg.get("sample_sft_path") if args.sample else data_cfg.get("sft_path")
    ) or paths_cfg.get("math_sft_sample", "sample_data/math_sft_sample.jsonl")

    if args.format_sft_final:
        stats = format_math_sft_file(
            input_path,
            output_path,
            limit=args.limit,
            final_answer_prefix=data_cfg.get("final_answer_prefix", DEFAULT_FINAL_ANSWER_PREFIX),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    split_paths = None
    if args.write_splits and not args.sample:
        split_paths = {
            "train": _config_value(data_cfg, "split_paths", "train"),
            "valid": _config_value(data_cfg, "split_paths", "valid"),
            "test": _config_value(data_cfg, "split_paths", "test"),
        }

    stats = build_math_data(
        input_path,
        output_path,
        user_template=data_cfg.get("user_template", DEFAULT_USER_TEMPLATE),
        filters=data_cfg.get("filters", {}),
        split=data_cfg.get("split", {}),
        split_paths=split_paths,
        official_sft_compatible=args.official_sft_compatible,
    )
    print(
        "Built math data: "
        f"read={stats.read}, converted={stats.converted}, "
        f"empty={stats.skipped_empty}, duplicate={stats.skipped_duplicate}, "
        f"too_short={stats.skipped_too_short}, too_long={stats.skipped_too_long}, "
        f"output={output_path}"
    )
    if split_paths:
        print(f"Splits: train={stats.train}, valid={stats.valid}, test={stats.test}")


if __name__ == "__main__":
    main()
