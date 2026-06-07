"""Data conversion helpers for MiniMind-MathTutor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math tutor. Explain the solution clearly and end with "
    "a concise final answer."
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_text(sample: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def raw_math_to_sft(sample: dict[str, Any], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> dict[str, Any]:
    """Convert a simple math QA record to MiniMind SFT conversations format."""

    question = _first_text(sample, ("question", "problem", "prompt", "instruction", "input"))
    solution = _first_text(sample, ("solution", "analysis", "explanation", "rationale"))
    answer = _first_text(sample, ("answer", "final_answer", "target", "output"))

    if not question:
        raise ValueError("Sample is missing a question/problem/prompt field.")
    if not answer and not solution:
        raise ValueError("Sample is missing an answer or solution field.")

    assistant_parts = []
    if solution:
        assistant_parts.append(solution)
    if answer:
        assistant_parts.append(f"Final answer: {answer}")

    return {
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "\n\n".join(assistant_parts)},
        ],
        "meta": {
            "source": sample.get("source", "unknown"),
            "id": sample.get("id", sample.get("uid", "")),
        },
    }


def convert_raw_jsonl(input_path: str | Path, output_path: str | Path, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> int:
    rows = read_jsonl(input_path)
    converted = [raw_math_to_sft(row, system_prompt=system_prompt) for row in rows]
    write_jsonl(output_path, converted)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw math QA JSONL to MiniMind SFT conversations JSONL.")
    parser.add_argument("--input", default="sample_data/math_raw_sample.jsonl")
    parser.add_argument("--output", default="sample_data/math_sft_sample.jsonl")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    count = convert_raw_jsonl(args.input, args.output, args.system_prompt)
    print(f"Converted {count} records to {args.output}")


if __name__ == "__main__":
    main()
