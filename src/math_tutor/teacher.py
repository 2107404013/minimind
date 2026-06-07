"""Teacher-data utilities.

This module intentionally does not download or load Qwen2.5-Math-7B-Instruct.
It only prepares prompt records and validates saved teacher responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import read_jsonl, write_jsonl


TEACHER_PROMPT = (
    "Solve the following math problem. Give a clear step-by-step solution and "
    "finish with 'Final answer: ...'.\n\nProblem:\n{question}"
)


def build_teacher_requests(input_path: str | Path, output_path: str | Path) -> int:
    rows = read_jsonl(input_path)
    requests: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = str(row.get("question") or row.get("problem") or row.get("prompt") or "").strip()
        if not question:
            raise ValueError(f"Row {idx} has no question/problem/prompt field.")
        requests.append(
            {
                "id": row.get("id", f"sample-{idx}"),
                "question": question,
                "prompt": TEACHER_PROMPT.format(question=question),
                "teacher_model": "Qwen2.5-Math-7B-Instruct",
                "status": "pending",
            }
        )
    write_jsonl(output_path, requests)
    return len(requests)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare teacher-model generation requests without loading a model.")
    parser.add_argument("--input", default="sample_data/math_raw_sample.jsonl")
    parser.add_argument("--output", default="outputs/teacher/qwen_math_requests.jsonl")
    args = parser.parse_args()

    count = build_teacher_requests(args.input, args.output)
    print(f"Wrote {count} teacher request records to {args.output}")


if __name__ == "__main__":
    main()
