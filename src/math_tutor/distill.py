"""Candidate-answer distillation helpers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .data import read_jsonl, write_jsonl


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [score / temperature for score in scores]
    offset = max(scaled)
    exps = [math.exp(score - offset) for score in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def build_candidate_distribution(sample: dict[str, Any], temperature: float = 1.0) -> dict[str, Any]:
    candidates = sample.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Each sample must contain a non-empty candidates list.")
    scores = [float(item.get("teacher_score", item.get("score", 0.0))) for item in candidates]
    probs = softmax(scores, temperature=temperature)
    return {
        "id": sample.get("id", ""),
        "question": sample.get("question", ""),
        "candidates": [
            {"text": item.get("text", item.get("answer", "")), "teacher_prob": prob}
            for item, prob in zip(candidates, probs)
        ],
    }


def convert_candidates(input_path: str | Path, output_path: str | Path, temperature: float = 1.0) -> int:
    rows = read_jsonl(input_path)
    converted = [build_candidate_distribution(row, temperature=temperature) for row in rows]
    write_jsonl(output_path, converted)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare candidate-answer probability records for later KL training.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/math_candidate_kl.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    count = convert_candidates(args.input, args.output, args.temperature)
    print(f"Wrote {count} candidate KL records to {args.output}")


if __name__ == "__main__":
    main()
