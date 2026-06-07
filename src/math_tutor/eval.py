"""Simple evaluation helpers for math QA outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .data import read_jsonl


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("final answer:", "").strip()
    return text


def score_prediction(prediction: str, answer: str) -> dict[str, Any]:
    pred_norm = normalize_answer(prediction)
    answer_norm = normalize_answer(answer)
    return {
        "exact_match": pred_norm == answer_norm,
        "contains_answer": bool(answer_norm) and answer_norm in pred_norm,
    }


def evaluate_predictions(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    scored = []
    for row in rows:
        scored.append(score_prediction(str(row.get("prediction", "")), str(row.get("answer", ""))))
    total = len(scored)
    report = {
        "total": total,
        "exact_match": sum(item["exact_match"] for item in scored) / total if total else 0.0,
        "contains_answer": sum(item["contains_answer"] for item in scored) / total if total else 0.0,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL rows with prediction and answer fields.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/eval/math_eval_report.json")
    args = parser.parse_args()

    report = evaluate_predictions(args.input, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
