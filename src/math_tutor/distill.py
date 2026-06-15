"""Black-box candidate-level distillation helpers for MiniMind-MathTutor."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

from .data import read_jsonl
from .eval import answer_contains, extract_final_answer, relaxed_match


JudgeFn = Callable[[str, str, str], float]


def build_candidate_set(row: dict[str, Any], *, k: int = 4, qwen_judge: JudgeFn | None = None) -> dict[str, Any]:
    """Build K candidate answers without using teacher logits or token alignment."""

    question = _question_from_row(row)
    gold = _gold_from_row(row)
    teacher = _teacher_answer_from_row(row)
    mini = _mini_answer_from_row(row)
    standard = _standard_answer(gold)
    wrong = _perturb_answer(gold or teacher or standard)

    candidates = [
        {"source": "qwen_teacher", "text": teacher},
        {"source": "minimind_current", "text": mini},
        {"source": "gold_standard", "text": standard},
        {"source": "perturbed_wrong", "text": wrong},
    ][:k]

    scored = [score_candidate(question, candidate, gold, qwen_judge=qwen_judge) for candidate in candidates]
    return {
        "id": row.get("id", ""),
        "question": question,
        "gold_final_answer": extract_final_answer(gold),
        "candidates": scored,
    }


def score_candidate(
    question: str,
    candidate: dict[str, Any],
    gold: str,
    *,
    qwen_judge: JudgeFn | None = None,
) -> dict[str, Any]:
    """Score a candidate with gold-answer rules, optionally overridden by a Qwen judge."""

    text = str(candidate.get("text") or "")
    rule_correct = bool(gold) and answer_contains(text, gold)
    format_bonus = 0.1 if ("答案是" in text or "####" in text or re.search(r"answer\s+is", text, re.I)) else 0.0
    clarity_bonus = 0.1 if len(text.strip()) >= 16 and "\n" in text else 0.0
    rule_score = (1.0 if rule_correct else 0.0) + format_bonus + clarity_bonus
    judge_score = None
    if qwen_judge is not None:
        judge_score = float(qwen_judge(question, text, gold))
    score = judge_score if judge_score is not None else rule_score
    return {
        **candidate,
        "score": score,
        "rule_correct": rule_correct,
        "extracted_answer": extract_final_answer(text),
        "score_source": "qwen_judge" if judge_score is not None else "rule_based_mock",
    }


def candidate_set_to_preference_pair(candidate_set: dict[str, Any]) -> dict[str, Any]:
    """Convert scored candidates into a DPO/ranking-friendly chosen/rejected pair."""

    candidates = list(candidate_set.get("candidates") or [])
    if len(candidates) < 2:
        raise ValueError("Need at least two candidates to build a preference pair.")
    ranked = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    chosen = ranked[0]
    rejected = ranked[-1]
    return {
        "id": candidate_set.get("id", ""),
        "question": candidate_set.get("question", ""),
        "gold_final_answer": candidate_set.get("gold_final_answer", ""),
        "chosen": [
            {"role": "user", "content": candidate_set.get("question", "")},
            {"role": "assistant", "content": chosen.get("text", "")},
        ],
        "rejected": [
            {"role": "user", "content": candidate_set.get("question", "")},
            {"role": "assistant", "content": rejected.get("text", "")},
        ],
        "chosen_source": chosen.get("source", ""),
        "rejected_source": rejected.get("source", ""),
        "chosen_score": chosen.get("score", 0.0),
        "rejected_score": rejected.get("score", 0.0),
    }


def build_black_box_distillation(
    input_path: str | Path,
    output_path: str | Path,
    *,
    output_format: str = "preferences",
    limit: int | None = None,
) -> int:
    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]
    candidate_sets = [build_candidate_set(row) for row in rows]
    if output_format == "candidates":
        output_rows = candidate_sets
    elif output_format == "preferences":
        output_rows = [candidate_set_to_preference_pair(item) for item in candidate_sets]
    else:
        raise ValueError("output_format must be 'candidates' or 'preferences'.")
    _write_jsonl_no_mkdir(output_path, output_rows)
    return len(output_rows)


def _question_from_row(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    for message in conversations:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return str(row.get("question") or row.get("problem") or row.get("input") or row.get("query") or "").strip()


def _teacher_answer_from_row(row: dict[str, Any]) -> str:
    for key in ("teacher_answer", "qwen_answer", "assistant", "solution", "output", "response"):
        value = row.get(key)
        if value:
            return str(value).strip()
    conversations = row.get("conversations") or []
    for message in conversations:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return _standard_answer(_gold_from_row(row))


def _mini_answer_from_row(row: dict[str, Any]) -> str:
    for key in ("mini_answer", "model_output", "prediction"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _gold_from_row(row: dict[str, Any]) -> str:
    for key in ("final_answer", "gold_answer", "answer"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _standard_answer(gold: str) -> str:
    final = extract_final_answer(gold)
    return f"答案是：{final}" if final else ""


def _perturb_answer(text: str) -> str:
    final = extract_final_answer(text)
    match = re.search(r"-?\d+(?:\.\d+)?", final)
    if match:
        value = match.group(0)
        try:
            wrong = str(int(float(value)) + 1)
        except ValueError:
            wrong = f"{value}1"
        return f"答案是：{wrong}"
    if relaxed_match(final, text):
        return f"{final} wrong"
    return "答案是：0"


def _write_jsonl_no_mkdir(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    if not output.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare black-box candidate-level distillation data; no token-level KL or training is run."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/math_blackbox_preferences.jsonl")
    parser.add_argument("--format", choices=["preferences", "candidates"], default="preferences")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    count = build_black_box_distillation(
        args.input,
        args.output,
        output_format=args.format,
        limit=args.limit,
    )
    print(f"Wrote {count} black-box {args.format} records to {args.output}")


if __name__ == "__main__":
    main()
