"""Small local RAG index builder for notes and samples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def build_index(docs_dir: str | Path, output_path: str | Path, chunk_size: int = 700, overlap: int = 120) -> int:
    docs_root = Path(docs_dir)
    records: list[dict[str, Any]] = []
    for doc_path in sorted(docs_root.glob("**/*.md")):
        text = doc_path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            records.append(
                {
                    "id": f"{doc_path.as_posix()}#{idx}",
                    "source": doc_path.as_posix(),
                    "text": chunk,
                    "terms": sorted(set(tokenize(chunk))),
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"chunks": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny JSON lexical RAG index from markdown notes.")
    parser.add_argument("--docs-dir", default="sample_docs")
    parser.add_argument("--output", default="outputs/rag/math_notes_sample.index.json")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    count = build_index(args.docs_dir, args.output, args.chunk_size, args.overlap)
    print(f"Indexed {count} chunks to {args.output}")


if __name__ == "__main__":
    main()
